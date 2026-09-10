import orbax.checkpoint as ocp
import os
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
import optax
import re
import copy
from functools import partial
from pathlib import Path

def is_jax_mesh_active():
    """Return True if a JAX device mesh context is currently active."""
    try:
        mesh = jax.sharding.get_abstract_mesh()
        return mesh is not None and len(mesh.shape) > 0
    except Exception:
        return False


def parse_training_stages(cfg):
    """
    Parse and validate training stages from config.

    Args:
        cfg: Hydra config

    Returns:
        List of stage dicts with 'trainable_params' and 'steps' keys,
        or None if no stages configured
    """
    if not cfg.trainer.get("training_stages"):
        return None

    stages = []
    for stage_cfg in cfg.trainer.training_stages:
        stage = {
            'trainable_params': list(stage_cfg['trainable_params']),
            'max_step': stage_cfg['max_step']
        }
        # Per-stage loss weight overrides
        if 'ce_weight' in stage_cfg:
            stage['ce_weight'] = float(stage_cfg['ce_weight'])
        if 'aux_losses' in stage_cfg:
            stage['aux_losses'] = {k: dict(v) for k, v in stage_cfg['aux_losses'].items()}
        # Per-stage LR schedule
        if 'lr_schedule' in stage_cfg:
            stage['lr_schedule'] = str(stage_cfg['lr_schedule'])
        if 'min_lr' in stage_cfg:
            stage['min_lr'] = float(stage_cfg['min_lr'])
        if 'warmup_frac' in stage_cfg:
            stage['warmup_frac'] = float(stage_cfg['warmup_frac'])
        # Absolute warmup step count. If set, takes precedence over warmup_frac —
        # deterministic warmup means changing trainer.steps doesn't change the LR
        # schedule shape (a fractional warmup silently scales with total run length
        # and confounds LR ablations across different-length runs).
        if 'warmup_steps' in stage_cfg:
            stage['warmup_steps'] = int(stage_cfg['warmup_steps'])
        # Per-stage peak learning rate (falls back to trainer.learning_rate)
        if 'learning_rate' in stage_cfg:
            stage['learning_rate'] = float(stage_cfg['learning_rate'])
        # WSD-specific keys. decay_steps is REQUIRED whenever lr_schedule='wsd'; the
        # other two default in _build_schedule. Anchoring the cooldown to end-of-stage
        # (i.e. deriving decay_start = stage_duration - decay_steps internally) is the
        # invariant that keeps re-timed stages from silently shipping a truncated
        # cooldown — do NOT let a `decay_start_step` field land here.
        if 'decay_steps' in stage_cfg:
            stage['decay_steps'] = int(stage_cfg['decay_steps'])
        if 'final_lr_frac' in stage_cfg:
            stage['final_lr_frac'] = float(stage_cfg['final_lr_frac'])
        if 'decay_shape' in stage_cfg:
            stage['decay_shape'] = str(stage_cfg['decay_shape'])
        stages.append(stage)

    # Validation
    if not stages:
        return None

    for i, stage in enumerate(stages):
        if i > 0 and stage['max_step'] <= stages[i-1]['max_step']:
            raise ValueError(
                f"Stage {i} max_step ({stage['max_step']}) must be > stage {i-1} max_step ({stages[i-1]['max_step']})"
            )
        if stage.get('lr_schedule') == 'wsd' and 'decay_steps' not in stage:
            raise ValueError(
                f"Stage {i} has lr_schedule='wsd' but 'decay_steps' is required "
                f"(cooldown length in stage-local steps)"
            )

    if stages[-1]['max_step'] != cfg.trainer.steps:
        raise ValueError(
            f"Final stage max_step ({stages[-1]['max_step']}) must equal total steps ({cfg.trainer.steps})"
        )

    return stages

def get_current_stage_idx(step, stages):
    """
    Determine which stage we're in based on current step.

    Args:
        step: Current training step
        stages: List of stage configs from parse_training_stages()

    Returns:
        stage_idx: Index of current stage (0-indexed)
    """
    if stages is None:
        return 0

    for idx, stage in enumerate(stages):
        if step < stage['max_step']:
            return idx
    return len(stages) - 1

def promote_trainable_to_fp32(weights, training_stages):
    """Promote every bf16 trainable weight to fp32 IN-PLACE on the tree.

    Why: optax.adamw with default mu_dtype=None inherits param dtype for BOTH mu and nu
    (optax._src.transform.py:277-278). If params are bf16, mu/nu are bf16, and the update
    add-back `w_bf16 + Δw_bf16` rounds Δw away whenever |Δw| < ULP(|w|). For weights
    initialized near unit magnitude (mem_q_norm, mem_o_norm, mem_layernorm — all ones),
    bf16 ULP is ~7.8e-3, and the LR-1e-4 update is 100× below the threshold: EVERY step
    rounds to zero. Empirically confirmed on
    perperiod_spec_zeroinit_4layer-2026-07-27-15-52-46: those four params showed Δ=0.0
    exact across 22 steps.

    Fix: store trainable weights in fp32, cast to bf16 only inside the forward pass.
    optimizer.init runs AFTER this promotion, so mu/nu are per-leaf fp32 automatically.
    Frozen weights (main 4B, embed_tokens, etc.) stay bf16 — no memory penalty.

    Union across all stages: stage 1 sees only `.*mem_.*` as trainable, but stage 2 unlocks
    `.*embed_model.*`; promoting only stage-1's set would mean stage-2's newly-unlocked
    embed_model leaves hit the identical trap mid-run. Promote the union up front.

    Full-path key construction (not just path[0].key): future refactors that nest the
    weight tree would silently no-op the regex if we relied on top-level key alone —
    exactly the bug class we're fixing. Empirically merge_weights (models/utils.py:13)
    produces a flat dict with dotted keys today, so path[0].key happens to work; using
    the join is degradation-safe.

    Args:
        weights: model.weights, flat dict {dotted_key: jax.Array}
        training_stages: list of stage configs from cfg.trainer.training_stages

    Returns:
        weights with fp32-promoted trainable leaves; frozen leaves unchanged.
    """
    patterns = set()
    for st in training_stages or []:
        for p in (st.get('trainable_params') or []):
            patterns.add(p)
    if not patterns:
        print("[promote_fp32] no training_stages / trainable_params found; skipping")
        return weights

    if "all" in patterns:
        compiled = None  # match everything trainable
    else:
        compiled = [re.compile(p) for p in patterns]

    promoted = []  # (dotted_key, shape, dtype_before)

    def cast_fn(path, x):
        key = ".".join(str(getattr(p, 'key', p)) for p in path)
        matched = (compiled is None) or any(p.search(key) for p in compiled)
        if not matched:
            return x
        # Idempotent: only promote bf16 leaves. Already-fp32 or int leaves pass through.
        if getattr(x, 'dtype', None) == jnp.bfloat16:
            promoted.append((key, tuple(x.shape), 'bfloat16'))
            return x.astype(jnp.float32)
        return x

    weights = jax.tree_util.tree_map_with_path(cast_fn, weights)

    print(f"\n[promote_fp32] promoted {len(promoted)} leaves bf16 -> fp32:")
    total_elems = 0
    large_warn_thresh = int(os.environ.get("PROMOTE_FP32_LARGE_LEAF_WARN", 500_000_000))
    for key, shape, _ in sorted(promoted):
        elems = 1
        for d in shape: elems *= d
        total_elems += elems
        print(f"  {key:<60}  shape={list(shape)}  elems={elems}")
        if elems > large_warn_thresh:
            gb_fp32 = elems * 4 / 1e9
            print(f"  [WARN] {key} is large ({elems} elems, {gb_fp32:.1f} GB fp32). "
                  f"Adam mu+nu double that. Raise PROMOTE_FP32_LARGE_LEAF_WARN if intentional.")
    print(f"[promote_fp32] total promoted params: {total_elems} ({total_elems * 4 / 1e9:.2f} GB fp32; "
          f"+{total_elems * 2 / 1e9:.2f} GB vs bf16). Plus 2x for Adam mu+nu.\n")

    return weights


def setup_optimizer_for_stage(cfg, model, stage_config=None, all_stages=None):
    """
    Setup optimizer for a specific training stage.

    Args:
        cfg: Hydra config
        model: Model
        stage_config: Dict with 'trainable_params' key, or None for default
        all_stages: List of all stage configs (needed to compute stage duration for schedules)

    Returns:
        optax optimizer
    """
    # Build learning rate (constant or scheduled).
    # Peak LR: per-stage 'learning_rate' overrides the global trainer.learning_rate.
    peak_lr = cfg.trainer.learning_rate
    if stage_config is not None and stage_config.get('learning_rate') is not None:
        peak_lr = float(stage_config.get('learning_rate'))
    lr = peak_lr
    lr_schedule_type = stage_config.get('lr_schedule') if stage_config is not None else None
    # Warmup: prefer absolute `warmup_steps` (deterministic), fall back to
    # `warmup_frac`. Fractional warmup silently scales with trainer.steps, so a
    # LR ablation across different-length runs conflates schedule change with LR
    # change. Deterministic warmup keeps the schedule shape constant regardless
    # of run length.
    warmup_frac = stage_config.get('warmup_frac', 0.0) if stage_config is not None else 0.0
    warmup_steps_cfg = stage_config.get('warmup_steps') if stage_config is not None else None

    if lr_schedule_type in ('cosine', 'wsd') or warmup_frac > 0 or warmup_steps_cfg is not None:
        # Compute stage duration from stage boundaries
        stage_max_step = stage_config['max_step'] if stage_config is not None else cfg.trainer.steps
        prev_max_step = 0
        if all_stages is not None and stage_config is not None:
            stage_idx = all_stages.index(stage_config)
            if stage_idx > 0:
                prev_max_step = all_stages[stage_idx - 1]['max_step']
        stage_duration = stage_max_step - prev_max_step
        min_lr = stage_config.get('min_lr', 0.0) if stage_config is not None else 0.0
        if warmup_steps_cfg is not None:
            warmup_steps = int(warmup_steps_cfg)
        elif warmup_frac > 0:
            warmup_steps = max(1, int(stage_duration * warmup_frac))
        else:
            warmup_steps = 0

        # WSD-only knobs. decay_start is DERIVED as stage_duration - decay_steps and
        # never written into the config, so re-timing a stage's boundary can't leave a
        # cooldown that truncates at max_step or starts with the wrong number of steps
        # left. Enforce warmup + decay < stage_duration (strict): equality gives a
        # zero-length stable phase, which is a config error, not a valid WSD run.
        decay_steps = None
        final_lr_frac = 0.1
        decay_shape = 'linear'
        if lr_schedule_type == 'wsd':
            if stage_config is None or 'decay_steps' not in stage_config:
                raise ValueError("lr_schedule='wsd' requires 'decay_steps' in the stage config")
            decay_steps = int(stage_config['decay_steps'])
            final_lr_frac = float(stage_config.get('final_lr_frac', 0.1))
            decay_shape = str(stage_config.get('decay_shape', 'linear'))
            if decay_shape not in ('linear', 'one_minus_sqrt'):
                raise ValueError(
                    f"decay_shape must be 'linear' or 'one_minus_sqrt', got {decay_shape!r}"
                )
            if warmup_steps <= 0:
                raise ValueError(
                    f"lr_schedule='wsd' requires warmup_steps > 0 (got {warmup_steps}); "
                    f"a zero-length warmup on a fresh optimizer count is exactly the "
                    f"bias-correction hole WSD's warmup exists to fill"
                )
            if warmup_steps + decay_steps >= stage_duration:
                raise ValueError(
                    f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
                    f"must be strictly less than stage_duration ({stage_duration}); "
                    f"otherwise the stable phase has zero or negative length"
                )

        # Build a schedule for a given peak LR — reused for per-group LRs below.
        def _build_schedule(peak):
            if lr_schedule_type == 'cosine' and warmup_steps > 0:
                return optax.warmup_cosine_decay_schedule(
                    init_value=1e-8, peak_value=peak,
                    warmup_steps=warmup_steps,
                    decay_steps=max(1, stage_duration - warmup_steps),
                    end_value=min_lr,
                )
            elif lr_schedule_type == 'cosine':
                return optax.cosine_decay_schedule(
                    init_value=peak, decay_steps=stage_duration,
                    alpha=min_lr / peak if peak > 0 else 0.0,
                )
            elif lr_schedule_type == 'wsd':
                # Warmup (linear 0->peak) -> stable (constant peak) -> cooldown
                # (peak -> peak*final_lr_frac). optax.join_schedules SHIFTS the step
                # arg by the boundary before calling each sub-schedule, so `cooldown`
                # receives phase-local t = 0..decay_steps. Do NOT subtract any global
                # offset inside the closure — that's the double-subtract bug where LR
                # sits at peak through the entire cooldown. Upper clip on frac keeps
                # `1 - sqrt(frac)` non-negative past decay_steps (optax keeps calling
                # the last sub-schedule for step > sum(boundaries)).
                stable_steps = stage_duration - warmup_steps - decay_steps
                warmup = optax.linear_schedule(
                    init_value=1e-8, end_value=peak, transition_steps=warmup_steps,
                )
                stable = optax.constant_schedule(peak)
                floor_lr = peak * final_lr_frac
                if decay_shape == 'one_minus_sqrt':
                    def cooldown(t):  # phase-local
                        frac = jnp.clip(t / decay_steps, 0.0, 1.0)
                        return floor_lr + (peak - floor_lr) * (1.0 - jnp.sqrt(frac))
                else:  # linear
                    def cooldown(t):  # phase-local
                        frac = jnp.clip(t / decay_steps, 0.0, 1.0)
                        return floor_lr + (peak - floor_lr) * (1.0 - frac)
                return optax.join_schedules(
                    schedules=[warmup, stable, cooldown],
                    boundaries=[warmup_steps, warmup_steps + stable_steps],
                )
            else:
                # warmup-only (constant peak after warmup)
                warmup = optax.linear_schedule(
                    init_value=1e-8, end_value=peak, transition_steps=warmup_steps,
                )
                constant = optax.constant_schedule(peak)
                return optax.join_schedules(
                    schedules=[warmup, constant], boundaries=[warmup_steps],
                )

        lr = _build_schedule(peak_lr)
        _type_name = lr_schedule_type if lr_schedule_type in ('cosine', 'wsd') else 'constant'
        extra = ""
        if lr_schedule_type == 'wsd':
            extra = (f", decay_steps={decay_steps}, final_lr_frac={final_lr_frac}, "
                     f"decay_shape={decay_shape}")
        print(f"Using {_type_name} schedule: "
              f"peak_lr={peak_lr}, warmup_steps={warmup_steps}, stage_duration={stage_duration}, "
              f"min_lr={min_lr}{extra}")
    else:
        # No schedule requested — reuse peak_lr as a constant. Assigned in the
        # per-group builder below via _build_schedule when we get there; here `lr`
        # stays as the scalar for the single-LR path.
        def _build_schedule(peak):
            return peak

    params = model.weights
    if stage_config is None:
        # Fallback to model config
        trainable_patterns = cfg.model.get("trainable_params", ["all"])
    else:
        trainable_patterns = stage_config['trainable_params']

    if "all" in trainable_patterns:
        return optax.chain(
            optax.clip_by_global_norm(cfg.trainer.clip_grad_norm),
            optax.adamw(learning_rate=lr, weight_decay=cfg.trainer.weight_decay)
        ), model, lr

    # Compile regex patterns for efficiency
    patterns = [re.compile(p) for p in trainable_patterns]
    main_model_use_lora = False
    embed_model_use_lora = False
    model_use_lora = False

    def map_fn(path, _):
        nonlocal main_model_use_lora, embed_model_use_lora, model_use_lora
        # path[0].key is the weight name in a standard dict-based model
        key = path[0].key if hasattr(path[0], 'key') else str(path[0])

        if any(p.search(key) for p in patterns):
            print(key)
            if "main_model" in key and "a_proj" in key:
                main_model_use_lora = True
            if "embed_model" in key and "a_proj" in key:
                embed_model_use_lora = True
            if "main_model" not in key and "embed_model" not in key and "a_proj" in key:
                model_use_lora = True
            return False
        return True

    print("\n\nThe following layers are trainable:")
    mask = jax.tree_util.tree_map_with_path(map_fn, params)
    print("\n\n")
    
    cfg_ref = model.forward.args[0]
    new_cfg = copy.deepcopy(cfg_ref)
    if "main_model" in new_cfg:
        new_cfg["main_model"]["use_lora"] = main_model_use_lora
    if "embed_model" in new_cfg:
        new_cfg["embed_model"]["use_lora"] = embed_model_use_lora
    if "main_model" not in new_cfg and "embed_model" not in new_cfg:
        new_cfg["use_lora"] = model_use_lora

    # 3. Recreate the partial forward function
    model.forward = partial(model.forward.func, new_cfg)
    model.cfg = new_cfg

    # ── Optimizer-state allocation ────────────────────────────────────────────────────────────
    # optax.adamw.init() allocates mu AND nu for EVERY leaf it is handed, and a downstream
    # freeze() only zeroes the resulting updates — it does NOT prevent the allocation. So the
    # default chain below costs 2x ALL params in optimizer state on every step of every stage,
    # however little is trainable (~16 GB/chip for the 4B here). Verified on CPU with a 1k
    # trainable + 100k frozen tree: state = 202,001 elements ~= 2 x total params.
    #
    # MEM_MASKED_OPTIMIZER=1 wraps adamw in optax.masked so moments exist only for TRAINABLE
    # leaves (same tree: 202,001 -> 2,001 elements, with byte-identical updates —
    # frozen_moved 0.0, trainable_moved 9.9999e-02 under both chains).
    #
    # ⚠️ OFF BY DEFAULT, and it is not a free win: masked() changes the opt_state PYTREE
    # (MaskedState / MaskedNode placeholders), so orbax CANNOT restore a checkpoint written
    # under the default chain — a resume dies in load_checkpoint with
    # `ValueError: Item "default" and args PyTreeRestoreArgs(...MaskedNode()...)`.
    # Every checkpoint in gs://memory-layers-training predates this. Use it for FRESH runs where
    # the ~16 GB/chip matters (it is what cancels LoRA's memory advantage — see
    # wiki/implementations/2026-07-20-optimizer-moment-allocation.md); resuming needs a
    # load_checkpoint fallback that is not written yet.
    # mu in bf16 saves ~1.15 GB/chip vs fp32 without breaking Adam. The asymmetry
    # between mu (safe in bf16) and nu (must stay fp32) comes down to sign, not
    # magnitude alone:
    #
    # mu update (β1=0.9): m ← m + 0.1·(g − m). The delta g−m fluctuates BOTH signs
    #   around zero because gradients scatter around their running mean. bf16
    #   truncation of an unbiased two-sided quantity acts as a small noise floor
    #   on the running average — no systematic drift.
    #
    # nu update (β2=0.999): v ← v + 0.001·(g² − v). Because g² ≥ 0, the delta can
    #   only step v UP through the ULP threshold (needs g² > 2.95·v, happens ~8%
    #   of steps); it can never step v DOWN (would need g² < −0.95·v, impossible).
    #   Under bf16, v ratchets monotonically up → effective LR √v decays without
    #   recovery. That's a one-way drift, not "some updates lost" — much worse
    #   than the mu case. nu MUST stay fp32.
    #
    # NB: optax.scale_by_adam DOES expose nu_dtype; the absence in optax.adamw's
    # shortcut isn't the argument — the ratchet mechanism above is. Do not hack
    # nu_dtype to bf16 on the theory that "if optax allowed it, it'd be safe".

    # ── Per-group learning rates (opt-in via cfg.trainer.learning_rates) ─────────
    # Config shape:
    #   trainer:
    #     learning_rates:
    #       mem:   1e-4
    #       embed: 1e-5
    #       main:  1e-5
    #
    # Design: SINGLE global adamw driven by the mem-group base schedule, then a
    # per-leaf `scale_by_tree` scales the final update by ratio[leaf] where
    # ratio = lr_by_group[grp] / peak_mem_lr. Frozen leaves get ratio=0.
    # Net effect: each leaf sees `update = -base_lr(step) * ratio[leaf] * adam_update`.
    #
    # Why not optax.multi_transform: it partitions opt_state by group label tree.
    # Under staged training the trainer PRESERVES opt_state across stage boundaries
    # (to keep the LR schedule position); when stage 1 unfreezes more params the
    # group_label_tree changes shape and the preserved multi_transform state
    # becomes structurally incompatible → TypeError deep in optax's inner update.
    # A single adamw's opt_state is a flat pytree matching params — same structure
    # regardless of which leaves are trainable this stage, so stage-boundary
    # preservation works unchanged (same code path as the single-LR arm).
    #
    # Group assignment ORDERED regex match, mem-first: main_model.layers.14.mem_q_proj
    # contains BOTH "main_model" and "mem_" — checking mem first puts it in the
    # mem group. Same for embed_model.mem_k_proj / .embed_proj_conv.
    lr_by_group = cfg.trainer.get("learning_rates") if hasattr(cfg.trainer, "get") else None
    if lr_by_group:
        _GROUP_PATTERNS = [
            ("mem",   re.compile(r"(mem_|spec_tokens|embed_proj_conv)")),
            ("embed", re.compile(r"embed_model")),
            ("main",  re.compile(r"main_model")),
        ]
        def _group_for(key: str, is_frozen: bool) -> str:
            if is_frozen:
                return "frozen"
            for name, pat in _GROUP_PATTERNS:
                if pat.search(key):
                    return name
            raise RuntimeError(
                f"[optimizer] trainable leaf {key!r} does not match any group in "
                f"{[n for n,_ in _GROUP_PATTERNS]!r}. Add a pattern or fix the key."
            )

        # Base LR = mem's peak; ratios scale each leaf relative to that.
        peak_mem = float(lr_by_group.get("mem", peak_lr))
        base_lr = _build_schedule(peak_mem)
        ratios_by_group = {
            "mem":    float(lr_by_group.get("mem",    peak_mem)) / peak_mem,
            "embed":  float(lr_by_group.get("embed",  peak_mem)) / peak_mem,
            "main":   float(lr_by_group.get("main",   peak_mem)) / peak_mem,
            "frozen": 0.0,
        }

        group_counts = {"mem": 0, "embed": 0, "main": 0, "frozen": 0}
        def _joint_ratio(path, param, is_frozen):
            key = ".".join(str(getattr(p, 'key', p)) for p in path)
            grp = _group_for(key, bool(is_frozen))
            group_counts[grp] += 1
            return jnp.asarray(ratios_by_group[grp], dtype=jnp.float32)
        ratio_tree = jax.tree_util.tree_map_with_path(_joint_ratio, params, mask)

        print(f"\n[optimizer] per-group param counts:")
        for grp in list(group_counts.keys()):
            rate = ratios_by_group[grp] * peak_mem
            print(f"  {grp:8s}: {group_counts[grp]} leaves  (peak LR = {rate:.3e})")
        total_trainable = sum(c for g, c in group_counts.items() if g != "frozen")
        if total_trainable == 0:
            raise RuntimeError(
                "[optimizer] no trainable leaves in ANY group — stage config "
                "or group patterns are broken."
            )
        for grp in lr_by_group:
            if group_counts.get(grp, 0) == 0:
                print(f"[optimizer] note: group {grp!r} has 0 trainable leaves "
                      f"this stage (its LR is inert until a later stage unlocks it).")

        # Custom scale_by_tree — a stateless transform that multiplies each
        # update leaf by the corresponding ratio_tree leaf. Kept private so the
        # ratio_tree closes over the current stage's mask.
        def _scale_by_tree(ratio_pytree):
            def init_fn(_):
                return optax.EmptyState()
            def update_fn(updates, state, params=None):
                del params
                scaled = jax.tree_util.tree_map(
                    lambda u, r: u * r, updates, ratio_pytree,
                )
                return scaled, state
            return optax.GradientTransformation(init_fn, update_fn)

        # MEM_MASKED_OPTIMIZER=1: allocate mu/nu only for trainable leaves.
        # Masked passes frozen leaves' updates through UNCHANGED (they're the
        # raw grads before adamw); scale_by_tree at the tail multiplies by
        # ratio=0 for frozen so those pass-throughs are zeroed out. Semantics
        # remain byte-identical to the unmasked chain, but frozen moments
        # aren't allocated — big HBM saving at stages that freeze most of the
        # model. See wiki/implementations/2026-07-20-optimizer-moment-allocation.md.
        # ⚠️ Changes opt_state pytree (MaskedNode placeholders) — cannot resume
        # from a checkpoint written under the unmasked chain. Fresh runs only.
        if os.environ.get("MEM_MASKED_OPTIMIZER") == "1":
            trainable_mask = jax.tree_util.tree_map(lambda frozen: not frozen, mask)
            print("[optimizer] MEM_MASKED_OPTIMIZER=1: moments allocated for TRAINABLE params "
                  "only (opt_state pytree differs — cannot resume unmasked-mode ckpts)")
            inner = optax.masked(
                optax.adamw(learning_rate=base_lr, weight_decay=cfg.trainer.weight_decay, mu_dtype=jnp.bfloat16),
                trainable_mask,
            )
        else:
            inner = optax.adamw(learning_rate=base_lr, weight_decay=cfg.trainer.weight_decay, mu_dtype=jnp.bfloat16)
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.trainer.clip_grad_norm),
            inner,
            _scale_by_tree(ratio_tree),
        )
        return optimizer, model, base_lr

    # ── Single-LR path (backward-compat: no cfg.trainer.learning_rates) ──────────
    if os.environ.get("MEM_MASKED_OPTIMIZER") == "1":
        # `mask` is True for FROZEN leaves (map_fn returns False when a pattern matches);
        # optax.masked applies its inner transform where the mask is True, so invert it.
        trainable_mask = jax.tree_util.tree_map(lambda frozen: not frozen, mask)
        print("[optimizer] MEM_MASKED_OPTIMIZER=1: moments allocated for TRAINABLE params only "
              "(opt_state pytree differs — cannot resume pre-existing checkpoints)")
        inner = optax.masked(
            optax.adamw(learning_rate=lr, weight_decay=cfg.trainer.weight_decay, mu_dtype=jnp.bfloat16),
            trainable_mask,
        )
    else:
        inner = optax.adamw(learning_rate=lr, weight_decay=cfg.trainer.weight_decay, mu_dtype=jnp.bfloat16)

    optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.trainer.clip_grad_norm),
            inner,
            # Required in BOTH modes: masked() passes updates through UNCHANGED where its mask is
            # False, so without this frozen leaves would receive their raw gradient as an update.
            optax.transforms.freeze(mask)
        )

    return optimizer, model, lr

def setup_optimizer(cfg, model):
    """Setup optimizer using model's trainable_params config.

    Returns (optimizer, model, lr) where lr is the schedule callable or constant.
    """
    stage_config = None
    if cfg.trainer.get("training_stages"):
        stage_config = cfg.trainer.training_stages[0]
    return setup_optimizer_for_stage(cfg, model, stage_config)

def get_gcs_region() -> str | None:
    region = os.environ.get("GCS_REGION")
    if region:
        return region
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
        )
        zone_path = urllib.request.urlopen(req, timeout=2).read().decode()
        zone = zone_path.split("/")[-1]          # "us-central2-b"
        return "-".join(zone.split("-")[:-1])     # "us-central2"
    except Exception:
        return None


def setup_gcs_credentials():
    """Set GOOGLE_APPLICATION_CREDENTIALS from GCS_USER_EMAIL env var if present."""
    email = os.environ.get("GCS_USER_EMAIL")
    if not email:
        return
    credentials_path = Path(f"~/.config/gcloud/legacy_credentials/{email}/adc.json").expanduser()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    os.environ["GCLOUD_PROJECT"] = os.environ.get("GCS_BUCKET_PROJECT")
    print("GCloud credentials set!")


def ensure_gcs_bucket(bucket_name: str, region: str | None):
    from google.cloud import storage
    project = os.environ.get("GCS_BUCKET_PROJECT")
    os.environ["GCLOUD_PROJECT"] = project
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        client.create_bucket(bucket_name, location=region)
        print(f"Created GCS bucket gs://{bucket_name} in {region}")


def upload_config_to_gcs(cfg, gcs_run_dir: str):
    """Upload .hydra/ directory to {gcs_run_dir}/.hydra/ (process 0 only)."""
    if jax.process_index() != 0:
        return
    from google.cloud import storage
    project = os.environ.get("GCS_BUCKET_PROJECT")
    os.environ["GCLOUD_PROJECT"] = project
    bucket_name = gcs_run_dir[5:].split("/")[0]
    prefix = gcs_run_dir[5 + len(bucket_name) + 1:]  # path within bucket
    client = storage.Client(project=os.environ.get("GCLOUD_PROJECT"))
    bucket = client.bucket(bucket_name)
    hydra_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
    for filename in os.listdir(hydra_dir):
        local_path = os.path.join(hydra_dir, filename)
        if os.path.isfile(local_path):
            blob_path = f"{prefix}/.hydra/{filename}"
            bucket.blob(blob_path).upload_from_filename(local_path)
    print(f"[0] .hydra/ uploaded to {gcs_run_dir}/.hydra/")


# The timestamp half of a run-dir / run id.
RUN_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def run_dir_name(cfg) -> str:
    """`{run_name}-{YYYY-MM-DD}-{HH-MM-SS}` — THE identity of one launch.

    This single string names the GCS run-dir *and* (via wandb_run_id_from_run_dir) the wandb run,
    so the two are 1:1 and cannot drift. That matters because **run_name alone is not unique**:
    every launch mints its own dir, and re-using a run_name months later re-uses the prefix. An
    identity keyed on the name would make a fresh launch collide with the older run's wandb run
    and let an eval box scanning by name pick up the older run's checkpoints — logging another
    model's scores into this run's curve.

    Pure: no GCS calls (that's _build_gcs_run_dir's job), so callers can ask for the identity
    without side effects.

    `trainer.run_start_time` PINS the timestamp instead of taking it from Hydra. That is what
    lets a launcher compute the run-dir BEFORE training starts and hand the same one to a
    training box and an eval box in parallel (scripts/infrastructure/multi-tpu-box-run.sh) —
    otherwise the dir only becomes knowable once train.py prints it, forcing a serial launch.
    Note the pinned value need NOT match hydra.run.dir's own `${now:}` stamp (that dir is local
    scratch); the GCS run-dir and the wandb id follow this one.
    """
    stamp = cfg.get("trainer", {}).get("run_start_time", None)
    if stamp:
        stamp = str(stamp)
        if not RUN_STAMP_RE.match(stamp):
            # Fail here rather than let it through: a malformed stamp makes a run-dir that
            # wandb_run_id_from_run_dir refuses to parse, and an eval box pointed at
            # "{name}-{stamp}" would poll a dir that can never exist.
            raise ValueError(
                f"trainer.run_start_time must be YYYY-MM-DD-HH-MM-SS, got {stamp!r}")
        return f"{get_run_name(cfg)}-{stamp}"

    # Date/time from the Hydra output dir (outputs/YYYY-MM-DD/HH-MM-SS/...)
    output_dir = HydraConfig.get().runtime.output_dir  # absolute path
    parts = output_dir.replace("\\", "/").split("/")
    try:
        idx = next(i for i, p in enumerate(parts) if p == "outputs")
        date_str, time_str = parts[idx + 1], parts[idx + 2]
    except (StopIteration, IndexError):
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
    return f"{get_run_name(cfg)}-{date_str}-{time_str}"


def _build_gcs_run_dir(cfg) -> str | None:
    """Returns gs://{bucket}/{run_dir_name(cfg)} if GCS_BUCKET is set."""
    bucket = os.environ.get("GCS_BUCKET")
    if not bucket:
        return None
    ensure_gcs_bucket(bucket, get_gcs_region())
    return f"gs://{bucket}/{run_dir_name(cfg)}"


def setup_checkpointing(cfg):
    from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions

    resume_from = cfg.trainer.get("resume_from")
    resume_step = None
    resume_from_dir = None
    if resume_from:
        resume_from = str(resume_from)
        # If a step number is embedded at the end of the path (e.g. gs://.../qwen3_mem_embed/100000),
        # strip it to get the CheckpointManager directory and record the step explicitly.
        parts = resume_from.rstrip("/").rsplit("/", 1)
        if len(parts) == 2 and parts[1].isdigit():
            resume_step = int(parts[1])
            resume_from_dir = parts[0]
        else:
            resume_from_dir = resume_from
        if resume_from_dir and not resume_from_dir.startswith("gs://"):
            resume_from_dir = os.path.abspath(resume_from_dir)

    gcs_run_dir = _build_gcs_run_dir(cfg)
    if gcs_run_dir:
        upload_config_to_gcs(cfg, gcs_run_dir)
        checkpoint_dir = f"{gcs_run_dir}/{cfg.model.name}"
    else:
        checkpoint_dir = os.path.abspath(
            os.path.join(HydraConfig.get().runtime.output_dir, cfg.model.name)
        )

    # max_to_keep bounds how long a checkpoint stays evaluable: orbax rotates all but the last N,
    # so the window is N * checkpoint_interval steps. An eval box must reach a checkpoint inside
    # that window or the curve gets holes — raise N when checkpointing frequently.
    # See wiki/evaluation/eval-boxes.md.
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.trainer.get("max_to_keep", 4),
        save_interval_steps=cfg.trainer.checkpoint_interval,
        multiprocessing_options=MultiprocessingOptions(primary_host=0),
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.StandardCheckpointer(),
        options=options,
    )
    return checkpoint_manager, resume_step, resume_from_dir

def load_checkpoint(checkpoint_manager, model, opt_state, return_stage_idx=False, resume_step=None, resume_from_dir=None):
    step = 0
    stage_idx = 0
    load_manager = checkpoint_manager
    if resume_from_dir is not None:
        from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions
        load_options = ocp.CheckpointManagerOptions(
            multiprocessing_options=MultiprocessingOptions(primary_host=0),
        )
        load_manager = ocp.CheckpointManager(
            resume_from_dir,
            ocp.StandardCheckpointer(),
            options=load_options,
        )
    ckpt_step = resume_step if resume_step is not None else load_manager.latest_step()
    if ckpt_step is not None:
        def restore_and_reshard(r, a):
            # Fall back to initialized value for missing/None leaves.
            val = a if r is None else r
            # Cast to the target's dtype before resharding. `a` may be fp32 here even for a
            # leaf that was bf16 on disk: promote_trainable_to_fp32 (train.py, runs before
            # setup_checkpointing) promotes every trainable leaf of THIS run's model BEFORE
            # restore, but the restored value `r` carries whatever dtype the SOURCE checkpoint
            # was saved in — orbax's restore does not itself cast to match `a`. Left as a bare
            # device_put (sharding-only), this silently re-traps a warm-started or resumed leaf
            # back in bf16 and its ULP-truncated adamw updates (see
            # wiki/experiments/2026-08-07-bf16-ulp-freeze-empirical-confirmation.md), invisibly
            # undoing promote_trainable_to_fp32 for exactly the leaves it exists to protect —
            # confirmed happening in practice: a warm start from a bf16-saved checkpoint left
            # mem_q_proj/mem_o_proj/embed_model.mem_{k,v}_proj back at dtype=bfloat16 despite
            # promote_fp32 having promoted them moments earlier.
            if hasattr(val, 'dtype') and hasattr(a, 'dtype') and val.dtype != a.dtype:
                val = val.astype(a.dtype)
            # Re-apply the sharding from the initialized array so TP einsums work.
            if hasattr(a, 'sharding'):
                return jax.device_put(val, a.sharding)
            return val

        if resume_step is not None:
            # Warm start: only restore weights, keep fresh optimizer state and step=0.
            weights_only_state = {"weights": model.weights}
            try:
                restored = load_manager.restore(
                    ckpt_step,
                    args=ocp.args.PyTreeRestore(item=weights_only_state, partial_restore=True)
                )
            except Exception:
                # Fall back to a direct per-item restore. Try 'default' (this repo's StandardCheckpointer
                # layout) then 'state' (checkpoints saved by the exp/zeroinit-style composite manager,
                # which stores weights under a 'state' item alongside 'data_iter'). Lets us warm-start
                # across the two checkpoint formats.
                checkpointer = ocp.PyTreeCheckpointer()
                base = str(load_manager.directory).rstrip("/") + f"/{ckpt_step}"
                restored = None
                last_err = None
                for sub in ("default", "state"):
                    try:
                        restored = checkpointer.restore(f"{base}/{sub}", item=weights_only_state, partial_restore=True)
                        break
                    except Exception as e:
                        last_err = e
                if restored is None:
                    raise last_err
            model.weights = jax.tree_util.tree_map(restore_and_reshard, restored['weights'], model.weights, is_leaf=lambda x: x is None)
        else:
            # Resume interrupted run: restore full state and continue from ckpt_step.
            abstract_state = {"weights": model.weights, "opt_state": opt_state, "stage_idx": 0}
            try:
                restored = load_manager.restore(
                    ckpt_step,
                    args=ocp.args.PyTreeRestore(item=abstract_state, partial_restore=True)
                )
            except Exception:
                checkpointer = ocp.PyTreeCheckpointer()
                ckpt_path = str(load_manager.directory).rstrip("/") + f"/{ckpt_step}/default"
                restored = checkpointer.restore(ckpt_path, item=abstract_state, partial_restore=True)
            model.weights = jax.tree_util.tree_map(restore_and_reshard, restored['weights'], model.weights, is_leaf=lambda x: x is None)
            opt_state = jax.tree_util.tree_map(restore_and_reshard, restored['opt_state'], opt_state, is_leaf=lambda x: x is None)
            stage_idx = restored.get('stage_idx', 0)
            step = ckpt_step

    if return_stage_idx:
        return step, opt_state, stage_idx
    return step, opt_state

def load_inference_checkpoint(checkpoint_manager, model, step=None):
    if step is None:
        step = checkpoint_manager.latest_step()
    # Only restore weights, ignore opt_state
    abstract_state = {"weights": model.weights}

    # Explicit restore_args, built from model.weights' OWN concrete sharding (i.e. THIS box's
    # freshly-initialized mesh), are required for a correct cross-topology restore. Without
    # them, orbax has no target sharding to reshard onto and falls back to the checkpoint's
    # ON-DISK sharding metadata -- literal saved device ids, harmless on a box with AT LEAST as
    # many chips as the checkpoint was saved with (those ids still exist), but a hard crash on a
    # box with FEWER chips the moment it hits a saved device id (e.g. a v6e-8 checkpoint's id 7)
    # that doesn't exist here: "sharding passed to deserialization should be specified,
    # concrete... Got None". See
    # wiki/implementations/2026-08-24-checkpoint-restore-cross-topology-sharding.md.
    restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)

    # We need PyTreeRestore for partial_restore=True
    # StandardCheckpointer might not support PyTreeRestoreArgs,
    # so we may need to use a PyTreeCheckpointer if the manager isn't already one.
    try:
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.PyTreeRestore(item=abstract_state, restore_args=restore_args, partial_restore=True)
        )
    except Exception:
        # Fallback if the manager is using StandardCheckpointer
        # We can create a temporary PyTreeCheckpointer to do the job
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt_path = str(checkpoint_manager.directory).rstrip("/") + f"/{step}/default"
        restored = checkpointer.restore(
            ckpt_path,
            item=abstract_state,
            restore_args=restore_args,
            partial_restore=True
        )

    # Restore None back to original empty arrays, then put on device
    def restore_none(r, a):
        return a if r is None else r
    model.weights = jax.tree_util.tree_map(restore_none, restored['weights'], model.weights)
    # Multi-host: orbax restores sharded weights as GLOBAL arrays (not fully addressable),
    # and a bare device_put on those raises "must be a fully addressable array" — they are
    # already placed by the restore. Only put host-side leaves (same guard style as
    # save_checkpoint's unshard below).
    def _put_local(x):
        if hasattr(x, 'is_fully_addressable') and not x.is_fully_addressable:
            return x
        return jax.device_put(x)
    model.weights = jax.tree_util.tree_map(_put_local, model.weights)
    return step

def save_checkpoint(checkpoint_manager, model, opt_state, step, stage_idx=0):
    # Sharded native save. Prior implementation (up to 2026-07-23) did an
    # explicit tree_map(jax.device_put(x, PartitionSpec()), state) + jax.device_get
    # to materialize a FULL replicated + host-side copy of weights + opt_state
    # before writing. That queued every leaf's un-shard concurrently, driving
    # peak HBM to ~35 GB on v6e (limit 33.55 GB) at 4B scale and OOMing with
    # `RESOURCE_EXHAUSTED: 47.5M needed, 8.2M free` — see
    # wiki/implementations/2026-07-23-hbm-sharding-probe.md.
    #
    # StandardSave inspects each leaf's .sharding and has every host stream
    # only its local shards to disk in parallel — no gather, no host materialize.
    # ALL hosts must still call .save() (CheckpointManagerOptions has
    # multiprocessing_options); no process_index guard.
    jax.effects_barrier()

    def prune(tree):
        # Orbax cannot save zero-sized arrays; drop them from the pytree.
        # Works on sharded jax arrays too (.size is defined on any Array-like).
        if isinstance(tree, dict):
            return {k: prune(v) for k, v in tree.items() if getattr(v, 'size', -1) != 0}
        return jax.tree_util.tree_map(lambda x: None if getattr(x, 'size', -1) == 0 else x, tree)

    save_weights = prune(model.weights)
    save_opt_state = prune(opt_state)

    checkpoint_manager.save(
        step,
        args=ocp.args.StandardSave(
            {"weights": save_weights, "opt_state": save_opt_state, "stage_idx": stage_idx}
        ),
    )

    checkpoint_manager.wait_until_finished()
    jax.effects_barrier()

    if not (os.path.exists(checkpoint_manager.directory) and len([d for d in os.listdir(checkpoint_manager.directory) if os.path.isdir(os.path.join(checkpoint_manager.directory, d))]) > 0):
        print(f"[{jax.process_index()}] Saved step {step} to {checkpoint_manager.directory}", flush=True)


def process_train_pairs(tokens, masks):
    # Per-row CE gate (B,): 1.0 = normal cross-entropy, 0.0 = masked (retrieval-only rows).
    # Defaults to all-ones for datasets/batches that don't provide it.
    def _ce_enable(ref_tokens):
        if "ce_enable" in masks:
            return masks["ce_enable"]
        n = ref_tokens.shape[0]
        return jnp.ones((n,), dtype=jnp.float32)

    if isinstance(tokens, dict) and "teacher_batch" in tokens:
        inputs = {"batch": tokens["batch"][:, :-1], "docs": tokens["docs"], "teacher_batch": tokens["teacher_batch"][:, :-1]}
        targets = tokens["batch"][:, 1:]
        input_masks = {
            "batch_mask": masks["batch_mask"][:, :-1],
            "docs_mask": masks["docs_mask"],
            "pos_doc_mask": masks["pos_doc_mask"],
            "teacher_mask": masks["teacher_mask"][:, :-1],
        }
        if "student_distill_mask" in masks:
            input_masks["student_distill_mask"] = masks["student_distill_mask"][:, 1:]
        if "teacher_distill_mask" in masks:
            input_masks["teacher_distill_mask"] = masks["teacher_distill_mask"][:, 1:]
        loss_masks = masks["loss_mask"][:, 1:]
        ce_enable = _ce_enable(tokens["batch"])
    elif isinstance(tokens, dict) and "batch" in tokens:
        inputs = {"batch": tokens["batch"][:, :-1], "docs": tokens["docs"]}
        targets = tokens["batch"][:, 1:]

        input_masks = {"batch_mask": masks["batch_mask"][:, :-1], "docs_mask": masks["docs_mask"]}
        if "pos_doc_mask" in masks:
            input_masks["pos_doc_mask"] = masks["pos_doc_mask"]
        loss_masks = masks["loss_mask"][:, 1:]
        ce_enable = _ce_enable(tokens["batch"])
    else:
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        input_masks = masks["batch_mask"][:, :-1]
        loss_masks = masks["loss_mask"][:, 1:]
        ce_enable = _ce_enable(tokens)
    return inputs, targets, input_masks, loss_masks, ce_enable


def get_run_name(cfg):
    """
    Generate a descriptive W&B run name from config.
    
    Format: {model}_sz{mem_size}_k{top_k}_{PK/noPK}[_memKL{weight}]_lr{lr}
    """
    trainer = cfg.get("trainer", {})
    
    # Check if run_name or wandb_name is explicitly provided
    custom_run_name = trainer.get("run_name") or trainer.get("wandb_name")
    if custom_run_name:
        return custom_run_name
        
    model_name = cfg.model.main_model.model_id.replace("/", "-")

    #data config

    bio_interval = cfg.dataset.get("bio_interval", 1)
    
    # Memory config (if exists)
    mem_cfg = cfg.model.get("memory", {})
    mem_size = mem_cfg.get("mem_size", "")
    mem_top_k = mem_cfg.get("mem_top_k", "")
    use_pk = "PK" if mem_cfg.get("mem_use_product_keys", False) else "noPK"
    
    # Aux loss config
    aux_losses = trainer.get("aux_losses", {})
    mem_kl_cfg = aux_losses.get("mem_uniform_kl", {})
    mem_kl_weight = mem_kl_cfg.get("weight", None)
    mem_kl_enabled = mem_kl_cfg.get("enabled", False)
    mem_kl_str = f"_memKL{mem_kl_weight}" if mem_kl_weight and mem_kl_enabled else ""
    
    lr = trainer.get("learning_rate", "N/A")
    # Build name
    if "embed" in cfg.model.get("name", ""):
        run_name = f"squad_{model_name}_data{cfg.dataset.name}_data-interval{bio_interval}_sz{mem_size}_k{mem_top_k}_{use_pk}{mem_kl_str}_lr{lr}"
    else:
        run_name = f"squad_{model_name}_data{cfg.dataset.name}_lr{lr}"
    
    return run_name

# A run-dir basename: <run_name>-<YYYY-MM-DD>-<HH-MM-SS>. The name may contain '-', so anchor on
# the fixed-width timestamp at the end.
RUN_DIR_RE = re.compile(r"^(?P<name>.+)-(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")


def wandb_run_id_from_run_dir(run_dir):
    """Deterministic wandb run id for ONE launch, derived from its run-dir (see run_dir_name).

    Keying on the run-dir rather than the run name is what makes wandb runs and GCS folders 1:1:
    the timestamp makes every launch unique, so re-using a run_name can never append to an older
    run's curve, and an eval box pointed at a dir can compute the id without being told it.

    NOT byte-equal to the dir: wandb ids are length-capped (<=64) and a real dir is longer —
    '<51-char run_name>-<19-char stamp>' is 71 — so the name is slugged to wandb's charset and
    truncated to 40, giving <=60. Uniqueness comes from the timestamp, which is kept in full, so
    truncating the name is safe.

    Accepts a bare basename or a full gs:// path.
    """
    base = str(run_dir).rstrip("/").split("/")[-1]
    m = RUN_DIR_RE.match(base)
    if not m:
        raise ValueError(
            f"not a run-dir basename ('<run_name>-<YYYY-MM-DD>-<HH-MM-SS>'): {base!r}")
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", m.group("name"))[:40].strip("-")
    return f"{slug}-{m.group('ts')}"


def resolve_wandb_run_id(cfg, run_dir):
    """trainer.wandb_run_id -> the id to pass to wandb.init, or None to let wandb generate one.

    null/absent (default) => None: unchanged legacy behaviour, a fresh auto-id run per launch.
    "auto"                => derived from this launch's run_dir (see wandb_run_id_from_run_dir).
    any other string      => used verbatim. This is the escape hatch for keeping ONE curve across
                             a preemption-resume: a resume mints a NEW run-dir, so "auto" would
                             open a second wandb run — pass the original id to continue the first.
    """
    rid = cfg.get("trainer", {}).get("wandb_run_id", None)
    if rid is None:
        return None
    return wandb_run_id_from_run_dir(run_dir) if rid == "auto" else str(rid)


def freeze_dict(d):
    """Convert nested dict to hashable tuple of tuples for JIT static args."""
    if d is None:
        return None
    return tuple(
        (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
        for k, v in sorted(d.items())
    )


def unfreeze_dict(t):
    """Convert frozen tuple back to dict."""
    if t is None:
        return None
    return {k: dict(v) if isinstance(v, tuple) else v for k, v in t}


# Port for the jax.distributed coordinator on a multi-host GCE slice. 8476 is JAX's own default
# for TPU clusters; it only has to be free and identical across the slice's workers.
_JAX_COORDINATOR_PORT = int(os.environ.get("JAX_COORDINATOR_PORT", "8476"))


def _gce_instance_attribute(name):
    """One `instance/attributes/<name>` value from GCE metadata, or None if absent/unreachable."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"http://metadata.google.internal/computeMetadata/v1/instance/attributes/{name}",
            headers={"Metadata-Flavor": "Google"},
        )
        return urllib.request.urlopen(req, timeout=2).read().decode().strip()
    except Exception:
        return None


def _tpu_worker_endpoints():
    """`worker-network-endpoints` from GCE metadata, or None if absent/unreachable."""
    return _gce_instance_attribute("worker-network-endpoints")


def init_jax_distributed():
    """jax.distributed.initialize(), skipped on a GCE VM with an attached TPU.

    JAX auto-detects the TPU slice from `worker-network-endpoints`. A Cloud TPU node publishes
    "id:id:ip" triples there; a GCE VM with an attached TPU (flex-start — machine types
    ct6e-*/ct5p-*, see wiki/infrastructure/experiment-launch-instructions.md §2.2) publishes the
    bare instance name instead. JAX's parser does `worker.split(':')[2]` on each entry, so a name
    with no colons raises `IndexError: list index out of range` from deep inside
    cloud_tpu_cluster, before anything useful has run.

    So bare-name metadata means "JAX's own auto-detect cannot parse this", NOT "this is one host".
    Both shapes exist and they need opposite handling — split on the NUMBER of endpoints:

      1 endpoint   single-host GCE box -> skip; distributed init is a no-op at process_count 1.
      2+ endpoints MULTI-host flex slice (e.g. 2 x ct6e-standard-4t under a 2x4 workload policy,
                   runbook §2.3) -> MUST initialize, passing coordinator/num_processes/process_id
                   explicitly because auto-detect would hit the IndexError above.

    Do not be fooled by device discovery on a multi-host slice: libtpu forms the mesh on its own,
    so jax.device_count()==8 / process_count()==2 look correct even with init skipped. The
    DISTRIBUTED SYSTEM is what is missing, and it fails later and elsewhere — orbax's
    StandardCheckpointer raises "Distributed system is not available; please initialize it via
    jax.distributed.initialize()" from setup_checkpointing, long after a probe would have passed.
    That is exactly how this bug reached a training launch on 2026-07-20.

    EVERY entry point that touches JAX must call this rather than jax.distributed.initialize()
    directly. train.py and evals/eval_worker.py are separate processes and each needs it; fixing
    only one leaves the other to fail at exactly the same line.
    """
    # Deliberate standalone use of ONE slice worker (its 4 chips) while the peer does other
    # work: metadata still advertises the whole slice, so auto-detection must be overridden.
    # Pair with the libtpu confinement env (TPU_SKIP_MDS_QUERY / TPU_PROCESS_BOUNDS /
    # TPU_CHIPS_PER_PROCESS_BOUNDS) — see eval_msa_rag.sh / train_ground_s1_doccode.sh.
    if os.environ.get("JAX_FORCE_SINGLE_HOST") == "1":
        print("[jax] JAX_FORCE_SINGLE_HOST=1: skipping jax.distributed.initialize() "
              "(this slice worker runs standalone)")
        return
    endpoints = _tpu_worker_endpoints()
    if endpoints is not None and not any(e.count(":") >= 2 for e in endpoints.split(",")):
        workers = [e for e in endpoints.split(",") if e]
        if len(workers) <= 1:
            print(
                f"[jax] GCE-attached TPU (worker-network-endpoints={endpoints!r}); "
                "single host, skipping jax.distributed.initialize()"
            )
            return
        # Multi-host slice. The bare names are GCE instance names, resolvable over the VPC's
        # internal DNS, so worker 0 is a usable coordinator. process_id comes from the metadata
        # key the TPU agent sets per host (agent-worker-number); TPU_WORKER_ID inside `tpu-env`
        # carries the same value if that key is ever absent.
        process_id = _gce_instance_attribute("agent-worker-number")
        if process_id is None:
            raise RuntimeError(
                f"multi-host GCE TPU slice ({len(workers)} workers: {endpoints!r}) but metadata "
                "key 'agent-worker-number' is absent, so this process cannot know its rank. "
                "Set JAX_PROCESS_ID or fix the box image."
            )
        coordinator = f"{workers[0]}:{_JAX_COORDINATOR_PORT}"
        print(
            f"[jax] multi-host GCE TPU slice: {len(workers)} workers, process_id={process_id}, "
            f"coordinator={coordinator} (explicit init — auto-detect cannot parse bare names)"
        )
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=len(workers),
            process_id=int(process_id),
        )
        return
    jax.distributed.initialize()
