"""
Trainer Class

Handles training and validation loops with support for auxiliary losses.
"""

import os
import re
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial

# Which _train_step buffers to donate. Signature: (forward, optimizer, weights, opt_state, ...)
# so 2=weights, 3=opt_state. Donation lets XLA reuse an input buffer for the output.
#
# DEFAULT IS OFF, deliberately. donate_argnums=(2,3) arrived with the train-speed work (90e2aa2)
# and is new since the last known-good run (707a128). It makes stage 3 (main_model trainable)
# produce NaN GRADIENTS with a perfectly finite loss, which deadlocks the run: every step is
# skipped by the NaN guard, so Adam's count never advances and the LR stays pinned at the warmup's
# init_value (1e-8). Measured on ONE box, back to back, ckpt 16000 / LR=1e-12 / identical batches
# (scripts/embed/debug_donate_ab.sh):
#
#     MEM_DONATE=both (2,3)  -> 5/20 steps NaN
#     MEM_DONATE=opt  (3,)   -> 5/20 steps NaN     <- opt_state alone is ALSO enough
#     MEM_DONATE=none ()     -> 0/20 steps NaN     <- clean
#
# opt_state is never read by the backward, so donation is not corrupting a buffer the gradient
# depends on. What it does is change XLA's global buffer assignment, and with it the fusion of the
# backward: the same weights and batch give losses differing in the 5th decimal across modes
# (0.8455632 vs 0.8455887). So donation is a TRIGGER, not the root cause — the root cause is a
# latent numerical hazard in the backward that only fires once main_model is trainable (stage 2,
# main frozen, is clean under donation; ["all"], main trainable with no freeze wrapper, NaNs).
# See wiki/implementations/2026-07-17-stage3-grad-nan-donation.md. Turning donation off restores
# April's behaviour and is verified clean; it costs the HBM saving donation bought (the frozen-main
# stages fit comfortably regardless — the `none` arm ran the real bs16/seq512 config without OOM).
#
#   none -> ()     default: no donation
#   opt  -> (3,)   opt_state only            (KNOWN BAD: still NaNs — kept only for repro)
#   both -> (2,3)  weights + opt_state       (KNOWN BAD: the default that broke the run)
_DONATE_MODE = os.environ.get("MEM_DONATE", "none").lower()
_DONATE = {"both": (2, 3), "opt": (3,), "none": ()}.get(_DONATE_MODE, ())
if os.environ.get("MEM_NO_DONATE") == "1":   # back-compat with the first probe
    _DONATE = ()
from tqdm import tqdm
from losses import compute_aux_losses
from utils import load_checkpoint, save_checkpoint, process_train_pairs, freeze_dict, unfreeze_dict, parse_training_stages, get_current_stage_idx, setup_optimizer_for_stage
from evals import get_evaluator
from data import get_dataset

class Trainer:
    """
    Trainer class for memory-augmented language models.
    
    Handles:
    - Training loop with auxiliary losses
    - Validation loop with auxiliary losses
    - Logging to W&B
    - Checkpointing
    """
    
    def __init__(self, cfg, model, data, optimizer, checkpoint_manager, resume_step=None, resume_from_dir=None, lr_schedule=None):
        """
        Initialize trainer.

        Args:
            cfg: Hydra config
            model: Model instance with weights and forward function
            data: Dataset instance with train/eval generators
            optimizer: Optax optimizer
            checkpoint_manager: Orbax checkpoint manager
        """
        self.cfg = cfg
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.resume_step = resume_step
        self.resume_from_dir = resume_from_dir
        self.lr_schedule = lr_schedule

        # Parse training stages
        self.training_stages = parse_training_stages(cfg)
        self.current_stage_idx = 0

        # Axis A (wiki/experiments/2026-07-16-train-speed-axes.md): stop_gradient the frozen weights
        # so XLA prunes their backward (~2x speedup in frozen-main stages, quality-neutral). Off by
        # default (byte-identical). current_trainable_patterns tracks the live stage's trainable set.
        self.stop_grad_frozen = cfg.trainer.get("stop_grad_frozen", False)
        self.current_trainable_patterns = (
            tuple(self.training_stages[0]["trainable_params"]) if self.training_stages
            else tuple(cfg.model.get("trainable_params", ["all"]))
        )

        # Initialize optimizer state
        self.opt_state = optimizer.init(model.weights)

        # CE weight for main loss
        self.ce_weight = jnp.array(cfg.trainer.get("ce_weight", 1.0))

        # Prepare aux loss config (as hashable frozen tuple for JIT)
        self.base_aux_loss_dict = None
        self.aux_loss_config = None
        if cfg.trainer.get("aux_losses"):
            self.base_aux_loss_dict = {k: dict(v) for k, v in cfg.trainer.aux_losses.items()}
            aux_dict = dict(self.base_aux_loss_dict)
            # Apply stage 0 overrides if stages are configured
            if self.training_stages is not None:
                self._apply_stage_loss_overrides(self.training_stages[0], aux_dict)
            self.aux_loss_config = freeze_dict(aux_dict)
        
        # Apply stage 0 ce_weight override if present
        if self.training_stages is not None and 'ce_weight' in self.training_stages[0]:
            self.ce_weight = jnp.array(self.training_stages[0]['ce_weight'])

        # Instantiate evaluators, loading per-eval datasets where specified
        evals_cfg = cfg.trainer.get("evals", {})
        self.eval_keys = []
        self.evaluators = []
        self.evaluator_datasets = []
        
        self.evaluator_doc_datasets = []

        for key, e in (evals_cfg.items() if evals_cfg else []):
            self.eval_keys.append(key)

            # Support nested config structure
            actual_eval_cfg = e.get("eval", e)
            self.evaluators.append(get_evaluator(actual_eval_cfg, key=key))

            actual_ds_cfg = e.get("dataset", e.get("dataset_cfg"))
            if actual_ds_cfg:
                print(f"Loading dataset '{actual_ds_cfg.get('name', 'unnamed')}' for eval '{key}'...")
                self.evaluator_datasets.append(get_dataset(actual_ds_cfg, model))
            else:
                self.evaluator_datasets.append(None)  # falls back to shared eval_data

            doc_ds_cfg = actual_eval_cfg.get("doc_dataset")
            if doc_ds_cfg:
                print(f"Loading doc_dataset '{doc_ds_cfg.get('name', 'unnamed')}' for eval '{key}'...")
                self.evaluator_doc_datasets.append(get_dataset(doc_ds_cfg, model))
            else:
                self.evaluator_doc_datasets.append(None)
    
    def _apply_stage_loss_overrides(self, stage_config, aux_dict):
        """
        Apply stage-specific aux_losses overrides to an aux_dict (in-place).
        
        Args:
            stage_config: Stage dict with optional 'aux_losses' key
            aux_dict: Base aux loss config dict to modify
        """
        stage_aux = stage_config.get('aux_losses', {})
        for loss_name, loss_overrides in stage_aux.items():
            if loss_name in aux_dict:
                aux_dict[loss_name].update(loss_overrides)
            else:
                aux_dict[loss_name] = dict(loss_overrides)
    
    def _get_stage_loss_config(self, stage_config):
        """
        Get ce_weight and aux_loss_config for a given stage, 
        merging stage overrides with the base config.
        
        Args:
            stage_config: Stage dict from training_stages
            
        Returns:
            ce_weight: jnp scalar
            aux_loss_config: Frozen dict for JIT
        """
        ce_weight = jnp.array(stage_config.get('ce_weight', float(self.cfg.trainer.get('ce_weight', 1.0))))
        
        if self.base_aux_loss_dict is not None:
            aux_dict = {k: dict(v) for k, v in self.base_aux_loss_dict.items()}
            self._apply_stage_loss_overrides(stage_config, aux_dict)
            return ce_weight, freeze_dict(aux_dict)
        return ce_weight, None

    @staticmethod
    @partial(jax.jit, static_argnames=("forward", "optimizer", "aux_loss_config", "trainable_patterns", "lr_schedule"),
             donate_argnums=_DONATE)  # donate weights + opt_state buffers (halves their HBM footprint)
    def _train_step(forward, optimizer, weights, opt_state, inputs, targets, input_masks, loss_masks, ce_weight, aux_loss_config=None, ce_enable=None, trainable_patterns=None, lr_schedule=None):
        """
        JIT-compiled training step.

        Args:
            ce_weight: Scalar weight for the main cross-entropy loss
            trainable_patterns: optional tuple of regex strings (the stage's trainable_params). When
                set (trainer.stop_grad_frozen=true), weights NOT matching any pattern are
                stop_gradient'd before the forward, so XLA prunes their backward. Quality-neutral:
                a frozen weight's grad is discarded by optax.freeze anyway, and stop_gradient on a
                weight zeros only ITS grad — the activation-gradient path to trainable params is
                unchanged, so trainable-param updates are identical. Saves ~2x the step in
                frozen-main stages. Default None = no stop_gradient (byte-identical to before).
                See wiki/experiments/2026-07-16-train-speed-axes.md (Axis A).

        Returns:
            weights: Updated weights
            opt_state: Updated optimizer state
            ce_loss: Cross-entropy loss (unweighted, for logging)
            aux_result: Dict with auxiliary loss values
        """
        # Unfreeze config for use inside JIT
        aux_cfg = unfreeze_dict(aux_loss_config)
        collect_aux = aux_cfg is not None and len(aux_cfg) > 0

        _frozen_pats = [re.compile(p) for p in trainable_patterns] if trainable_patterns else None

        def loss_fn(w):
            if _frozen_pats is not None:
                # Axis A: detach frozen weights (keys matching NO trainable pattern) so their backward
                # is pruned. Static per-leaf choice (pattern match is at trace time) — correct branch.
                def _sg_if_frozen(path, leaf):
                    key = path[0].key if hasattr(path[0], "key") else str(path[0])
                    return leaf if any(pt.search(key) for pt in _frozen_pats) else jax.lax.stop_gradient(leaf)
                w = jax.tree_util.tree_map_with_path(_sg_if_frozen, w)

            pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)

            output = forward(inputs, w, pad_mask=pad_mask, collect_aux=collect_aux)
            logits = output.logits
            aux_data = output.aux
            
            one_hot = jax.nn.one_hot(targets, logits.shape[-1])
            ce_loss = optax.softmax_cross_entropy(logits, one_hot)
            # Per-row CE gate: rows with ce_enable=0 (retrieval-only similarity rows)
            # contribute nothing to cross-entropy but still feed doc_access_loss below
            # (which uses the ungated loss_masks), so they train retrieval without
            # teaching the LM to reproduce the memorized document.
            ce_mask = loss_masks if ce_enable is None else loss_masks * ce_enable[:, None]
            main_loss = (ce_loss * ce_mask).sum() / (ce_mask.sum() + 1e-9)

            # Compute auxiliary losses (doc_access uses the ungated loss_masks → all rows)
            aux_result = compute_aux_losses(aux_data, loss_masks, input_masks, inputs, aux_cfg)
            total_loss = main_loss * ce_weight + aux_result["total"]
            
            return total_loss, (main_loss, aux_result)
        
        (total_loss, (main_loss, aux_result)), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights)

        # NaN/Inf robustness: skip optimizer update when loss or grads are non-finite
        grad_norm = optax.global_norm(grads)

        # Per-param-group grad norms (weight-0 telemetry). Logged as train/grad_norm_<group>.
        # mem = memory projections, embed = key model, value = Stage-2 value model, main = frozen
        # 4B (should stay ~0). Freeze zeroes *updates* not grads, so read alongside the
        # stage-transition trainable_params log to confirm the A->B embed unfreeze took.
        def _group_norm(grp):
            leaves = [v for k, v in grads.items() if grp in k]
            return optax.global_norm(leaves) if leaves else jnp.array(0.0, dtype=grad_norm.dtype)

        def _norm_where(pred):
            leaves = [v for k, v in grads.items() if pred(k)]
            return optax.global_norm(leaves) if leaves else jnp.array(0.0, dtype=grad_norm.dtype)
        if isinstance(aux_result, dict) and "losses" in aux_result:
            aux_result["losses"]["grad_norm_mem"] = _group_norm("mem_")
            aux_result["losses"]["grad_norm_embed"] = _group_norm("embed_model")
            aux_result["losses"]["grad_norm_value"] = _group_norm("value_model")
            aux_result["losses"]["grad_norm_main"] = _group_norm("main_model")
            # DISJOINT groups. The four above OVERLAP — they are substring matches, so
            # main_model.layers.14.mem_q_proj / main_model.mem_k count in BOTH grad_norm_mem and
            # grad_norm_main. That makes "grad_norm_main is NaN" unreadable: you cannot tell the
            # frozen 4B's own attention/MLP from the memory projections that live inside it.
            # These split it, so a NaN localizes to a subsystem:
            #   main_core  = the 4B proper (attn/mlp/embed_tokens/lm_head/norm)
            #   main_mem   = the memory read/route params that live under main_model
            #   embed_mem  = the embed trunk's mem_k_proj/mem_v_proj
            aux_result["losses"]["grad_norm_main_core"] = _norm_where(
                lambda k: "main_model" in k and "mem_" not in k)
            aux_result["losses"]["grad_norm_main_mem"] = _norm_where(
                lambda k: "main_model" in k and "mem_" in k)
            aux_result["losses"]["grad_norm_embed_core"] = _norm_where(
                lambda k: "embed_model" in k and "mem_" not in k)
            aux_result["losses"]["grad_norm_embed_mem"] = _norm_where(
                lambda k: "embed_model" in k and "mem_" in k)

        loss_finite = jnp.isfinite(total_loss)
        grad_finite = jnp.isfinite(grad_norm)
        is_valid = loss_finite & grad_finite

        def update_fn(args):
            w, s, g = args
            updates, new_s = optimizer.update(g, s, w)
            new_w = optax.apply_updates(w, updates)
            return new_w, new_s

        def skip_fn(args):
            w, s, _ = args
            jax.debug.print(
                "WARNING: loss or grad_norm is not finite (loss={loss}, grad_norm={grad_norm}), skipping update",
                loss=total_loss, grad_norm=grad_norm,
            )
            return w, s

        weights, opt_state = jax.lax.cond(
            is_valid, update_fn, skip_fn, (weights, opt_state, grads)
        )

        # Compute LR inside the JIT so both processes produce it symmetrically —
        # avoids a solo post-step dispatch on process 0 only.
        if lr_schedule is not None and callable(lr_schedule):
            def is_adam(x):
                return hasattr(x, 'mu') and hasattr(x, 'nu')
            adams = [l for l in jax.tree_util.tree_leaves(opt_state, is_leaf=is_adam) if is_adam(l)]
            lr_val = lr_schedule(adams[0].count) if adams else jnp.array(-1.0)
        else:
            lr_val = jnp.array(-1.0)

        return weights, opt_state, main_loss, aux_result, ~loss_finite, ~grad_finite, grad_norm, lr_val
    
    def _run_evals(self, step):
        """
        Run all configured evaluators and return merged metrics.

        Args:
            step: Current training step (for logging context)

        Returns:
            all_metrics: Dict of metric_name -> value from all evaluators
        """
        all_metrics = {}
        for key, evaluator, eval_dataset, doc_dataset in zip(self.eval_keys, self.evaluators, self.evaluator_datasets, self.evaluator_doc_datasets):
            if eval_dataset is None:
                raise ValueError(f"Evaluator '{key}' is missing a dataset configuration. An explicit `dataset` must be provided.")

            extra_kwargs = {"doc_dataset": doc_dataset} if doc_dataset is not None else {}
            metrics = evaluator.evaluate(self.model, eval_dataset, step=step, aux_loss_config=self.aux_loss_config, **extra_kwargs)
            for metric_name, val in metrics.items():
                all_metrics[f"{key}/{metric_name}"] = val
        return all_metrics

    def _current_lr(self):
        """Learning rate at the optimizer's current schedule position (or constant)."""
        if self.lr_schedule is None:
            return -1.0
        if not callable(self.lr_schedule):
            return float(self.lr_schedule)
        def is_adam(x):
            return hasattr(x, 'mu') and hasattr(x, 'nu')
        adams = [l for l in jax.tree_util.tree_leaves(self.opt_state, is_leaf=is_adam) if is_adam(l)]
        if not adams:
            return -1.0
        return float(self.lr_schedule(int(adams[0].count)))

    def _loader_state_path(self, base_dir, step):
        return f"{str(base_dir).rstrip('/')}/{step}/dataloader_state.json"

    def _save_loader_state(self, step):
        """Persist dataloader position next to the orbax checkpoint (warn-only).

        Handles both streaming (dict state from _StreamingIterator.get_state) and
        indexed (bytes state from Grain iterator.get_state) transparently: bytes
        get base64-wrapped so the JSON file works for both. Under is_indexed=True,
        Grain iterator state is small (~KBs) — negligible ckpt overhead.

        Previously guarded by `if is_indexed: return` on the theory that
        sample-at-step is derivable from step+config. That was correct in
        principle but the derivation was never implemented — on resume, the
        indexed iterator restarted at global index 0 instead of step*batch_size,
        so trainer's step N post-resume saw batch (N - restart_step) content
        instead of batch N. Empirically observed: same weights + wrong batch
        content = spuriously lower CE at overlapping resume steps.
        """
        if jax.process_index() != 0:
            return
        try:
            state = self.data.get_loader_state() if hasattr(self.data, "get_loader_state") else None
            if state is None:
                return
            import fsspec, json, base64
            # Grain iterator.get_state() returns bytes; wrap as JSON-safe dict.
            if isinstance(state, (bytes, bytearray)):
                state = {"__bytes_b64": base64.b64encode(bytes(state)).decode("ascii")}
            path = self._loader_state_path(self.checkpoint_manager.directory, step)
            with fsspec.open(path, "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Warning: could not save dataloader state at step {step}: {e}")

    def _restore_loader_state(self, step):
        """Restore dataloader position for a resumed run.

        Priority order:
          1. SKIP_LOADER_RESTORE=1  → skip (start data stream fresh).
          2. Sibling `dataloader_state.json` exists AND passes sanity check
             (indexed path only) → apply verbatim.
          3. Indexed path with missing / stale / invalid file → synthesize
             state at K = step * self.data.batch_size and apply it. This
             recovers correct position from any ckpt (pre-fix, post-fix, even
             one with a corrupted state file), which is what makes preempt +
             tpunanny-respawn cycles resilient.
          4. Streaming path with no synth available → fall back to fast-forward
             from index 0 (current behavior; a warn is logged).

        Undoes the base64 bytes-wrap from _save_loader_state before handing to
        the dataset's set_loader_state.
        """
        # SKIP_LOADER_RESTORE=1 -> resume weights+optimizer but start the data stream fresh.
        # Restoring the position is a REPLAY, not a seek: the saved cursor is a raw-item count, so
        # the loader re-reads/normalizes/tokenizes every item to get back into place (~13M items
        # for a step-38000 resume = hours, and 16 grain workers at ~45 GB each can OOM the host —
        # runbook §5). Skipping trades exact stream continuity for starting immediately.
        # Pair it with a CHANGED dataset.shuffle_seed, or the fresh stream is the same order the
        # run already consumed and you re-show its earliest samples first.
        if os.environ.get("SKIP_LOADER_RESTORE") == "1":
            print(
                "SKIP_LOADER_RESTORE=1: not restoring dataloader position; stream starts from 0. "
                f"Verify dataset.shuffle_seed differs from the original run's "
                f"(now {self.cfg.dataset.get('shuffle_seed')}), else this replays its early data.",
                flush=True,
            )
            return
        if not hasattr(self.data, "set_loader_state"):
            return

        is_indexed = getattr(self.data, "is_indexed", False)
        base = self.resume_from_dir if self.resume_from_dir is not None else self.checkpoint_manager.directory
        path = self._loader_state_path(base, step)

        import fsspec, json, base64

        # Read the sibling state file (may not exist).
        file_state = None
        try:
            with fsspec.open(path, "r") as f:
                file_state = json.load(f)
        except FileNotFoundError:
            print(f"[loader-restore] no state file at {path}", flush=True)

        # Un-wrap the base64 bytes envelope if present (indexed path).
        file_bytes = None
        if isinstance(file_state, dict) and set(file_state.keys()) == {"__bytes_b64"}:
            file_bytes = base64.b64decode(file_state["__bytes_b64"])

        # Path A: indexed. Sanity-check the file (if any) against step*batch_size,
        # else synthesize.
        if is_indexed:
            K = int(step) * int(self.data.batch_size)
            use_file = False
            if file_bytes is not None:
                # last-seen indices should peak around K-1 (fresh state saved
                # exactly at position K); allow slack for Grain's per-worker
                # prefetch buffer.
                try:
                    decoded = json.loads(file_bytes.decode())
                    observed_max = max(
                        int(v) for v in decoded["last_seen_indices"].values()
                    )
                    tol = max(int(0.01 * K), 4 * int(self.data.num_workers) * int(self.data.batch_size))
                    if abs(observed_max - (K - 1)) <= tol:
                        use_file = True
                    else:
                        print(
                            f"[loader-restore] state file at {path} is stale: "
                            f"observed max index {observed_max}, expected ~{K-1} "
                            f"(tolerance {tol}); synthesizing instead.",
                            flush=True,
                        )
                except Exception as e:
                    print(
                        f"[loader-restore] state file at {path} did not decode "
                        f"as Grain state ({e}); synthesizing instead.",
                        flush=True,
                    )
            if use_file:
                self.data.set_loader_state(file_bytes)
                print(f"[loader-restore] restored from file: {path}", flush=True)
            else:
                # Synthesize by handing an int K to set_loader_state; qa.py's
                # generator() materializes it using the fresh iterator's
                # get_state() as a template.
                self.data.set_loader_state(K)
                print(
                    f"[loader-restore] synthesized indexed state at "
                    f"K={K} (step={step} * batch_size={self.data.batch_size}); "
                    f"file={'missing' if file_bytes is None else 'stale'}",
                    flush=True,
                )
            return

        # Path B: streaming (non-indexed). No synth available — fall back to
        # legacy fast-forward, or start from 0 if the file is missing.
        if file_state is None:
            print(f"[loader-restore] no state file and streaming path has no "
                  f"synthesis; stream starts from 0.", flush=True)
            return
        try:
            self.data.set_loader_state(file_state)
            print(f"[loader-restore] restored streaming state from {path}", flush=True)
        except Exception as e:
            print(f"[loader-restore] could not restore streaming state ({e}); "
                  f"stream starts from 0.", flush=True)

    def train(self):
        """
        Run training loop with optional stage transitions.
        """
        # Load from checkpoint if exists
        step, self.opt_state, self.current_stage_idx = load_checkpoint(
            self.checkpoint_manager, self.model, self.opt_state, return_stage_idx=True,
            resume_step=self.resume_step, resume_from_dir=self.resume_from_dir,
        )
        # Stage re-application on resume. The optimizer/loss weights are initialized from
        # stage 0, so a mid-stage restore would silently keep stage-0 settings for the rest
        # of the run unless we re-apply. Two cases:
        #  - Within-stage resume (loaded stage_idx == expected at this step): re-apply the
        #    stage config DIRECTLY while preserving the loaded opt_state. Going through the
        #    transition block below would re-init opt_state and reset Adam's `count` to 0,
        #    which restarts the LR schedule from warmup — losing the schedule position we
        #    just restored.
        #  - Cross-boundary resume (loaded stage != expected): fall through to the sentinel
        #    that forces the transition block, which does want to reset counts for the new
        #    stage's schedule.
        if self.training_stages is not None and step > 0:
            expected_stage_idx = get_current_stage_idx(step, self.training_stages)
            if self.current_stage_idx == expected_stage_idx:
                stage_config = self.training_stages[self.current_stage_idx]
                self.current_trainable_patterns = tuple(stage_config['trainable_params'])
                self.optimizer, self.model, self.lr_schedule = setup_optimizer_for_stage(
                    self.cfg, self.model, stage_config, all_stages=self.training_stages)
                self.ce_weight, self.aux_loss_config = self._get_stage_loss_config(stage_config)
                print(f"[resume] re-applied stage {self.current_stage_idx} config at step {step} "
                      f"(opt_state preserved to keep LR schedule position)")
            else:
                self.current_stage_idx = -1
        # Full resume (step > 0, not a warm start): restore dataloader position so the
        # stream fast-forwards to where the interrupted run left off.
        if step > 0 and self.resume_step is None:
            self._restore_loader_state(step)
        pbar = tqdm(total=self.cfg.trainer.steps)
        loss_nan_count = 0
        grad_nan_count = 0
        # Consecutive non-finite samples (counted at the log cadence, not per step — reading
        # loss_nan/grad_nan forces a device sync, and the loop is pipelined to avoid one per step).
        consecutive_nan_samples = 0

        # ── weight-monitor: bf16 ULP freeze detector ──────────────────────────────────────
        # Snapshot a hand-picked set of weights BEFORE the training loop and diff every
        # log_interval steps. Reads model.weights directly (the fp32 master under the
        # promote_trainable_to_fp32 fix), so it's cast-independent, cancellation-proof,
        # and unambiguous. n_changed == 0 means the weight is silently frozen at the
        # bf16 ULP threshold — the failure mode documented in
        # wiki/implementations/2026-07-27-fp32-master-weights.md.
        #
        # Watched-weight selection follows the three-category hypothesis:
        #   FROZEN category (init ~1.0 or high pretrained magnitude):
        #       ULP >> LR·adam update → w_bf16 + Δw_bf16 rounds Δw to zero every step.
        #   BORDERLINE category (init ~0.02):
        #       ULP ≈ LR·adam update → some elements move, most don't.
        #   ESCAPE / control (zero-init):
        #       ULP=0 at init → first update always lands → sanity-check for "backward alive."
        #
        # Multi-host safety: mem_q_proj / mem_o_proj are sharded P('model','data') /
        # P('data','model'). Under fsdp_devices>1 they are NOT fully addressable on process
        # 0. jax.device_get from process 0 alone would hang on a cross-host all-gather while
        # process 1 races ahead → coordination-service timeout. Fix: keep snapshots as
        # on-device jax.Arrays and compute the diff via a JIT'd function returning a scalar,
        # which is replicated across hosts and safe to bring back. The JIT compiles once,
        # amortized. Runs on ALL processes so the collective is symmetric.
        _WEIGHT_MONITOR_KEYS = [
            # --- memory branch (fresh-init in memory_utils.py) ---
            "main_model.spec_tokens",              # BORDERLINE: init centroid+0.02σ
            # (layer index for memory branch is the FIRST entry of qwen3_mem_embed.yaml's
            # mem_layers — currently [14] for qwen3_mem_embed and [14] for perperiod variants.
            # If a config uses a different mem_layers, these keys won't exist and the
            # `if k in self.model.weights` filter drops them silently — verify the
            # startup log lists them.)
            "main_model.layers.14.mem_q_norm",      # FROZEN: init ones (RMS-norm scale)
            "main_model.layers.14.mem_o_norm",      # FROZEN: init ones
            "main_model.layers.14.mem_layernorm",   # FROZEN: init ones
            "main_model.layers.14.mem_layer_scale", # trapped: init 0.1 scalar (4× short at LR=1e-4)
            "main_model.layers.14.mem_q_proj",      # BORDERLINE: init randn*0.02
            "main_model.layers.14.mem_o_proj",      # ESCAPE / control: zero-init, always moves

            # --- embed-model side memory projections (borderline category) ---
            "embed_model.mem_k_proj",              # BORDERLINE: init randn*0.02
            "embed_model.mem_v_proj",              # BORDERLINE: init randn*0.02

            # --- main_model Qwen3-4B backbone at layers 9/17/26 across the 36-layer stack ---
            # These are PRETRAINED (loaded from Qwen3-4B safetensors) — magnitudes typically
            # cluster near 1.0 by pretraining dynamics for the norm scales, more varied for
            # the projection weights. Startup magnitude log below shows the actual verdict.
            "main_model.norm",                              # final RMSNorm
            "main_model.layers.9.input_layernorm",
            "main_model.layers.9.post_attention_layernorm",
            "main_model.layers.9.q_norm",                   # Qwen3 QK-norm on Q
            "main_model.layers.9.k_norm",                   # QK-norm on K
            "main_model.layers.17.input_layernorm",
            "main_model.layers.17.q_norm",
            "main_model.layers.26.input_layernorm",
            "main_model.layers.26.q_norm",

            # main_model projections at 2 layers (varied pretrained magnitudes)
            "main_model.layers.9.q_proj",
            "main_model.layers.9.up_proj",
            "main_model.layers.26.q_proj",
            "main_model.layers.26.up_proj",
        ]
        # Snapshot: reference the current jax.Array, no host materialization. optimizer.update
        # returns fresh arrays each step (immutable jax), so the ref is stable — it captures
        # the weight state at the moment of snapshot regardless of subsequent updates.
        _wmon_prev = {k: self.model.weights[k]
                      for k in _WEIGHT_MONITOR_KEYS if k in self.model.weights}

        # ULP fraction per dtype: 1 ULP ≈ |w| × 2^(-mantissa_bits).
        # bf16=7 mantissa bits (implicit + 7 explicit), fp16=10, fp32=23.
        def _ulp_scale_for(dtype):
            if dtype == jnp.bfloat16: return 2.0 ** -7
            if dtype == jnp.float16:  return 2.0 ** -10
            if dtype == jnp.float32:  return 2.0 ** -23
            return 2.0 ** -7  # default assume bf16

        # JIT'd per-step stats: (n_changed, wnorm_rms, delta_abs_mean, max_ulp_ratio).
        # All outputs are scalars replicated across hosts, so device_get on process 0 is safe.
        # ulp_scale is a static compile-time arg (varies by dtype).
        @partial(jax.jit, static_argnames=("ulp_scale",))
        def _wmon_stats(prev, curr, ulp_scale):
            p32 = prev.astype(jnp.float32)
            c32 = curr.astype(jnp.float32)
            n_changed = jnp.sum(prev != curr).astype(jnp.int64)
            wnorm_rms = jnp.sqrt(jnp.mean(c32 ** 2))
            delta_abs = jnp.abs(c32 - p32)
            delta_abs_mean = jnp.mean(delta_abs)
            # |Δw| / ULP(|w|): >1 means the update lands in dtype's precision;
            # <1 means it rounds to zero on the next w_dtype + Δw_dtype add.
            ulp = jnp.abs(c32) * ulp_scale + 1e-30
            max_ulp_ratio = jnp.max(delta_abs / ulp)
            return n_changed, wnorm_rms, delta_abs_mean, max_ulp_ratio

        # JIT'd startup magnitude stats (rms, mean|w|, max|w|, min|w|). Same
        # multi-host-safety pattern as above — scalar outputs are replicated.
        @jax.jit
        def _wmon_startup_stats(w):
            w32 = w.astype(jnp.float32)
            abs_w = jnp.abs(w32)
            return (jnp.sqrt(jnp.mean(w32 ** 2)),
                    jnp.mean(abs_w),
                    jnp.max(abs_w),
                    jnp.min(abs_w))

        # Store on the instance so the per-step block can access.
        self._wmon_keys = list(_wmon_prev.keys())
        self._wmon_prev = _wmon_prev
        self._wmon_stats_fn = _wmon_stats
        self._wmon_ulp_scale_by_key = {k: _ulp_scale_for(v.dtype)
                                       for k, v in _wmon_prev.items()}

        # Compute startup magnitude stats on ALL hosts (JIT reductions ⇒ replicated scalar).
        _startup_stats_by_key = {}
        for k, v in _wmon_prev.items():
            rms_arr, mean_abs_arr, max_abs_arr, min_abs_arr = _wmon_startup_stats(v)
            _startup_stats_by_key[k] = (float(rms_arr), float(mean_abs_arr),
                                        float(max_abs_arr), float(min_abs_arr))

        # Log on process 0 only. Verdict rule at LR=1e-4 target update:
        #   ULP@mean|w| > 1e-3 → FROZEN (update rounds away most of the time)
        #   1e-4 < ULP@mean|w| ≤ 1e-3 → MIXED (borderline)
        #   ULP@mean|w| ≤ 1e-4 → OK (updates land)
        if jax.process_index() == 0:
            print(f"\n[weight-monitor] snapshotting {len(_wmon_prev)} watched weights for elementwise diff:")
            for k, v in _wmon_prev.items():
                rms, mean_abs, max_abs, min_abs = _startup_stats_by_key[k]
                ulp_scale = self._wmon_ulp_scale_by_key[k]
                ulp_typ = mean_abs * ulp_scale
                if ulp_typ > 1e-3:
                    verdict = "FROZEN"
                elif ulp_typ > 1e-4:
                    verdict = "MIXED"
                else:
                    verdict = "OK"
                print(f"  {k:<52}  shape={list(v.shape)}  dtype={v.dtype}  size={v.size:>10d}  "
                      f"rms={rms:.4g}  mean|w|={mean_abs:.4g}  ulp={ulp_typ:.4g}  {verdict}")
            print()

        while step < self.cfg.trainer.steps:
            
            for tokens, masks in self.data.generator():
                if step >= self.cfg.trainer.steps:
                    break
                
                if self.training_stages is not None:
                    expected_stage_idx = get_current_stage_idx(step, self.training_stages)

                    # Check if we need to transition to next stage
                    if expected_stage_idx != self.current_stage_idx:
                        stage_config = self.training_stages[expected_stage_idx]
                        print(f"\n=== Transitioning to Stage {expected_stage_idx} at step {step} ===")
                        print(f"Trainable params: {stage_config['trainable_params']}")
                        self.current_trainable_patterns = tuple(stage_config['trainable_params'])

                        # Reconstruct the optimizer for the new stage's trainable-params mask / LR
                        # schedule. The mask lives in the optimizer OBJECT (a static/structural
                        # argument to optax.transforms.freeze), not in opt_state's array data, so
                        # swapping it never requires touching opt_state. mu/nu are supposed to
                        # carry over UNCHANGED across a transition — there's nothing to "transfer,"
                        # it never left — only the schedule-related step counter(s) need resetting
                        # to 0 so the new stage's warmup/cosine schedule starts fresh.
                        #
                        # Previously this rebuilt a full second opt_state via
                        # self.optimizer.init(...) (allocating fresh mu/nu for EVERY param, full
                        # size, regardless of trainability — see
                        # utils.py::setup_optimizer_for_stage's own comment on that cost) while the
                        # OLD opt_state was still referenced, then immediately discarded the fresh
                        # mu/nu in favor of the old ones. That transient ~2x optimizer-state
                        # footprint is the diagnosed cause of a reproducible OOM at this exact
                        # boundary (over budget by ~1.7-1.9MB on a v6e-8 slice — see
                        # wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md).
                        # This in-place reset is verified byte-identical to the old rebuild-and-
                        # patch behavior in tests/test_stage_transition_opt_state.py (that test
                        # also caught a real bug in an earlier draft of this fix: resetting only
                        # the Adam-moments node's count missed a SEPARATE schedule-counter node —
                        # hence checking every count-bearing node via `_fields`, not just `is_adam`).
                        self.optimizer, self.model, self.lr_schedule = setup_optimizer_for_stage(self.cfg, self.model, stage_config, all_stages=self.training_stages)

                        def _has_count(x):
                            # Not hasattr(x, 'count'): every plain tuple has a built-in .count()
                            # METHOD, unrelated to a state namedtuple's count FIELD.
                            return 'count' in getattr(x, '_fields', ())

                        def _reset_count(leaf):
                            return leaf._replace(count=jnp.zeros_like(leaf.count))

                        self.opt_state = jax.tree_util.tree_map(_reset_count, self.opt_state, is_leaf=_has_count)

                        # Update loss weights for new stage
                        self.ce_weight, self.aux_loss_config = self._get_stage_loss_config(stage_config)
                        print(f"CE weight: {float(self.ce_weight)}")

                        self.current_stage_idx = expected_stage_idx

                        # Log stage transition
                        if self.cfg.trainer.get("use_wandb", False) and jax.process_index() == 0:
                            # train/step is the explicit x-axis (see train.py's define_metric):
                            # shared mode ignores wandb.log's step= kwarg, so put the step INTO
                            # the log dict as train/step. Kept in the dict for non-shared too
                            # so both modes agree.
                            wandb.log({
                                "train/step": step,
                                "stage": self.current_stage_idx,
                                "stage_trainable_params": str(stage_config['trainable_params']),
                                "stage_ce_weight": float(self.ce_weight),
                                "step": step
                            }, step=step)

                inputs, targets, input_masks, loss_masks, ce_enable = process_train_pairs(tokens, masks)

                # Axis A: pass the trainable patterns (=> stop_gradient frozen weights) only when
                # enabled and not "all"-trainable (which is a no-op anyway). Off => None => unchanged.
                _tp = self.current_trainable_patterns
                _pass_tp = _tp if (self.stop_grad_frozen and _tp and "all" not in _tp) else None
                self.model.weights, self.opt_state, ce_loss, aux_result, loss_nan, grad_nan, grad_norm, step_lr = self._train_step(
                    self.model.forward, self.optimizer, self.model.weights, self.opt_state,
                    inputs, targets, input_masks, loss_masks, self.ce_weight, self.aux_loss_config, ce_enable,
                    _pass_tp, self.lr_schedule
                )

                # Pipeline the loop: pulling losses to the host forces a device sync every step,
                # serializing the drain+redispatch bubble that would otherwise hide behind the next
                # step's compute (~10% of real step time — wiki/experiments/2026-07-16-train-speed-axes.md,
                # Bottleneck B). Only pull/log every `log_interval` steps so JAX dispatches ahead.
                # Default log_interval=1 is byte-identical to the old per-step behavior. The on-device
                # NaN guard in _train_step skips bad updates EVERY step regardless; only the host-side
                # nan_count telemetry is sampled at the log cadence.
                log_interval = int(self.cfg.trainer.get("log_interval", 1))
                # 0 disables the tripwire. Default 50 samples = 500 steps at log_interval=10.
                nan_abort_after_samples = int(self.cfg.trainer.get("nan_abort_after_samples", 50))
                is_last = (step + 1) >= self.cfg.trainer.steps
                if step % log_interval == 0 or is_last:
                    loss_nan_count += int(loss_nan)
                    grad_nan_count += int(grad_nan)

                    # Tripwire for the silent-deadlock failure mode. skip_fn returns weights AND
                    # opt_state unchanged, so a persistent NaN grad freezes the weights, Adam's
                    # count never advances, and the LR stays pinned at the schedule's init_value.
                    # The run then looks alive (tqdm ticks, loss prints ~0.8) while making exactly
                    # zero progress — the stage-3 donation bug burned ~3900 steps this way before
                    # anyone noticed. Abort loudly instead of grinding overnight.
                    if bool(loss_nan) or bool(grad_nan):
                        consecutive_nan_samples += 1
                        if nan_abort_after_samples > 0 and consecutive_nan_samples >= nan_abort_after_samples:
                            raise RuntimeError(
                                f"Aborting: {consecutive_nan_samples} consecutive non-finite "
                                f"samples (~{consecutive_nan_samples * log_interval} steps) at step "
                                f"{step}. Weights and opt_state are frozen and the LR is pinned at "
                                f"{float(step_lr):.3g} — this run is making no progress. "
                                f"Set trainer.nan_abort_after_samples=0 to disable this check."
                            )
                    else:
                        consecutive_nan_samples = 0
                    # Compute total loss for display (ce_weight * ce_loss + aux_total)
                    ce_weight_val = float(self.ce_weight)
                    total_loss = ce_weight_val * float(ce_loss) + float(aux_result.get("total", 0.0))
                    pbar.set_description(f"Loss: {total_loss:.4f} | CE: {ce_loss:.4f} (w={ce_weight_val})")

                    # weight-monitor: elementwise diff on watched weights (reads fp32 master).
                    # MUST run on ALL hosts — the JIT'd stats function touches sharded weights
                    # (mem_q_proj sharded P('model','data'), mem_o_proj P('data','model')),
                    # so calling triggers cross-host collectives. Skip on one host and you
                    # get "unexpected peer in launch group" + SW_INJECT_ERROR shutdown.
                    # Compute-side runs symmetrically; only the log/print gated on process 0.
                    wmon_by_key = {}
                    for k in self._wmon_keys:
                        prev = self._wmon_prev[k]
                        curr = self.model.weights[k]
                        ulp_scale = self._wmon_ulp_scale_by_key[k]
                        n_changed_arr, rms_arr, delta_arr, ulp_r_arr = self._wmon_stats_fn(
                            prev, curr, ulp_scale
                        )
                        # All outputs are replicated scalars, safe to bring to host.
                        wmon_by_key[k] = (
                            int(n_changed_arr), float(rms_arr),
                            float(delta_arr), float(ulp_r_arr),
                            int(curr.size), curr.dtype,
                        )
                        self._wmon_prev[k] = curr

                    # Log to W&B (process 0 only)
                    if self.cfg.trainer.get("use_wandb", False) and jax.process_index() == 0:
                        # train/step is the explicit x-axis (see train.py's define_metric):
                        # shared mode ignores wandb.log's step= kwarg, so put the step INTO
                        # the log dict as train/step. This makes resumes plot at the true
                        # training step instead of a per-process auto-increment.
                        log_dict = {
                            "train/step": step,
                            "train/total_loss": total_loss,
                            "train/ce_loss": float(ce_loss),
                            "train/ce_weight": ce_weight_val,
                            "train/grad_norm": float(grad_norm),
                            "train/lr": float(step_lr),
                            "train/loss_nan_count": loss_nan_count,
                            "train/grad_nan_count": grad_nan_count,
                            "step": step,
                        }
                        # Log individual aux losses
                        for aux_name, aux_val in aux_result.get("losses", {}).items():
                            log_dict[f"train/{aux_name}"] = float(aux_val)

                        # Group classification for per-group aggregate. ORDERED
                        # regex match — mem-first is load-bearing: `main_model.
                        # layers.14.mem_q_proj` contains BOTH "main_model" and
                        # "mem_", must land in mem. Same for embed_model.mem_*.
                        # Must match utils.py::setup_optimizer_for_stage.
                        _WRATIO_GROUP_PATTERNS = [
                            ("mem",   re.compile(r"(mem_|spec_tokens|embed_proj_conv)")),
                            ("embed", re.compile(r"embed_model")),
                            ("main",  re.compile(r"main_model")),
                        ]
                        def _wratio_group_for(key):
                            for name, pat in _WRATIO_GROUP_PATTERNS:
                                if pat.search(key):
                                    return name
                            return None

                        group_ratios = {"mem": [], "embed": [], "main": []}
                        for k, (n_changed, rms, delta_abs, ulp_ratio, total, dtype) in wmon_by_key.items():
                            # All weight-monitor keys share the `weight/` top-level prefix so
                            # wandb groups them under one "weight" section in the UI (alongside
                            # the existing "train", "charts", "system" sections).
                            log_dict[f"weight/wchanged/{k}"]             = n_changed / max(total, 1)
                            log_dict[f"weight/wchanged/{k}/n"]           = n_changed
                            log_dict[f"weight/wnorm/{k}/rms"]            = rms
                            log_dict[f"weight/wdelta/{k}/abs_mean"]      = delta_abs
                            log_dict[f"weight/wdelta/{k}/max_ulp_ratio"] = ulp_ratio
                            # wratio = |Δw|/|w| = scale-invariant per-step change.
                            # Healthy band ~[1e-4, 1e-3]: below → weight barely
                            # moves over 100k steps → under-trained; above → step
                            # motion overwhelms Adam SNR → unstable.
                            ratio = float(delta_abs) / max(float(rms), 1e-12)
                            log_dict[f"weight/wratio/{k}"] = ratio
                            grp = _wratio_group_for(k)
                            if grp is not None:
                                group_ratios[grp].append(ratio)
                            # Grep-able console line — first 20 steps, then every 100.
                            if step < 20 or step % 100 == 0:
                                print(f"[wmon step={step}] {k}: n_changed={n_changed}/{total} "
                                      f"({100*n_changed/max(total,1):.1f}%) rms={rms:.4g} "
                                      f"Δ_mean={delta_abs:.4g} max_ulp_r={ulp_ratio:.4g} "
                                      f"ratio={ratio:.4g} dtype={dtype}",
                                      flush=True)
                        # Per-group aggregates (mean of per-key ratios in each
                        # group). Read this against the [1e-4, 1e-3] band to
                        # gauge whether the group's effective LR is too high
                        # (unstable) or too low (under-trained).
                        for grp, ratios in group_ratios.items():
                            if ratios:
                                log_dict[f"weight/wratio_group_mean/{grp}"] = sum(ratios) / len(ratios)

                        # Per-group |m|/√v: Adam's normalized-update magnitude,
                        # approximated as mean(|Δw|) / effective_lr for keys in
                        # the group. Diagnoses whether small |Δw| is
                        # gradient-noise-dominated (m/√v ~ SNR ~ 0.2 = pure-noise
                        # floor √((1-β1)/(1+β1)) for β1=0.9) or LR-limited.
                        #
                        # If |m|/√v sits at the noise floor even at a bigger LR,
                        # raising LR further won't help — the group needs bigger
                        # batches (to reduce grad noise), not more LR.
                        #
                        # Approximation caveats: |Δw| = lr * (m/√v ± wd·w),
                        # so we're off by up to wd·w/effective_lr per weight.
                        # For wd=0.1 and |w|=0.024 at lr=1e-5, that's ~2.4% error;
                        # tolerable for a diagnostic. Also ignores clip_by_global_norm.
                        lr_by_group = self.cfg.trainer.get("learning_rates") if hasattr(self.cfg.trainer, "get") else None
                        base_lr = float(step_lr) if step_lr is not None else None
                        group_delta_sums = {"mem": 0.0, "embed": 0.0, "main": 0.0}
                        group_delta_counts = {"mem": 0, "embed": 0, "main": 0}
                        for k, (n_changed, rms, delta_abs, ulp_ratio, total, dtype) in wmon_by_key.items():
                            grp = _wratio_group_for(k)
                            if grp is not None:
                                group_delta_sums[grp] += float(delta_abs)
                                group_delta_counts[grp] += 1
                        if base_lr is not None and base_lr > 0:
                            if lr_by_group:
                                peak_mem = float(lr_by_group.get("mem", base_lr))
                                lr_eff = {
                                    "mem":   base_lr,
                                    "embed": base_lr * float(lr_by_group.get("embed", peak_mem)) / peak_mem,
                                    "main":  base_lr * float(lr_by_group.get("main",  peak_mem)) / peak_mem,
                                }
                            else:
                                lr_eff = {"mem": base_lr, "embed": base_lr, "main": base_lr}
                            for grp, count in group_delta_counts.items():
                                if count > 0 and lr_eff[grp] > 0:
                                    mean_delta = group_delta_sums[grp] / count
                                    log_dict[f"weight/mnorm_group_mean/{grp}"] = mean_delta / lr_eff[grp]

                        wandb.log(log_dict, step=step)
                pbar.update(1)
                
                # Evaluation
                if self.cfg.trainer.eval_interval > 0 and (step + 1) % self.cfg.trainer.eval_interval == 0:
                    eval_metrics = self._run_evals(step + 1)
                    if self.cfg.trainer.get("use_wandb", False) and jax.process_index() == 0 and eval_metrics and not self.cfg.trainer.get("evals"):
                        # train/step for the shared-mode x-axis (see train.py's define_metric).
                        val_log = {"train/step": step + 1, "step": step + 1}
                        for k, v in eval_metrics.items():
                            val_log[f"val/{k}"] = float(v)
                        wandb.log(val_log, step=step + 1)
                
                # Checkpointing
                if step % self.cfg.trainer.checkpoint_interval == 0 and step > 0:
                    save_checkpoint(self.checkpoint_manager, self.model, self.opt_state, step, self.current_stage_idx)
                    self._save_loader_state(step)

                step += 1

        # Final save
        save_checkpoint(self.checkpoint_manager, self.model, self.opt_state, step, self.current_stage_idx)
        self._save_loader_state(step)
        self.checkpoint_manager.wait_until_finished()
