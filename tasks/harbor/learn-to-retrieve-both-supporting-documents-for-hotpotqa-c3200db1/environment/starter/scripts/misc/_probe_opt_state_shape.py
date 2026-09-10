"""Probe the actual opt_state pytree shape + sharding after optimizer.init.

Replicates exactly what trainer.__init__ does (imports the model, creates the optimizer,
calls optimizer.init(model.weights)), then walks the pytree and prints each leaf's
sharding, focusing on the Adam count scalar.

Answers: is opt_state.count actually replicated across all 8 chips, or does it inherit
a sharding that puts it on a specific device?"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/memory-layers"))

import socket
HOSTNAME = socket.gethostname()

print(f"[{HOSTNAME}] loading utils + init_jax_distributed", flush=True)
from utils import init_jax_distributed, setup_optimizer_for_stage, parse_training_stages
init_jax_distributed()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

print(f"[{HOSTNAME}] jax.process_index()={jax.process_index()} n_devices={jax.device_count()}", flush=True)

# Compose the same Hydra config the training run uses. Uses spec64 to match the smoke.
root = os.path.expanduser("~/memory-layers")
cfg_dir = os.path.join(root, "configs")
with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
    cfg = compose(config_name="train", overrides=[
        "model=qwen3_mem_embed_spec64",
        "trainer=staged",
        f"trainer.tp_devices={os.environ.get('PROBE_TP_DEVICES', '1')}",
        "dataset=qa_hard_neg_think_sft4b",
        "dataset.num_workers=0",
        "trainer.steps=20000",
        "trainer.eval_interval=1000000000",
        "eval_set@trainer.evals=none",
    ])

print(f"[{HOSTNAME}] loading model", flush=True)
from models import get_model
t0 = time.time()
model = get_model(cfg.model, cfg.trainer.tp_devices)
print(f"[{HOSTNAME}] model loaded ({time.time()-t0:.1f}s)", flush=True)

# Build the optimizer via the same code path training uses
stages = parse_training_stages(cfg)
stage_0 = stages[0] if stages else None
print(f"[{HOSTNAME}] building optimizer for stage 0", flush=True)
optimizer, model, _lr_sched = setup_optimizer_for_stage(cfg, model, stage_0, stages)

t0 = time.time()
opt_state = optimizer.init(model.weights)
print(f"[{HOSTNAME}] optimizer.init done ({time.time()-t0:.1f}s)", flush=True)

# Walk pytree, find count scalars
print(f"[{HOSTNAME}] === opt_state pytree structure ===", flush=True)
print(f"[{HOSTNAME}] top-level type: {type(opt_state).__name__}", flush=True)

def find_scalars(x, path=""):
    """Yield (path, leaf) for every leaf that is a scalar array."""
    if hasattr(x, "shape") and hasattr(x, "sharding"):
        if x.shape == ():
            yield (path, x)
        return
    if isinstance(x, (list, tuple)):
        for i, e in enumerate(x):
            yield from find_scalars(e, f"{path}[{i}]")
    elif hasattr(x, "__dict__"):
        for k, v in vars(x).items():
            if k.startswith("_"): continue
            yield from find_scalars(v, f"{path}.{k}")
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from find_scalars(v, f"{path}[{k!r}]")

# Try to find any scalars (count, etc.)
n_shown = 0
for path, leaf in find_scalars(opt_state):
    print(f"[{HOSTNAME}] SCALAR: path={path}", flush=True)
    print(f"[{HOSTNAME}]   dtype={leaf.dtype} shape={leaf.shape}", flush=True)
    print(f"[{HOSTNAME}]   sharding={leaf.sharding!r}", flush=True)
    try:
        addressable_shards = leaf.addressable_shards
        print(f"[{HOSTNAME}]   n_addressable_shards={len(addressable_shards)}", flush=True)
        for i, s in enumerate(addressable_shards):
            print(f"[{HOSTNAME}]     shard[{i}]: device={s.device} data={s.data}", flush=True)
    except Exception as e:
        print(f"[{HOSTNAME}]   addressable_shards err: {e}", flush=True)
    # THE test: try float() on it, same as trainer.py:335
    try:
        v = float(leaf)
        print(f"[{HOSTNAME}]   float() OK: {v}", flush=True)
    except Exception as e:
        print(f"[{HOSTNAME}]   float() FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
    n_shown += 1
    if n_shown >= 10:
        break

print(f"[{HOSTNAME}] === walking chain for Adam-count specifically ===", flush=True)
# opt_state is a tuple of one-per-transformation states. Adamw's is ScaleByAdamState.
def find_adam(x, path=""):
    if hasattr(x, "mu") and hasattr(x, "nu") and hasattr(x, "count"):
        yield (path, x)
    if isinstance(x, (list, tuple)):
        for i, e in enumerate(x):
            yield from find_adam(e, f"{path}[{i}]")
    if hasattr(x, "inner_state"):
        yield from find_adam(x.inner_state, f"{path}.inner_state")

for path, adam_state in find_adam(opt_state):
    print(f"[{HOSTNAME}] ADAM: path={path}", flush=True)
    c = adam_state.count
    print(f"[{HOSTNAME}]   count.dtype={c.dtype} count.shape={c.shape}", flush=True)
    print(f"[{HOSTNAME}]   count.sharding={c.sharding!r}", flush=True)
    try:
        print(f"[{HOSTNAME}]   float(count) = {float(c)}", flush=True)
    except Exception as e:
        print(f"[{HOSTNAME}]   float(count) FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)

# The main question: are mu/nu sharded the same way as the weights, or replicated?
# Compare each moment leaf's sharding against the corresponding weight leaf's sharding.
print(f"[{HOSTNAME}] === mu/nu vs weight sharding audit ===", flush=True)

def _spec_str(sh):
    """PartitionSpec string, or 'REPLICATED' for a fully-replicated NamedSharding."""
    try:
        spec = sh.spec
        if all(a is None for a in tuple(spec)) or len(tuple(spec)) == 0:
            return "REPLICATED"
        return str(spec)
    except AttributeError:
        return repr(sh)

def flat_arrays(x, path=""):
    """Yield (path, array) for every leaf that is a jax array with shape+sharding."""
    if hasattr(x, "shape") and hasattr(x, "sharding") and hasattr(x, "dtype"):
        yield (path, x)
        return
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat_arrays(v, f"{path}[{k!r}]")
    elif isinstance(x, (list, tuple)):
        for i, e in enumerate(x):
            yield from flat_arrays(e, f"{path}[{i}]")
    elif hasattr(x, "__dict__"):
        for k, v in vars(x).items():
            if k.startswith("_"): continue
            yield from flat_arrays(v, f"{path}.{k}")

# Flat weight index: bare key → sharding string
weight_shard = {k: _spec_str(v.sharding) for k, v in model.weights.items()
                if hasattr(v, "sharding")}

# Walk opt_state, extract mu/nu leaves. The weight's leaf key ends the path
# (e.g. "...mu['main_model.layers.0.self_attn.q_proj.kernel']").
moment_rows = []
for path, adam_state in find_adam(opt_state):
    for moment_name in ("mu", "nu"):
        moment = getattr(adam_state, moment_name, None)
        if moment is None:
            continue
        for lp, leaf in flat_arrays(moment, f"{path}.{moment_name}"):
            # Try to extract the terminal weight key from the leaf path
            wkey = lp.rsplit("[", 1)[-1].rstrip("]").strip("'\"") if "[" in lp else lp
            w_shard = weight_shard.get(wkey, "<no matching weight>")
            m_shard = _spec_str(leaf.sharding)
            match = "OK" if w_shard == m_shard else "MISMATCH"
            moment_rows.append((match, moment_name, wkey, w_shard, m_shard, leaf.shape))

# Summary counts
from collections import Counter
match_counts = Counter(r[0] for r in moment_rows)
print(f"[{HOSTNAME}] moment leaves total={len(moment_rows)}  {dict(match_counts)}", flush=True)

# Print first 8 rows, plus first 8 MISMATCH rows if any
shown = 0
for r in moment_rows:
    if shown < 8:
        print(f"[{HOSTNAME}]   [{r[0]}] {r[1]:2s}  {r[2][:70]:70s}  weight={r[3]}  moment={r[4]}  shape={r[5]}", flush=True)
        shown += 1
mismatches = [r for r in moment_rows if r[0] == "MISMATCH"]
if mismatches:
    print(f"[{HOSTNAME}] --- MISMATCH samples (first 8 of {len(mismatches)}) ---", flush=True)
    for r in mismatches[:8]:
        print(f"[{HOSTNAME}]   {r[1]}  {r[2][:70]:70s}  weight={r[3]}  moment={r[4]}  shape={r[5]}", flush=True)

# Per-chip HBM contribution proxy: sum bytes on this process's local devices
try:
    total_moment_bytes_local = 0
    for r in moment_rows:
        shape = r[5]
        n_elems = 1
        for d in shape: n_elems *= int(d)
        # Assume bf16 (2 bytes) for moments — optax adamw defaults follow the param dtype
        total_moment_bytes_local += n_elems * 2
    print(f"[{HOSTNAME}] total moment bytes across all leaves (unsharded logical): {total_moment_bytes_local/1e9:.2f} GB", flush=True)
except Exception as e:
    print(f"[{HOSTNAME}] size probe err: {e}", flush=True)


# ────────────────────────────────────────────────────────────────────────────────────────────
# ACTIVATIONS probe: measure per-chip HBM before/after a real _train_step to isolate the
# activation footprint (peak - post-opt_state.init baseline).
# ────────────────────────────────────────────────────────────────────────────────────────────
print(f"[{HOSTNAME}] === per-chip HBM usage ===", flush=True)

def print_hbm(tag):
    for i, dev in enumerate(jax.local_devices()):
        stats = dev.memory_stats() if hasattr(dev, "memory_stats") else None
        if not stats:
            print(f"[{HOSTNAME}] {tag}  chip[{i}] (no memory_stats)", flush=True)
            continue
        in_use = stats.get('bytes_in_use', 0) / 1e9
        peak   = stats.get('peak_bytes_in_use', 0) / 1e9
        limit  = stats.get('bytes_limit', 0) / 1e9
        reservable = stats.get('bytes_reservable_limit', 0) / 1e9
        print(f"[{HOSTNAME}] {tag}  chip[{i}] in_use={in_use:.2f}GB peak={peak:.2f}GB "
              f"limit={limit:.2f}GB reservable={reservable:.2f}GB", flush=True)

print_hbm("post_opt_init  ")  # baseline: weights + opt_state moments only

# ---- Build a synthetic batch matching the real train-time shapes -----------
# Mirrors scripts/embed/bench_approx_topk.py::_one_batch — process_train_pairs
# with random tokens (values don't matter; the step-cost depends only on shape).
from utils import process_train_pairs, freeze_dict
B = int(cfg.dataset.batch_size)
seq_len = int(cfg.dataset.seq_len)
M = int(cfg.dataset.num_chunks_per_doc)
d_seq = int(cfg.dataset.doc_chunk_seq_len)
k1, k2 = jax.random.split(jax.random.PRNGKey(0))
vocab = 1000
tokens = {
    "batch": jax.random.randint(k1, (B, seq_len), 1, vocab, dtype=jnp.int32),
    "docs":  jax.random.randint(k2, (B * M, d_seq), 1, vocab, dtype=jnp.int32),
}
masks = {
    "batch_mask":   jnp.ones((B, seq_len), dtype=jnp.int32),
    "docs_mask":    jnp.ones((B * M, d_seq), dtype=jnp.float32),
    "loss_mask":    jnp.ones((B, seq_len), dtype=jnp.float32),
    "pos_doc_mask": jnp.ones((B, M), dtype=jnp.int32),
    "ce_enable":    jnp.ones((B,), dtype=jnp.float32),
}
# Spec-token models need a spec_token_mask (all-zeros = no spec tokens in this synthetic batch,
# guarded by the model's `num_spec_tokens` config).
if cfg.model.get("memory", {}).get("num_spec_tokens", 0) > 0:
    masks["spec_token_mask"] = jnp.zeros((B, seq_len), dtype=jnp.int32)

inputs, targets, input_masks, loss_masks, ce_enable = process_train_pairs(tokens, masks)

# Stage-0 aux loss config (avoid instantiating the trainer to skip eval-dataset loads)
base = {k: dict(v) for k, v in cfg.trainer.aux_losses.items()}
stage0 = cfg.trainer.training_stages[0] if cfg.trainer.get("training_stages") else {}
for loss_name, ov in dict(stage0.get("aux_losses", {})).items():
    if loss_name in base:
        base[loss_name].update(ov)
    else:
        base[loss_name] = dict(ov)
aux_loss_config = freeze_dict(base)
ce_weight = jnp.array(float(stage0.get("ce_weight", cfg.trainer.get("ce_weight", 1.0))))

from trainer.trainer import Trainer
print_hbm("pre_train_step ")
t0 = time.time()
w, opt_state, ce_loss, aux_result, loss_nan, grad_nan, grad_norm, _lr = Trainer._train_step(
    model.forward, optimizer, model.weights, opt_state,
    inputs, targets, input_masks, loss_masks, ce_weight, aux_loss_config, ce_enable,
    None, None,  # trainable_patterns, lr_schedule
)
float(ce_loss)  # force device sync so the step actually completes
print(f"[{HOSTNAME}] train_step done ({time.time()-t0:.1f}s), ce_loss={float(ce_loss):.4f}", flush=True)
print_hbm("post_train_step")  # peak_bytes_in_use here captures the activation peak

# ---- Re-verify mu/nu sharding survived the optimizer update ---------------
print(f"[{HOSTNAME}] === mu/nu sharding AFTER 1 real train_step ===", flush=True)
moment_rows_post = []
for path, adam_state in find_adam(opt_state):
    for moment_name in ("mu", "nu"):
        moment = getattr(adam_state, moment_name, None)
        if moment is None:
            continue
        for lp, leaf in flat_arrays(moment, f"{path}.{moment_name}"):
            wkey = lp.rsplit("[", 1)[-1].rstrip("]").strip("'\"") if "[" in lp else lp
            w_shard = weight_shard.get(wkey, "<no matching weight>")
            m_shard = _spec_str(leaf.sharding)
            match = "OK" if w_shard == m_shard else "MISMATCH"
            moment_rows_post.append((match, moment_name, wkey, w_shard, m_shard, leaf.shape))
match_counts_post = Counter(r[0] for r in moment_rows_post)
print(f"[{HOSTNAME}] post-step moment leaves total={len(moment_rows_post)}  {dict(match_counts_post)}", flush=True)
mismatches_post = [r for r in moment_rows_post if r[0] == "MISMATCH"]
if mismatches_post:
    print(f"[{HOSTNAME}] --- POST-STEP MISMATCH samples (first 5 of {len(mismatches_post)}) ---", flush=True)
    for r in mismatches_post[:5]:
        print(f"[{HOSTNAME}]   {r[1]}  {r[2][:70]:70s}  weight={r[3]}  moment={r[4]}  shape={r[5]}", flush=True)


# ────────────────────────────────────────────────────────────────────────────────────────────
# EVAL probe: measure per-chip HBM through the actual eval path (embed → un-shard → generate).
# Simulates GenerationEmbedEvaluator.evaluate at the shapes from
# configs/eval/tasks/gen_embed_science_qa_hard_neg_think.yaml:
#   num_samples=32, batch_size=32, seq_len=2048, num_chunks_per_doc=16, doc_chunk_seq_len=256,
#   max_new_tokens=512
# ────────────────────────────────────────────────────────────────────────────────────────────
print(f"[{HOSTNAME}] === EVAL memory probe (embed + allgather + generate) ===", flush=True)

# Eval-config shapes
E_B = 32
E_seq_len = 2048
E_M = 16
E_d_seq = 256
E_max_new = 512

try:
    from models.qwen3_mem_embed import embed_forward
    from models.utils import split_weights
    from inference import _generate_tokens
    from functools import partial

    _main_w, embed_w = split_weights(model.weights, ["main_model", "embed_model"])
    embed_cfg = model.cfg["embed_model"]

    def embed_fn(docs, dmask):
        return embed_forward(embed_cfg, docs, embed_w, dmask)

    # Synthetic eval-shape docs (sharded on data axis, matches evaluator line 238-244)
    k_docs, k_prompt = jax.random.split(jax.random.PRNGKey(1))
    docs_e = jax.device_put(
        jax.random.randint(k_docs, (E_B * E_M, E_d_seq), 1, vocab, dtype=jnp.int32),
        jax.sharding.PartitionSpec("data", None)
    )
    dmask_e = jax.device_put(
        jnp.ones((E_B * E_M, E_d_seq), dtype=jnp.bool_),
        jax.sharding.PartitionSpec("data", None)
    )

    print_hbm("pre_embed_fn  ")
    t_e = time.time()
    mem_k, mem_v, mem_mask_flat, eff = embed_fn(docs_e, dmask_e)
    mem_k.block_until_ready()
    print(f"[{HOSTNAME}] embed_fn done ({time.time()-t_e:.1f}s), mem_k.shape={mem_k.shape}", flush=True)
    print_hbm("post_embed_fn ")

    # Un-shard to REPLICATED on every chip — matches generation_embed.py:251-253 exactly
    mem_k_rep = jax.device_put(
        np.array(jax.experimental.multihost_utils.process_allgather(mem_k, tiled=True)),
        jax.sharding.PartitionSpec()
    )
    mem_v_rep = jax.device_put(
        np.array(jax.experimental.multihost_utils.process_allgather(mem_v, tiled=True)),
        jax.sharding.PartitionSpec()
    )
    mem_mask_rep = jax.device_put(
        np.array(jax.experimental.multihost_utils.process_allgather(mem_mask_flat, tiled=True)),
        jax.sharding.PartitionSpec()
    )
    mem_k_rep.block_until_ready()
    print(f"[{HOSTNAME}] allgather done, mem_k_rep.shape={mem_k_rep.shape} nbytes={mem_k_rep.nbytes/1e6:.1f}MB", flush=True)
    print_hbm("post_allgather")

    # Inject into params (matches line 260-264)
    params_with_mem = dict(model.weights)
    params_with_mem["main_model.mem_k"]    = mem_k_rep
    params_with_mem["main_model.mem_v"]    = mem_v_rep
    params_with_mem["main_model.mem_mask"] = mem_mask_rep

    # Prompt input at eval seq_len=2048
    prompt = jax.device_put(
        jax.random.randint(k_prompt, (E_B, E_seq_len), 1, vocab, dtype=jnp.int32),
        jax.sharding.PartitionSpec("data", None)
    )
    pmask = jax.device_put(
        jnp.ones((E_B, E_seq_len), dtype=jnp.bool_),
        jax.sharding.PartitionSpec("data", None)
    )
    if cfg.model.get("memory", {}).get("num_spec_tokens", 0) > 0:
        spec_mask = jax.device_put(
            jnp.zeros((E_B, E_seq_len), dtype=jnp.bool_),
            jax.sharding.PartitionSpec("data", None)
        )
        gen_pad_mask = {"batch_mask": pmask, "spec_token_mask": spec_mask}
    else:
        gen_pad_mask = pmask

    print_hbm("pre_generate  ")
    t_g = time.time()
    gen_tokens = _generate_tokens(
        model.forward,
        model.init_kv,
        params_with_mem,
        prompt,
        E_max_new,
        pad_mask=gen_pad_mask,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
    )
    gen_tokens.block_until_ready()
    print(f"[{HOSTNAME}] _generate_tokens done ({time.time()-t_g:.1f}s), gen_tokens.shape={gen_tokens.shape}", flush=True)
    print_hbm("post_generate ")

except Exception as e:
    import traceback
    print(f"[{HOSTNAME}] EVAL PROBE FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
    print(f"[{HOSTNAME}] traceback: {traceback.format_exc()[:800]}", flush=True)
    print_hbm("after_eval_fail")


# ────────────────────────────────────────────────────────────────────────────────────────────
# SAVE probe: exercise the new sharded-save path and verify HBM peak stays close to
# post-train baseline (not the ~35 GB spike the old unshard-then-write path produced).
# Writes to a scratch GCS path so we exercise the real orbax code path (Zarr-to-GCS).
# ────────────────────────────────────────────────────────────────────────────────────────────
print(f"[{HOSTNAME}] === SAVE probe (orbax native sharded save) ===", flush=True)

try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions
    from utils import save_checkpoint

    # Scratch path — ALL hosts must write to the SAME directory (orbax multi-host
    # requires it: primary_host writes metadata that secondaries block on). Use the
    # launcher-broadcast RUN_START_TIME as the unique-per-launch component so both
    # hosts pick the same path.
    run_id = os.environ.get("RUN_START_TIME", "no_run_start_time")
    scratch_dir = f"gs://memory-layers-training/_probe/opt_state_shape/{run_id}"

    save_options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        multiprocessing_options=MultiprocessingOptions(primary_host=0),
    )
    save_mgr = ocp.CheckpointManager(scratch_dir, ocp.StandardCheckpointer(), options=save_options)

    print_hbm("pre_save      ")
    t_s = time.time()
    save_checkpoint(save_mgr, model, opt_state, step=0, stage_idx=0)
    print(f"[{HOSTNAME}] save_checkpoint done ({time.time()-t_s:.1f}s), scratch={scratch_dir}", flush=True)
    print_hbm("post_save     ")

except Exception as e:
    import traceback
    print(f"[{HOSTNAME}] SAVE PROBE FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
    print(f"[{HOSTNAME}] traceback: {traceback.format_exc()[:1200]}", flush=True)
    print_hbm("after_save_fail")

print(f"[{HOSTNAME}] DONE", flush=True)
