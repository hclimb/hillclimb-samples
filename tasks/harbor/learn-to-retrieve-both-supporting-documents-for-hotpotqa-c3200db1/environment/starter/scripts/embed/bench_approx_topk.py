"""Standalone A/B microbenchmark: does MEM_APPROX_TOPK help the qa_hard_neg_think_sft4b run?

The train-time memory top-k goes through models/memory_utils.bank_top_k, which switches
between exact jax.lax.top_k and jax.lax.approx_max_k on the MEM_APPROX_TOPK env var. For the
qa_hard_neg_think_sft4b config (qwen3_mem_embed, mem_top_k=64, tp_devices=1) the bank is built
from the batch: M = batch(16) x num_chunks_per_doc(16) x doc_chunk_seq_len(256) = 65536 keys,
scored per (B, mem_num_heads=4, T) — i.e. a top-64 over 65536, the sorting-bound regime where
approx_max_k is claimed ~6x faster.

Two modes (run each as its own process so the import-time env toggle is clean):

  time    Times the REAL train step (Trainer._train_step, jitted) at the true training shapes,
          reading MEM_APPROX_TOPK / MEM_APPROX_RECALL from env. Prints a [BENCH] line. Run once
          per setting (exact vs approx) and diff the medians -> the actual step-time benefit.

  recall  Env-independent. Pulls the REAL per-token score matrix (aux 'mem_scores', [B,T,N,M])
          from one forward pass, then measures how well approx_max_k recovers exact top-64
          across recall targets -> the selection fidelity the training gradient would see.

Mirrors train.py wiring (get_model -> setup_optimizer -> get_dataset) and reuses stage-0 of the
staged schedule (mem+conv trainable, ce_weight=0, doc_access_loss weight 0.1) so the step is
byte-identical to real training except for the top-k op. Does NOT modify the trainer.

Usage (on the box, from repo root):
  MEM_APPROX_TOPK=0 uv run python scripts/embed/bench_approx_topk.py --mode time
  MEM_APPROX_TOPK=1 uv run python scripts/embed/bench_approx_topk.py --mode time
  uv run python scripts/embed/bench_approx_topk.py --mode recall

--overrides appends extra Hydra overrides (last-wins on duplicate keys), so the same harness
times other configs of the same model family, e.g. the grounding 4-layer shape
(scripts/embed/bench_ground4layer_approx.sh).
"""
import argparse
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")
# This script lives in scripts/embed/, so Python puts that dir (not the repo root) on
# sys.path[0]. train.py works only because it sits at the root; add the root here so the
# repo modules (utils, models, data, trainer) import the same way they do for train.py.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir

# Overrides shared by both modes: exactly the qa_hard_neg_think_sft4b run's model/data,
# minus wandb/eval/checkpoint machinery the microbench doesn't need.
_OVERRIDES = [
    "model=qwen3_mem_embed",
    "model.main_model.model_id=Qwen/Qwen3-4B",
    "model.memory.mem_top_k=64",
    "dataset=qa_hard_neg_think_sft4b",
    "trainer.use_wandb=false",
    # Select the empty eval set via the defaults LIST (not a value override): trainer=staged
    # pulls in standard.yaml, whose `- /eval_set@evals: standard` default is resolved before any
    # value override, and eval_set/standard doesn't exist. The real run sidesteps this the same
    # way with `eval_set@trainer.evals=qa_hard_neg_think_sft4b`. We don't instantiate the Trainer,
    # so 'none' is fine — it just has to compose.
    "eval_set@trainer.evals=none",
    "trainer.eval_interval=0",
]


def _merged_overrides(extra):
    """_OVERRIDES + extra, deduped by override key (extra wins). Hydra compose rejects the
    same key appearing twice, so replacement has to happen here, not by appending."""
    def key(o):
        return o.split("=", 1)[0].lstrip("+~")
    merged = {key(o): o for o in _OVERRIDES}
    for o in extra:
        merged[key(o)] = o
    return list(merged.values())


def _compose_cfg(extra_overrides=()):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.2"):
        return compose(config_name="train", overrides=_merged_overrides(extra_overrides))


def _build_stage0_loss_cfg(cfg):
    """Replicate Trainer's stage-0 aux config + ce_weight without instantiating the trainer
    (avoids loading eval datasets). Stage 0 = mem+conv, ce_weight=0, doc_access_loss weight 0.1."""
    from utils import freeze_dict

    base = {k: dict(v) for k, v in cfg.trainer.aux_losses.items()}
    stage0 = cfg.trainer.training_stages[0] if cfg.trainer.get("training_stages") else {}
    for loss_name, ov in dict(stage0.get("aux_losses", {})).items():
        if loss_name in base:
            base[loss_name].update(ov)
        else:
            base[loss_name] = dict(ov)
    aux_loss_config = freeze_dict(base)
    ce_weight = jnp.array(float(stage0.get("ce_weight", cfg.trainer.get("ce_weight", 1.0))))
    return aux_loss_config, ce_weight


def _one_batch(cfg, model):
    """A synthetic batch with the EXACT shapes/dtypes the real generator yields for this config
    (data/qa.py::generator, provide_docs=True), passed through the real process_train_pairs.

    Why synthetic: the step-time we're measuring depends only on tensor shapes (M=65536 top-k),
    not token values, and building the real bank hammers the HF API (16 grain workers -> 429
    rate-limit on the uncached datasets). Shapes:
      batch [B, seq_len]         docs [B*num_chunks_per_doc, doc_chunk_seq_len]  (flattened bank)
      masks: batch_mask/loss_mask [B, seq_len], docs_mask [B*M, d_seq], pos_doc_mask [B, M]
    docs_mask all-ones => all 65536 keys valid => full top-k (realistic worst case).
    Caveat for --mode recall: no checkpoint is loaded (get_model random-inits mem_*), so the score
    distribution is that of an UNTRAINED memory layer either way; recall here is a rough proxy and
    the definitive quality check is Tier-2 A/B loss.
    """
    from utils import process_train_pairs

    B = int(cfg.dataset.batch_size)
    seq_len = int(cfg.dataset.seq_len)
    M = int(cfg.dataset.num_chunks_per_doc)
    d_seq = int(cfg.dataset.doc_chunk_seq_len)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    vocab = 1000  # any id < vocab_size is a valid embedding lookup for both Qwen3 trunks
    tokens = {
        "batch": jax.random.randint(k1, (B, seq_len), 1, vocab, dtype=jnp.int32),
        "docs": jax.random.randint(k2, (B * M, d_seq), 1, vocab, dtype=jnp.int32),
    }
    masks = {
        "batch_mask": jnp.ones((B, seq_len), dtype=jnp.int32),
        "docs_mask": jnp.ones((B * M, d_seq), dtype=jnp.float32),
        "loss_mask": jnp.ones((B, seq_len), dtype=jnp.float32),
        "pos_doc_mask": jnp.ones((B, M), dtype=jnp.int32),
        "ce_enable": jnp.ones((B,), dtype=jnp.float32),
    }
    return process_train_pairs(tokens, masks)


def run_time(cfg, model, optimizer, iters, warmup):
    from trainer.trainer import Trainer

    inputs, targets, input_masks, loss_masks, ce_enable = _one_batch(cfg, model)
    aux_loss_config, ce_weight = _build_stage0_loss_cfg(cfg)
    w, opt_state = model.weights, optimizer.init(model.weights)

    def step(w, opt_state):
        return Trainer._train_step(
            model.forward, optimizer, w, opt_state,
            inputs, targets, input_masks, loss_masks, ce_weight, aux_loss_config, ce_enable,
        )

    times = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        w, opt_state, ce_loss, aux_result, loss_nan, grad_nan, grad_norm, _lr = step(w, opt_state)
        float(ce_loss)  # force a device sync -> t captures the full step wall-clock
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        print(f"  step {i:>3} {'(warmup)' if i < warmup else ''} {dt*1e3:8.2f} ms", flush=True)

    ts = sorted(times)
    print(
        f"\n[BENCH] mode=time MEM_APPROX_TOPK={os.environ.get('MEM_APPROX_TOPK', '0')} "
        f"MEM_APPROX_RECALL={os.environ.get('MEM_APPROX_RECALL', '0.95')} "
        f"n={len(ts)} median={statistics.median(ts)*1e3:.2f}ms mean={sum(ts)/len(ts)*1e3:.2f}ms "
        f"p10={ts[int(0.1*(len(ts)-1))]*1e3:.2f}ms p90={ts[int(0.9*(len(ts)-1))]*1e3:.2f}ms",
        flush=True,
    )


def run_recall(cfg, model, targets_list):
    K = int(cfg.model.memory.mem_top_k)
    inputs, _, input_masks, loss_masks, _ = _one_batch(cfg, model)
    pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)

    out = model.forward(inputs, model.weights, pad_mask=pad_mask, collect_aux=True)
    if not out.aux or "mem_scores" not in out.aux or not out.aux["mem_scores"]:
        raise SystemExit(f"no mem_scores in aux; keys={list(out.aux.keys()) if out.aux else None}")
    # aux['mem_scores'] is a list/tuple, and its element is itself a (array,) tuple in the
    # full-lookup path — unwrap nested list/tuple wrappers down to the [B,T,N,M] score array.
    scores = out.aux["mem_scores"]
    while isinstance(scores, (list, tuple)):
        scores = scores[0]
    assert scores.ndim == 4, f"expected [B,T,N,M] scores, got shape {scores.shape}"
    B, T, N, M = scores.shape
    print(f"[RECALL] scores shape [B={B}, T={T}, N={N}, M={M}], K={K}", flush=True)

    exact_val, exact_idx = jax.lax.top_k(scores, K)  # [B,T,N,K]
    exact_idx_np = np.asarray(exact_idx).reshape(-1, K)
    exact_val_np = np.asarray(exact_val).reshape(-1, K)

    # Restrict to query positions that actually drive the loss (loss_mask==1); fall back to all.
    lm = np.asarray(loss_masks).reshape(B, T)  # loss_masks is [B, T]
    valid_bt = lm.reshape(-1) > 0
    valid_rows = np.repeat(valid_bt, N)  # [B*T*N] after the N-expansion of the reshape
    if valid_rows.sum() < 64:
        valid_rows = np.ones_like(valid_rows, dtype=bool)
    rows = np.nonzero(valid_rows)[0]
    rng = np.random.default_rng(0)
    if rows.size > 4000:
        rows = rng.choice(rows, size=4000, replace=False)
    print(f"[RECALL] measuring over {rows.size} valid (position,head) rows", flush=True)

    for rt in targets_list:
        approx_val, approx_idx = jax.lax.approx_max_k(scores, K, recall_target=rt)
        approx_idx_np = np.asarray(approx_idx).reshape(-1, K)
        recalls, mass = [], []
        for r in rows:
            ex, ap = exact_idx_np[r], set(approx_idx_np[r].tolist())
            hit = np.fromiter((i in ap for i in ex), dtype=bool, count=K)
            recalls.append(hit.mean())
            ev = exact_val_np[r]
            denom = ev.sum() if ev.sum() != 0 else 1.0
            mass.append(ev[hit].sum() / denom)
        print(
            f"[RECALL] target={rt:.2f}  mean_recall@{K}={np.mean(recalls):.4f}  "
            f"min_recall={np.min(recalls):.4f}  score_mass_recovered={np.mean(mass):.4f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["time", "recall", "check"], required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--recall-targets", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="extra Hydra overrides merged over the built-in ones (last-wins)")
    args = ap.parse_args()

    if args.overrides:
        print(f"[BENCH] extra_overrides={args.overrides}", flush=True)
    cfg = _compose_cfg(args.overrides)

    # Import the repo modules (validates the sys.path fix — see top of file). Cheap: loads the
    # code, not the 4B weights. 'check' stops here so we can smoke-test config+imports fast.
    from models import get_model
    from utils import setup_optimizer

    if args.mode == "check":
        print(f"[CHECK] compose+imports OK  model={cfg.model.name}  "
              f"mem_top_k={cfg.model.memory.mem_top_k}  dataset={cfg.dataset.name}  "
              f"tp_devices={cfg.trainer.tp_devices}  trainer.steps={cfg.trainer.steps}", flush=True)
        return

    from utils import init_jax_distributed; init_jax_distributed()

    model = get_model(cfg.model, cfg.trainer.tp_devices)
    optimizer, model, _ = setup_optimizer(cfg, model)

    if args.mode == "time":
        run_time(cfg, model, optimizer, args.iters, args.warmup)
    else:
        run_recall(cfg, model, args.recall_targets)


if __name__ == "__main__":
    main()
