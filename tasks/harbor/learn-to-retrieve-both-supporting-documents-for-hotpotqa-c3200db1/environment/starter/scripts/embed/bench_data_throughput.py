"""Measure the REAL dataloader's throughput — Bottleneck #1 (data feeding the TPU).

Builds the ACTUAL QADataset grain pipeline for the qa_hard_neg_think_sft4b run (real tokenizer +
real sources), NO model / NO TPU, and consumes N batches timing each next(). Pulling as fast as
possible drains grain's prefetch buffer, so the steady-state per-batch time is the pipeline's
PRODUCTION CEILING — compare it to the compute rate (2.053 batch/s @ 487 ms device-bound /
1.935 @ 517 ms current synced loop). If the ceiling > 2.05/s the loader keeps up in real training;
if it stalls, data is the bottleneck.

HF mode is whatever HF_HUB_OFFLINE / GROUND_HF_PARQUET are in the env (the .sh sets them per arm),
printed at startup so the log is self-describing. Two costs this surfaces:

  startup      time to the FIRST batch — building the pipeline + filling the
               shuffle(buffer_size=100000) buffer (qa.py:397/449). Under live-HF that is ~100k
               examples over the network (429-prone, qa.py:326 _hf_retry); offline-parquet reads
               local disk.
  steady-state per-batch next() time after startup. Live-HF is gated by HF's request rate / 429
               backoff; the tail (p95 / max / #>1s / #>5s) is the stall signal.

Usage (on the box, from repo root; HF mode via env):
                          uv run python scripts/embed/bench_data_throughput.py   # live-HF (default env)
  HF_HUB_OFFLINE=1        uv run python scripts/embed/bench_data_throughput.py   # offline-parquet
                          uv run python scripts/embed/bench_data_throughput.py --check  # no data pull
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hydra import compose, initialize_config_dir

_OVERRIDES = [
    "model=qwen3_mem_embed",
    "model.main_model.model_id=Qwen/Qwen3-4B",
    "model.memory.mem_top_k=64",
    "dataset=qa_hard_neg_think_sft4b",
    "trainer.use_wandb=false",
    "eval_set@trainer.evals=none",
    "trainer.eval_interval=0",
]

# Compute-side reference rates from the profiler / loop-sync bench (v6e-8, this config).
_CEILING_MS = 486.99   # device-bound step
_LOOP_MS = 516.78      # current per-step-synced loop


def _compose_cfg():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.2"):
        return compose(config_name="train", overrides=_OVERRIDES)


def _build_tokenizer(model_id="Qwen/Qwen3-4B", hf_ckpt_dir="~/weights/huggingface"):
    """Exact replica of qwen3.load()'s tokenizer construction (models/qwen3.py:237-240) — CPU only,
    no weights. Fetches just the tokenizer files if the checkpoint dir is absent."""
    from transformers import PreTrainedTokenizerFast, AddedToken

    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id
    if not (model_ckpt_dir / "tokenizer.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir,
                          token=os.environ.get("HF_TOKEN"),
                          allow_patterns=["tokenizer*", "*.json", "vocab*", "merges*", "special_tokens*"])
    tok_cfg = json.loads((model_ckpt_dir / "tokenizer_config.json").read_text())
    tok_cfg["added_tokens_decoder"] = {int(k): AddedToken(**v) for k, v in tok_cfg["added_tokens_decoder"].items()}
    return PreTrainedTokenizerFast(tokenizer_file=str(model_ckpt_dir / "tokenizer.json"), **tok_cfg)


class _TokShim:
    """get_dataset only reads model.tokenizer (data/__init__.py) — a shim avoids loading the 4B."""
    def __init__(self, tok):
        self.tokenizer = tok


def _mode_str():
    off = os.environ.get("HF_HUB_OFFLINE") == "1"
    if off:
        return f"OFFLINE-PARQUET (GROUND_HF_PARQUET={os.environ.get('GROUND_HF_PARQUET', '~/hf_parquet')})"
    return "LIVE-HF STREAMING (load_dataset streaming=True, 429 backoff)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=240.0, help="wall-clock cap incl. startup")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print(f"[data-bench] HF mode: {_mode_str()}", flush=True)
    cfg = _compose_cfg()
    print(f"[data-bench] dataset={cfg.dataset.name} batch_size={cfg.dataset.batch_size} "
          f"num_workers={cfg.dataset.num_workers} seq_len={cfg.dataset.seq_len} "
          f"num_chunks_per_doc={cfg.dataset.num_chunks_per_doc} doc_chunk_seq_len={cfg.dataset.doc_chunk_seq_len}",
          flush=True)

    tok = _build_tokenizer(str(cfg.model.main_model.model_id))
    print(f"[data-bench] tokenizer built ({type(tok).__name__}, vocab={tok.vocab_size})", flush=True)

    from data import get_dataset
    dataset = get_dataset(cfg.dataset, _TokShim(tok))

    if args.check:
        print("[CHECK] compose + tokenizer + dataset construction OK (no data pulled)", flush=True)
        return

    gen = dataset.generator()

    # Startup: build pipeline + fill the 100k shuffle buffer + first fetch.
    print(f"[data-bench] pulling first batch (startup: pipeline build + 100k shuffle-buffer fill)...", flush=True)
    t0 = time.perf_counter()
    tokens, masks = next(gen)
    startup = time.perf_counter() - t0
    bshape = {k: tuple(v.shape) for k, v in tokens.items()}
    print(f"[data-bench] startup (time to first batch): {startup:.2f} s   batch shapes={bshape}", flush=True)

    # Steady-state: pull as fast as possible (drains buffer -> measures production ceiling).
    times = []
    deadline = time.perf_counter() + args.max_seconds
    for i in range(args.batches):
        t = time.perf_counter()
        tokens, masks = next(gen)
        dt = time.perf_counter() - t
        times.append(dt)
        if (i + 1) % 16 == 0 or dt > 1.0:
            print(f"  batch {i+1:>3}: {dt*1e3:8.1f} ms{'   <-- STALL' if dt > 1.0 else ''}", flush=True)
        if time.perf_counter() > deadline:
            print(f"[data-bench] hit --max-seconds cap after {i+1} batches", flush=True)
            break

    a = np.array(times)
    total = a.sum()
    thr = len(a) / total if total > 0 else float("nan")
    print("\n\n################ DATA THROUGHPUT ################", flush=True)
    print(f"  HF mode                : {_mode_str()}", flush=True)
    print(f"  startup (first batch)  : {startup:8.2f} s", flush=True)
    print(f"  batches measured       : {len(a)}", flush=True)
    print(f"  production ceiling      : {thr:8.3f} batch/s   (mean {a.mean()*1e3:.1f} ms/batch)", flush=True)
    print(f"  per-batch ms  median/p90/p95/p99/max : "
          f"{np.median(a)*1e3:.1f} / {np.percentile(a,90)*1e3:.1f} / {np.percentile(a,95)*1e3:.1f} / "
          f"{np.percentile(a,99)*1e3:.1f} / {a.max()*1e3:.1f}", flush=True)
    print(f"  stalls  #>1s / #>5s    : {int((a>1.0).sum())} / {int((a>5.0).sum())}", flush=True)
    print(f"  compute rate to feed   : {1000.0/_CEILING_MS:.3f} batch/s (@{_CEILING_MS:.0f}ms ceiling) / "
          f"{1000.0/_LOOP_MS:.3f} batch/s (@{_LOOP_MS:.0f}ms loop)", flush=True)
    verdict = "KEEPS UP" if thr >= 1000.0/_CEILING_MS else "BOTTLENECK (slower than compute)"
    print(f"  verdict (steady-state) : {verdict}", flush=True)
    print("################################################\n", flush=True)


if __name__ == "__main__":
    main()
