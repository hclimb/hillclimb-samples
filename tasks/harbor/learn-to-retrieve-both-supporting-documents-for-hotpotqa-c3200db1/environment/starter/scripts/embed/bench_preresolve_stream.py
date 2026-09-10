"""Test the "pre-resolve once → stream" data approach against the real 16-worker QADataset pipeline.

Hypothesis: the live-HF 429 came from per-worker METADATA resolution (the tree/list API), not shard
byte reads. If we resolve each source's parquet shard list ONCE (main process) and stream via explicit
`hf://` data_files, the workers never call the tree API → no 429 — and fsspec range-reads keep disk
bounded while covering the full dataset over time.

This runs the REAL QADataset (real tokenizer, real filter/transform, real grain mp_prefetch with 16
workers) LIVE (no HF_HUB_OFFLINE). Two arms:
  preresolve : monkeypatch data.qa._load_dataset_with_backoff -> resolve shard URLs once in main,
               load_dataset("parquet", data_files=<hf:// urls>, streaming=True).  [hypothesis]
  name       : the original path, load_dataset(name, streaming=True) per the repo config.  [control]

Measures per arm: startup (first batch), steady-state batch/s + per-batch tail, and HF-cache/disk
growth (streaming should stay ~flat). Compare to the ~2.05 batch/s compute rate. Watch the log for
"429". Run each as its own process (the .sh bounds the control with `timeout` in case it 429-stalls).
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from bench_data_throughput import _compose_cfg, _build_tokenizer, _TokShim


def _disk_kb():
    """Total size (KB) of the HF cache — grows if streaming downloads full shards; ~flat if range-read."""
    total = 0
    for p in ("~/.cache/huggingface", "~/.cache/huggingface/datasets", "~/.cache/huggingface/hub"):
        d = os.path.expanduser(p)
        if os.path.isdir(d):
            try:
                total = max(total, int(subprocess.run(["du", "-sk", d], capture_output=True, text=True).stdout.split()[0]))
            except Exception:
                pass
    return total


def _df_free_kb():
    try:
        return int(subprocess.run(["df", "-k", "--output=avail", "/"], capture_output=True, text=True).stdout.split()[-1])
    except Exception:
        return -1


def _install_preresolve(token):
    """Replace data.qa._load_dataset_with_backoff with a pre-resolving streamer (resolve URLs once in
    THIS process; QADataset.__init__ calls it in main, so workers get the already-built dataset)."""
    import data.qa as q
    from datasets import load_dataset
    from huggingface_hub import HfApi
    cache = {}

    def _pre(name, hf_config, split, token_arg):
        if name not in cache:
            files = None
            for a in range(8):
                try:
                    files = HfApi().list_repo_files(name, repo_type="dataset", token=token); break
                except Exception as e:  # noqa: BLE001
                    if "429" in str(e) and a < 7:
                        time.sleep(min(20 * (a + 1), 120)); continue
                    raise
            urls = [f"hf://datasets/{name}/{f}" for f in sorted(files)
                    if f.endswith(".parquet") and "valid" not in f.lower() and "test" not in f.lower()]
            cache[name] = urls
            print(f"[preresolve] {name}: resolved {len(urls)} train parquet shards (once, in main)", flush=True)
        return load_dataset("parquet", data_files=cache[name], split="train", streaming=True)

    q._load_dataset_with_backoff = _pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["preresolve", "name"], required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    args = ap.parse_args()

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"[preresolve-bench] arm={args.arm}  HF_HUB_OFFLINE={offline} (want LIVE=False)  "
          f"num_workers=16", flush=True)
    cfg = _compose_cfg()
    if args.arm == "preresolve":
        _install_preresolve(os.environ.get("HF_TOKEN"))

    tok = _build_tokenizer(str(cfg.model.main_model.model_id))
    from data import get_dataset
    dataset = get_dataset(cfg.dataset, _TokShim(tok))

    disk0, free0 = _disk_kb(), _df_free_kb()
    gen = dataset.generator()
    print(f"[preresolve-bench] pulling first batch (resolve + 100k shuffle-buffer fill)...", flush=True)
    t0 = time.perf_counter()
    tokens, masks = next(gen)
    startup = time.perf_counter() - t0
    disk1 = _disk_kb()
    print(f"[preresolve-bench] startup {startup:.2f}s  batch shapes={{k: tuple(v.shape) for k,v in tokens.items()}}  "
          f"hf-cache +{(disk1-disk0)/1024:.0f}MB during startup", flush=True)

    times = []
    deadline = time.perf_counter() + args.max_seconds
    for i in range(args.batches):
        t = time.perf_counter()
        tokens, masks = next(gen)
        dt = time.perf_counter() - t
        times.append(dt)
        if (i + 1) % 20 == 0 or dt > 1.0:
            print(f"  batch {i+1:>3}: {dt*1e3:8.1f} ms{'  <-- STALL' if dt > 1.0 else ''}  "
                  f"(hf-cache {(_disk_kb()-disk0)/1024:.0f}MB)", flush=True)
        if time.perf_counter() > deadline:
            print(f"[preresolve-bench] hit --max-seconds after {i+1} batches", flush=True)
            break

    diskN, freeN = _disk_kb(), _df_free_kb()
    a = np.array(times)
    thr = len(a) / a.sum() if a.sum() > 0 else float("nan")
    print("\n\n################ PRE-RESOLVE STREAM ({}) ################".format(args.arm), flush=True)
    print(f"  startup (first batch)   : {startup:8.2f} s", flush=True)
    print(f"  batches measured        : {len(a)}", flush=True)
    print(f"  throughput              : {thr:8.3f} batch/s  (compute needs 2.053)  {'KEEPS UP' if thr>=2.053 else 'below compute'}", flush=True)
    print(f"  per-batch ms med/p95/max: {np.median(a)*1e3:.1f} / {np.percentile(a,95)*1e3:.1f} / {a.max()*1e3:.1f}", flush=True)
    print(f"  stalls #>1s             : {int((a>1.0).sum())}", flush=True)
    print(f"  HF cache growth         : {(diskN-disk0)/1024:8.0f} MB   (streaming range-reads => want ~flat)", flush=True)
    print(f"  disk free delta         : {(freeN-free0)/1024:8.0f} MB   (negative = disk filled up)", flush=True)
    print("############################################################\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("!!!! bench_preresolve_stream FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(0)
