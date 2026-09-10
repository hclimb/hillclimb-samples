"""Download just enough HF parquet shards for a run, balanced across sources.

WHY THIS EXISTS
---------------
Live-HF streaming of a multi-source QA mix does not work: `num_workers × sources` shard
resolutions blow HF's per-account 1000-req/5-min quota during pipeline build, and qa.py's
crash handler responds to the 429 by rebuilding the pipeline, which re-blows it — a livelock
that never reaches step 1. See wiki/data/hf-rate-limits.md. The fix is to read local parquet
(HF_HUB_OFFLINE=1 + GROUND_HF_PARQUET), which needs the shards on disk first.

`scripts/misc/precache_hf.sh` did that, but bluntly: `GROUND_DATA_FRAC=0.5` takes the first
half of *every* repo's shards. Two problems that this script fixes:

1. **It over-downloads and still under-covers.** The full mix is ~75 GB against a ~97 GB boot
   disk, so a fraction is forced — yet a 100k-step run at batch_size=16 only ever consumes
   1.6M examples. Sizing by ROWS NEEDED instead of a blanket fraction gets a full epoch in
   ~12-15 GB.
2. **It silently unbalances the mix.** `interleave_datasets(..., stopping_strategy=
   "all_exhausted")` is uniform round-robin over sources (no `probabilities` unless a source
   sets `weight`), so every source must supply total_rows/N. Repos differ ~6x in rows-per-shard,
   so an equal shard *fraction* yields wildly unequal rows — and `all_exhausted` RESTARTS any
   source that runs dry, silently repeating it for the rest of the run.

Also selects shards at random (seeded) rather than taking the head: with an epoch needing only
~12% of science-qa, `files[:N]` biases toward whatever the shard order correlates with.

USAGE
-----
    # See the plan without downloading anything (no writes, minimal Hub calls):
    uv run python data/download_hf_data.py --dataset qa_hard_neg_think_sft4b --plan

    # Do it:
    uv run python data/download_hf_data.py --dataset qa_hard_neg_think_sft4b

    # Explicit repos instead of a Hydra dataset config:
    uv run python data/download_hf_data.py --repos vm2825/science-qa-hard-neg-think=1.0 ... \
        --total-rows 1600000

Then launch with `HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=<out>` (BOTH are required — qa.py's offline
branch is gated on HF_HUB_OFFLINE == "1", so GROUND_HF_PARQUET alone silently streams live).

Idempotent/resumable: hf_hub_download skips files already on disk, and the selection is a
deterministic function of (repo, seed), so re-running tops up rather than re-picking.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Planning (pure — no network, no disk; unit-tested in tests/test_download_hf_data.py) ──


@dataclass
class SourcePlan:
    """What we intend to fetch for one repo."""
    repo: str
    weight: float = 1.0
    total_rows: int = 0            # rows in the whole repo (measured)
    total_shards: int = 0          # parquet shards in the whole repo (measured)
    total_bytes: int = 0           # bytes of all shards (measured)
    target_rows: int = 0           # rows we need (computed)
    shards_needed: int = 0         # shards to fetch (computed)
    selected: List[str] = field(default_factory=list)
    est_bytes: int = 0

    @property
    def rows_per_shard(self) -> float:
        return self.total_rows / self.total_shards if self.total_shards else 0.0

    @property
    def bytes_per_shard(self) -> float:
        return self.total_bytes / self.total_shards if self.total_shards else 0.0


def target_rows_for(total_rows: int, weight: float, weights_sum: float) -> int:
    """Rows one source must supply.

    Mirrors data/qa.py's interleave EXACTLY: with no per-source `weight` the interleave is
    uniform round-robin (equal share); if ANY source sets a weight, qa.py switches to
    `probabilities=[w/sum(w)]` and the shares become proportional. Keep these in step — if this
    disagrees with qa.py, a source runs dry mid-run and `all_exhausted` silently repeats it.
    """
    if weights_sum <= 0:
        raise ValueError("weights must sum to > 0")
    return int(math.ceil(total_rows * (weight / weights_sum)))


def shards_for_rows(target_rows: int, rows_per_shard: float, headroom: float) -> int:
    """Shards needed to cover target_rows, with headroom, rounded up.

    Headroom matters because rows_per_shard is an AVERAGE and real shards vary (science-qa's
    shard 0 holds 55,355 rows against a ~44.9k mean). Landing short doesn't error — the source
    just restarts and repeats — so we over-provision slightly by default.
    """
    if rows_per_shard <= 0:
        return 0
    return int(math.ceil((target_rows * headroom) / rows_per_shard))


def select_shards(files: List[str], n: int, mode: str, seed: int, repo: str) -> List[str]:
    """Pick n shards. `random` (default) is a seeded sample; `head` is files[:n].

    `head` is what precache_hf.sh does and is biased: an epoch needs ~12% of science-qa, so the
    head is a 12% slice of whatever the shard order correlates with (topic/source/date). The
    seed is mixed with the repo name so different repos don't select correlated positions.
    Deterministic in (repo, seed, n) => re-running resumes instead of re-picking.
    """
    n = max(0, min(n, len(files)))
    if mode == "head":
        return list(files[:n])
    if mode != "random":
        raise ValueError(f"unknown selection mode: {mode!r} (want 'random' or 'head')")
    rng = random.Random(f"{seed}:{repo}")
    return sorted(rng.sample(list(files), n))


def build_plan(sources: List[SourcePlan], total_rows: int, headroom: float,
               mode: str, seed: int, files_by_repo: Dict[str, List[str]]) -> List[SourcePlan]:
    """Fill in target_rows / shards_needed / selected / est_bytes for every source."""
    wsum = sum(s.weight for s in sources)
    for s in sources:
        s.target_rows = target_rows_for(total_rows, s.weight, wsum)
        s.shards_needed = shards_for_rows(s.target_rows, s.rows_per_shard, headroom)
        files = files_by_repo.get(s.repo, [])
        s.selected = select_shards(files, s.shards_needed, mode, seed, s.repo)
        s.est_bytes = int(len(s.selected) * s.bytes_per_shard)
    return sources


def format_plan(sources: List[SourcePlan], total_rows: int, headroom: float) -> str:
    out = [
        f"{'repo':52s} {'need':>9s} {'shards':>12s} {'est GB':>8s} {'of repo':>8s}",
        "-" * 95,
    ]
    tot_b = tot_r = 0
    for s in sources:
        got = int(len(s.selected) * s.rows_per_shard)
        tot_b += s.est_bytes
        tot_r += got
        pct = (100.0 * len(s.selected) / s.total_shards) if s.total_shards else 0.0
        out.append(f"{s.repo[:52]:52s} {s.target_rows:9,d} "
                   f"{len(s.selected):5d}/{s.total_shards:<6d} {s.est_bytes/1e9:8.2f} {pct:7.1f}%")
    out.append("-" * 95)
    out.append(f"{'TOTAL':52s} {total_rows:9,d} {'':12s} {tot_b/1e9:8.2f}")
    out.append(f"(headroom {headroom:.2f}x -> ~{tot_r:,d} rows on disk)")
    return "\n".join(out)


# ── Hub I/O ──────────────────────────────────────────────────────────────────────────────


def _hf_retry(what: str, fn, attempts: int = 10):
    """Backoff on 429. The whole point of this script is to stop hammering the Hub, but even
    listing/downloading can trip the quota if something else is running."""
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if "429" not in str(e) or a == attempts - 1:
                raise
            d = min(30 * (a + 1), 240) + random.uniform(0, 10)
            print(f"  429 during {what}; retry in {d:.0f}s ({a+1}/{attempts})", flush=True)
            time.sleep(d)


def list_parquet(api, repo: str, token: Optional[str]) -> List[str]:
    files = _hf_retry(f"list {repo}",
                      lambda: api.list_repo_files(repo, repo_type="dataset", token=token))
    return sorted(f for f in files if f.endswith(".parquet"))


def repo_totals(api, repo: str, token: Optional[str], files: List[str]) -> tuple[int, int]:
    """(total_rows, total_bytes) for a repo.

    Bytes come from the repo tree. Rows are the awkward part: the datasets-server /size endpoint
    is one cheap call but 404s on PRIVATE repos, so we fall back to reading ONE shard's parquet
    footer and extrapolating. Either way the count is an estimate; actual rows are verified from
    local footers after download and topped up (see download_source).
    """
    total_bytes = 0
    try:
        info = _hf_retry(f"tree {repo}", lambda: api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, expand=True, token=token))
        for e in info:
            path = getattr(e, "path", "")
            if path.endswith(".parquet"):
                size = getattr(e, "size", None)
                lfs = getattr(e, "lfs", None)
                total_bytes += (getattr(lfs, "size", None) or size or 0)
    except Exception as e:  # noqa: BLE001
        print(f"  tree failed for {repo} ({str(e)[:60]}); bytes unknown", flush=True)

    rows = _rows_via_datasets_server(repo, token)
    if rows is None:
        rows = _rows_via_first_shard(repo, token, files)
    return rows or 0, total_bytes


def _rows_via_datasets_server(repo: str, token: Optional[str]) -> Optional[int]:
    import json
    import urllib.request
    url = f"https://datasets-server.huggingface.co/size?dataset={repo}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        return ((d.get("size") or {}).get("dataset") or {}).get("num_rows")
    except Exception:  # noqa: BLE001
        return None  # private repo / not converted -> caller falls back


def _rows_via_first_shard(repo: str, token: Optional[str], files: List[str]) -> Optional[int]:
    """Extrapolate: rows(shard 0) x n_shards. Downloads one shard."""
    if not files:
        return None
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = _hf_retry(f"probe {repo}", lambda: hf_hub_download(
            repo, files[0], repo_type="dataset", token=token,
            local_dir=os.path.expanduser(f"~/.cache/hf_probe/{repo.replace('/', '__')}")))
        n = pq.ParquetFile(p).metadata.num_rows
        print(f"  {repo}: probed shard 0 = {n:,} rows -> est {n * len(files):,} total", flush=True)
        return n * len(files)
    except Exception as e:  # noqa: BLE001
        print(f"  row probe failed for {repo}: {str(e)[:70]}", flush=True)
        return None


def local_rows(paths: List[str]) -> int:
    """Exact rows on disk, from parquet footers (no data read)."""
    import pyarrow.parquet as pq
    n = 0
    for p in paths:
        try:
            n += pq.ParquetFile(p).metadata.num_rows
        except Exception:  # noqa: BLE001
            pass
    return n


def download_source(s: SourcePlan, out_root: str, token: Optional[str], api,
                    files: List[str], mode: str, seed: int, verify: bool) -> int:
    """Download s.selected; then VERIFY exact rows and top up if the estimate fell short.

    The top-up is why this is worth doing: rows_per_shard is an average, so a source can land
    under target and then silently repeat all run under `all_exhausted`.
    """
    from huggingface_hub import hf_hub_download
    dest = os.path.join(out_root, s.repo.replace("/", "__"))
    got: List[str] = []
    for f in s.selected:
        p = _hf_retry(f"get {s.repo}:{f}", lambda f=f: hf_hub_download(
            s.repo, f, repo_type="dataset", token=token, local_dir=dest))
        got.append(p)
    if not verify:
        return len(got)

    have = local_rows(got)
    extra = [f for f in files if f not in set(s.selected)]
    rng = random.Random(f"{seed}:{s.repo}:topup")
    rng.shuffle(extra)
    while have < s.target_rows and extra:
        f = extra.pop()
        p = _hf_retry(f"top-up {s.repo}:{f}", lambda f=f: hf_hub_download(
            s.repo, f, repo_type="dataset", token=token, local_dir=dest))
        got.append(p)
        have = local_rows(got)
        print(f"  {s.repo}: topping up -> {have:,}/{s.target_rows:,} rows", flush=True)
    if have < s.target_rows:
        print(f"  WARNING {s.repo}: only {have:,} rows available (< {s.target_rows:,} needed). "
              f"interleave stopping_strategy='all_exhausted' will REPEAT this source.", flush=True)
    else:
        print(f"  {s.repo}: {have:,} rows on disk (target {s.target_rows:,})", flush=True)
    return len(got)


# ── Source resolution ────────────────────────────────────────────────────────────────────


def sources_from_dataset_cfg(name: str) -> List[SourcePlan]:
    """Read hf_name + weight out of configs/dataset/<name>.yaml via Hydra.

    Composing the real config (rather than parsing yaml) is the point: it resolves the
    `defaults: /dataset/sources@sources.X` tree and ${oc.env:HF_USERNAME} exactly as training
    will, so we cache precisely the repos the run will read.
    """
    from hydra import compose, initialize_config_dir
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(config_dir=os.path.join(root, "configs"), version_base="1.2"):
        # `eval_set@trainer.evals=none` is REQUIRED, not cosmetic: configs/trainer/standard.yaml
        # defaults evals to `eval_set/standard`, which does not exist on this branch, so a bare
        # compose of `train` dies with MissingConfigException before we ever see cfg.dataset.
        # (staged_ground.yaml carries the same override for the same reason.) We only want
        # cfg.dataset, so pin the trainer's eval set to the empty one.
        cfg = compose(config_name="train",
                      overrides=[f"dataset={name}", "eval_set@trainer.evals=none"])
    srcs = cfg.dataset.get("sources") or {}
    if not srcs:
        raise SystemExit(f"dataset '{name}' has no `sources:` (single-source configs use --repos)")
    return [SourcePlan(repo=s["hf_name"], weight=float(s.get("weight", 1.0)))
            for s in srcs.values()]


def sources_from_args(specs: List[str]) -> List[SourcePlan]:
    out = []
    for spec in specs:
        repo, _, w = spec.partition("=")
        out.append(SourcePlan(repo=repo, weight=float(w) if w else 1.0))
    return out


# ── main ─────────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download an epoch-sized, source-balanced subset of HF parquet shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Hydra dataset config name (reads its sources + weights)")
    src.add_argument("--repos", nargs="+", metavar="REPO[=WEIGHT]", help="explicit repo list")

    ap.add_argument("--steps", type=int, default=100_000, help="training steps to cover")
    ap.add_argument("--batch-size", type=int, default=16, help="per-step examples")
    ap.add_argument("--total-rows", type=int, default=None,
                    help="rows to cover (overrides --steps x --batch-size)")
    ap.add_argument("--headroom", type=float, default=1.25,
                    help="over-provision factor; shards vary in rows and a short source silently repeats")
    ap.add_argument("--select", choices=["random", "head"], default="random",
                    help="'random' = seeded sample (unbiased); 'head' = files[:n] (what precache_hf.sh does)")
    ap.add_argument("--seed", type=int, default=42, help="shard-selection seed (deterministic => resumable)")
    ap.add_argument("--out", default=os.environ.get("GROUND_HF_PARQUET", "~/hf_parquet"),
                    help="output root; pass as GROUND_HF_PARQUET at train time")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit; downloads nothing")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="abort if the plan exceeds this many GB")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="abort if the plan would leave less than this much free disk")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-download row verification + top-up")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    out_root = os.path.expanduser(args.out)
    total_rows = args.total_rows or (args.steps * args.batch_size)

    sources = (sources_from_dataset_cfg(args.dataset) if args.dataset
               else sources_from_args(args.repos))

    from huggingface_hub import HfApi
    api = HfApi()

    print(f"Sizing for {total_rows:,} rows "
          f"({args.steps:,} steps x {args.batch_size})" if not args.total_rows
          else f"Sizing for {total_rows:,} rows", flush=True)
    print(f"Sources: {len(sources)}  select={args.select} seed={args.seed} "
          f"headroom={args.headroom}x  out={out_root}\n", flush=True)

    files_by_repo: Dict[str, List[str]] = {}
    for s in sources:
        files_by_repo[s.repo] = list_parquet(api, s.repo, token)
        s.total_shards = len(files_by_repo[s.repo])
        s.total_rows, s.total_bytes = repo_totals(api, s.repo, token, files_by_repo[s.repo])
        if not s.total_rows:
            raise SystemExit(f"could not determine row count for {s.repo}")

    build_plan(sources, total_rows, args.headroom, args.select, args.seed, files_by_repo)
    print()
    print(format_plan(sources, total_rows, args.headroom))
    print()

    plan_gb = sum(s.est_bytes for s in sources) / 1e9
    if args.max_gb and plan_gb > args.max_gb:
        raise SystemExit(f"plan is {plan_gb:.1f} GB > --max-gb {args.max_gb}")
    free_gb = shutil.disk_usage(os.path.dirname(out_root) or "/").free / 1e9
    print(f"disk: {free_gb:.1f} GB free; plan needs ~{plan_gb:.1f} GB", flush=True)
    if free_gb - plan_gb < args.min_free_gb:
        raise SystemExit(
            f"would leave {free_gb - plan_gb:.1f} GB free (< --min-free-gb {args.min_free_gb}). "
            f"Free space or lower --steps/--headroom.")

    if args.plan:
        print("\n--plan: nothing downloaded.")
        return 0

    os.makedirs(out_root, exist_ok=True)
    for s in sources:
        print(f"\n=== {s.repo}: {len(s.selected)} shards ===", flush=True)
        download_source(s, out_root, token, api, files_by_repo[s.repo],
                        args.select, args.seed, verify=not args.no_verify)

    print(f"\nDONE -> {out_root}")
    print(f"Launch with:  HF_HUB_OFFLINE=1 GROUND_HF_PARQUET={out_root}")
    print("(BOTH are required: qa.py's offline branch is gated on HF_HUB_OFFLINE==1.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
