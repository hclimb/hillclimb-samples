# Epoch-Sized Offline Data (`data/download_hf_data.py`)

**Download only the rows a run will actually consume, balanced across sources.** A 100k-step run
at `batch_size=16` reads **1.6M examples**; the full hard-neg mix is **75 GB**. Sizing by rows
instead of a blanket shard fraction gets a full, unbiased epoch in **~12-15 GB** — small enough
to sit on the boot disk, which is what makes the offline path (and therefore any live-HF-free
run) possible. See [hf-rate-limits.md](hf-rate-limits.md) for why live streaming isn't an option.

## Quick start

```bash
# what would be downloaded (no writes, no shard fetches)
uv run python data/download_hf_data.py --dataset qa_hard_neg_think_sft4b --plan

# do it
uv run python data/download_hf_data.py --dataset qa_hard_neg_think_sft4b \
    --steps 100000 --batch-size 16 --headroom 1.25 --select random --seed 42 \
    --out ~/hf_parquet

# then train against it — BOTH env vars are required
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET=$HOME/hf_parquet uv run train.py ...
```

> `GROUND_HF_PARQUET` **alone does nothing**: `qa.py:313` gates the offline branch on
> `HF_HUB_OFFLINE == "1"`. Set only one and you silently get live-HF and the 429 livelock.

## How it sizes each source

`total_rows = steps × batch_size` (or `--total-rows`), split **the way `qa.py` interleaves**:

- no source sets `weight` → `interleave_datasets(..., stopping_strategy="all_exhausted")` is
  uniform round-robin → every source needs `total_rows / N` (1.6M / 4 = **400k each**);
- any source sets `weight` → qa.py switches to `probabilities=[w/Σw]` → shares are proportional.

`target_rows_for()` mirrors that split deliberately. **If the two ever disagree, a source runs
dry and `all_exhausted` silently RESTARTS it** — repeating those rows for the rest of the run.

Then `shards = ceil(target_rows × headroom / rows_per_shard)`. Rows-per-shard varies ~6× across
these repos, which is why a single global shard count (`precache_hf.sh`'s `GROUND_DATA_SHARDS=N`)
cannot balance the mix:

| repo | rows | GB | rows/shard | shards for 400k |
|------|-----:|---:|-----------:|----------------:|
| `vm2825/science-qa-hard-neg-think` | 3,320,432 | 33.46 | ~44.9k | 9 / 74 |
| `vm2825/diverseqa-hard-neg-think` | 2,676,612 | 34.48 | ~39.4k | 11 / 68 |
| `vm2825/triviaqa-…-hard-neg-sft4b` | 759,858 | 1.59 | ~47.5k | 9 / 16 |
| `ragrawal36/multihop_qa_sft` | 1,348,595 | 5.47 | **~7.6k** | **53 / 177** |
| **total** | | **75.00** | | **≈ 12.2 GB** |

*(measured 2026-07-16 via the HF API; `@1.25x` headroom ≈ 15.15 GB)*

## The levers

| flag | default | what it's for |
|------|---------|---------------|
| `--dataset <name>` | — | Compose the **real Hydra dataset config** for its sources + weights, so you cache exactly what training reads. Needs `eval_set@trainer.evals=none` internally — `trainer/standard.yaml` defaults to a non-existent `eval_set/standard`, so a bare compose of `train` raises. |
| `--repos a/b=2.0 c/d` | — | Explicit repos (+ optional weight), instead of a config. |
| `--steps` / `--batch-size` | 100000 / 16 | Rows to cover. |
| `--total-rows` | — | Override the above directly. |
| `--headroom` | 1.25 | Over-provision. rows/shard is an **average** (science-qa shard 0 = 55,355 vs ~44.9k mean); landing short doesn't error, it silently repeats. |
| `--select {random,head}` | random | `head` = `files[:n]` (what `precache_hf.sh` does) — an epoch is ~12% of science-qa, so the head is a biased slice of whatever shard order correlates with. `random` is seeded, so it's deterministic and resumable. |
| `--seed` | 42 | Mixed with the repo name, so repos don't pick correlated positions. |
| `--plan` | off | Print the plan, download nothing. |
| `--max-gb` / `--min-free-gb` | — / 5.0 | Abort before filling the boot disk. |
| `--no-verify` | off | Skip the row verification + top-up. |

## Verify-and-top-up

Row counts start as estimates (datasets-server `/size`, or — for **private** repos, which it
404s on — one probed shard extrapolated). After downloading, the script reads the **local parquet
footers** for exact rows and fetches more shards until the target is met, warning loudly if a repo
simply cannot supply its share. This is the guard against the silent-repeat failure above.

Idempotent: `hf_hub_download` skips files already present, and selection is a pure function of
`(repo, seed, n)`, so re-running tops up rather than re-picking.

## Relationship to `scripts/misc/precache_hf.sh`

`precache_hf.sh` still exists and also pre-pulls the **model** checkpoints (which this script does
not). For datasets, prefer this script: `precache_hf.sh`'s `GROUND_DATA_FRAC=0.5` takes the first
half of every repo, which over-downloads (~37 GB), still can't fit the disk, takes a biased head
slice, and — because it ignores rows-per-shard — leaves the mix badly unbalanced.
