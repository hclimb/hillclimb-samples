# HF Rate Limits (429) — why live-HF streaming livelocks

**TL;DR:** live-HF streaming of the multi-source QA mixes does not "run slowly" under rate
limiting — it **never produces a first batch**. The quota is per HF *account*, shared across every
box and process, and `data/qa.py`'s crash handler responds to a 429 by rebuilding the whole
pipeline, which re-blows the quota. Use the offline-parquet path
([qa-dataset.md](qa-dataset.md), `scripts/misc/precache_hf.sh`) for real runs.

## The quota

`1000 API requests / 5 minutes`, **per account**, authenticated or not (a valid `HF_TOKEN` does
*not* raise it — PRO/Team does). It is shared: a training run saturating it will also break
`curl`ing the Hub API from your laptop, `precache_hf.sh`, and any eval box, all at once.

## Where the requests come from

`num_workers × sources`, in a burst at pipeline build. Each grain worker process independently
resolves every streaming source's shards:

| config | sources | `num_workers` | resolutions per box |
|--------|---------|---------------|---------------------|
| `qa_hard_neg_think_sft4b` | 4 | **16** | **64** |
| `qa_sim15` / `qa_sim40` / `qa_simheavy` | 12 | 4 | 48 |
| `qa_sim15_sym` | 16 | 4 | 64 |
| `pretraining` | 5 | 16 | 80 |

Measured on a live box: ~**900 established TCP connections** from one training run at
`16 × 4`. `a45256c` measured that halving workers halves resolution load with **no throughput
gain** — which is why the `qa_sim*` mixes were dropped to 4. **The mixes that were failing at the
time got the fix; `qa_hard_neg_think_sft4b`, `pretraining`, `hard_neg_no_cot` and
`doc_copy_hard_neg` were left at 16.**

## The livelock (the important part)

`data/qa.py`'s generator wraps `next(iterator)` in a bare `except Exception` that **rebuilds the
entire pipeline**:

```python
except Exception as e:
    warnings.warn("Dataset iterator crashed (likely due to worker timeout/OOM during JIT
                   compilation). Rebuilding pipeline automatically. Error: {e}")
    if "429" in str(e):
        time.sleep(60 + random.uniform(0, 120))   # a45256c: slows the loop, doesn't break it
    pipeline = self._build_pipeline()             # ← re-resolves every source × every worker
```

That handler predates the 429s — `git log -L` dates it to **`84ca87e` (2026-03-18)**, written for
*worker timeout/OOM during JIT compilation*. A 429 is not that, but it **is** an `Exception`, so a
rate limit triggers a full pipeline rebuild, which re-blows the quota, which raises another 429.
The warning text is actively misleading: it blames OOM/JIT while printing a 429.

Observed: 12 rebuild cycles over 47 minutes, still at step 0, no checkpoint.

**No single change "caused" the 429s.** The quota pressure arrived with fleet scale (the 5-TPU sim
campaign, `29d99a6` → `db6d039` "staggered VM starts" → `a45256c` "revert fleet to 4 grain
workers", all 2026-07-01→03). Runs before that never tripped the March handler because there were
no 429s to catch. It is an *interaction* between a pre-existing broad handler and a new failure
mode — which is why bisecting `qa.py` for "the commit that broke it" finds nothing: the code path
for `qa_hard_neg_think_sft4b` is unchanged since 2026-04-19.

## What to do

1. **Offline-parquet** — the only measured-working path. Download with
   **[`data/download_hf_data.py`](../../data/download_hf_data.py)** (see
   [epoch-sized-data.md](epoch-sized-data.md)), then launch with `HF_HUB_OFFLINE=1` +
   `GROUND_HF_PARQUET`. **Both are required**: `qa.py`'s offline branch is gated on
   `HF_HUB_OFFLINE == "1"`, so setting only `GROUND_HF_PARQUET` is a no-op and you silently get
   live-HF. 49.7 batch/s, 0 stalls, 24× the compute rate — see
   [../experiments/2026-07-16-train-speed-axes.md](../experiments/2026-07-16-train-speed-axes.md).
2. **If you must stream live**, drop `dataset.num_workers` to 4 to match the sim mixes. Reduces
   the burst; does **not** guarantee you clear the quota, and does nothing about the livelock.
3. **Don't run two data-hungry things at once** on one account — the eval boxes stream too, and
   a saturating run will 429 your laptop's `curl` and `precache_hf.sh` alike.

## Known gaps

- The `except Exception → rebuild` handler should not treat a 429 as a crash; a rate limit wants
  a retry of the *existing* iterator (or a bounded, non-rebuilding backoff), not a rebuild. Not
  fixed yet — it needs care, since a genuinely dead grain worker *does* require a rebuild.
- `num_workers: 16` still stands in `qa_hard_neg_think_sft4b`, `pretraining`, `hard_neg_no_cot`,
  `doc_copy_hard_neg` despite `a45256c` finding 16 buys no throughput.
