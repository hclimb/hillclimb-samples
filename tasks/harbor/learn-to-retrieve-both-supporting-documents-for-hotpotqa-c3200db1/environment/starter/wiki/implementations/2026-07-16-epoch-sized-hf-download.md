# Epoch-sized, source-balanced HF download (`data/download_hf_data.py`)

*2026-07-16*

**A 100k-step run needs 1.6M rows; the mix is 75 GB. Size the download by rows-needed and it's
15 GB — which fits the boot disk, which is what makes the offline path possible at all.**

## Motivation

Live-HF streaming of `qa_hard_neg_think_sft4b` livelocks: `num_workers=16 × 4 sources` = 64 shard
resolutions blow HF's per-account 1000-req/5-min quota during pipeline build, and `qa.py`'s
`except Exception → rebuild` handler responds to the 429 by rebuilding the pipeline, re-blowing
the quota (12 cycles / 47 min / still step 0 — see [../data/hf-rate-limits.md](../data/hf-rate-limits.md)).
The measured-working alternative is offline-parquet, which needs shards on disk first.

`scripts/misc/precache_hf.sh` did that with `GROUND_DATA_FRAC=0.5` — the first half of every
repo's shards — and that has three problems:

1. **Over-downloads, still under-covers.** Half the mix is ~37 GB against a 97 GB disk that was
   94% full. Yet a 100k-step run at `batch_size=16` only consumes **1.6M examples**.
2. **Silently unbalances the mix.** `interleave_datasets(..., stopping_strategy="all_exhausted")`
   is uniform round-robin, so each source must supply `total/N`. rows-per-shard varies **~6×**
   (multihop ~7.6k vs science-qa ~44.9k), so an equal shard *fraction* gives wildly unequal rows —
   and `all_exhausted` **restarts** any source that runs dry. The old cache had 14G of science-qa
   but 177M of multihop: multihop would have repeated over and over all run.
3. **Head-biased.** `files[:N]` takes the front; an epoch is ~12-16% of science-qa, so that's a
   slice of whatever shard order correlates with.

## Measured inputs (2026-07-16, HF API)

| repo | rows | GB | rows/shard | shards for 400k |
|------|-----:|---:|-----------:|----------------:|
| science-qa-hard-neg-think | 3,320,432 | 33.46 | ~44.9k | 9 / 74 |
| diverseqa-hard-neg-think | 2,676,612 | 34.48 | ~39.4k | 11 / 68 |
| triviaqa-…-hard-neg-sft4b | 759,858 | 1.59 | ~47.5k | 9 / 16 |
| multihop_qa_sft | 1,348,595 | 5.47 | **~7.6k** | **53 / 177** |
| **total** | | **75.00** | | **≈12.2 GB** |

Beware the HF tree API **paginates** — an unpaginated read reported 49/128 shards for one repo and
undercounted every total (I initially reported 49 GB instead of 75 GB from exactly this).

Also measured, for the "should we pre-tokenize?" question: raw text is **10,077 bytes/row**
(science-qa) vs **~4,400** pre-tokenized — **2.3×**, concentrated in the negatives (`neg_docs`
7,546 B/row → `neg_doc_ids` ~2,600, i.e. 2.9×; they're 75% of every raw row). Not pursued here:
`QADataset` expects raw text columns, so consuming pre-tokenized data needs a loader that doesn't
exist. At 15 GB the disk pressure is gone anyway.

## Options considered

| Decision | Options | Chosen |
|---|---|---|
| Sizing | (a) shard fraction; (b) shard count; (c) **rows needed** | **(c)** — only (c) can balance sources whose rows/shard differ 6× |
| Source list | (a) hardcode; (b) **compose the Hydra dataset config** | **(b)** — caches exactly what training reads; can't drift |
| Shard choice | (a) head; (b) **seeded random** | **(b)** — head is a biased 12% slice; seeded keeps it deterministic/resumable |
| Row counts | (a) datasets-server; (b) probe a shard; (c) **both + verify locally** | **(c)** — datasets-server 404s on private repos; estimates then get verified from local footers and topped up |
| Extend `precache_hf.sh`? | (a) extend; (b) **new script** | **(b)** — bash + per-repo row math + verification wanted real code; precache_hf.sh still pre-pulls the *models*, which this doesn't |

## What was built

`data/download_hf_data.py`. The planning layer is **pure** (no network/disk) so the levers are
unit-testable: `target_rows_for` (mirrors qa.py's uniform-vs-`probabilities` interleave split),
`shards_for_rows` (headroom + round-up), `select_shards` (seeded sample vs head), `build_plan`.
I/O is separate: Hub listing/sizing with 429 backoff, then download → **verify exact rows from
local parquet footers → top up** if the estimate fell short, warning if a repo cannot supply its
share. `--plan` computes everything and downloads nothing; `--min-free-gb`/`--max-gb` guard the
disk. Levers documented in [../data/epoch-sized-data.md](../data/epoch-sized-data.md).

**Coupling to watch:** `target_rows_for` must track `qa.py`'s interleave. If they diverge, a
source runs dry and `all_exhausted` silently repeats it — a data bug with no error message. Noted
in both files.

**Latent repo bug found:** `configs/trainer/standard.yaml` defaults `evals` to `eval_set/standard`,
which **does not exist on this branch**, so any bare `compose(config_name="train")` raises
`MissingConfigException`. Training scripts don't hit it because they override the eval set.
`sources_from_dataset_cfg` composes with `eval_set@trainer.evals=none` for the same reason
`staged_ground.yaml` does. Left the config as-is (not this change's business).

## Test record

`uv run python tests/test_download_hf_data.py` (on `rohun-v6e-8-2`) — 40 checks, all pass:

```
PASS  uniform 4 sources: got=400000 want=400000
PASS  weighted 3:1 of 4: got=1200000 want=1200000
PASS  science-qa @1.0x: got=9 want=9
PASS  multihop @1.0x: got=53 want=53
PASS  headroom 1.25x: got=12 want=12
PASS  random is deterministic (=> resumable)
PASS  random is not the head  got ['data-00001.parquet', 'data-00012.parquet', ...]
PASS  different repo -> different pick
PASS  science-qa shards: got=9 want=9   PASS  diverseqa shards: got=11 want=11
PASS  triviaqa shards: got=9 want=9     PASS  multihop shards: got=53 want=53

  full-epoch plan @1.0x headroom = 12.18 GB (vs 75.00 GB for the whole mix)
PASS  epoch plan is ~12 GB  12.18 GB
PASS  science-qa-hard-ne can supply its share  3,320,432 >= 400,000
  @1.25x headroom = 15.15 GB
PASS  weighted: 3x source gets half: got=600000 want=600000
ALL PASS
```

End-to-end `--plan` against the live Hub + real Hydra config (`scripts/misc/run_download_hf_data_checks.sh`):

```
Sizing for 1,600,000 rows (100,000 steps x 16)
Sources: 4  select=random seed=42 headroom=1.25x  out=/home/rohunagrawal/hf_parquet

repo                                                      need       shards   est GB  of repo
vm2825/science-qa-hard-neg-think                       400,000    12/74         5.43    16.2%
vm2825/diverseqa-hard-neg-think                        400,000    13/68         6.59    19.1%
vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-s   400,000    11/16         1.09    68.8%
ragrawal36/multihop_qa_sft                             400,000    66/177        2.04    37.3%
TOTAL                                                1,600,000                 15.15
(headroom 1.25x -> ~2,075,420 rows on disk)

disk: 83.6 GB free; plan needs ~15.1 GB
```

Live plan == the unit test's 15.15 GB. Note 16% of science-qa vs **69%** of triviaqa — the
imbalance a global fraction cannot express.

**Full download, `rohun-v6e-8-0`** (`scripts/misc/download_hard_neg_data.sh`) — every source clears
its share, so nothing runs dry and repeats:

```
=== vm2825/science-qa-hard-neg-think: 12 shards ===
  vm2825/science-qa-hard-neg-think: 455,647 rows on disk (target 400,000)
=== vm2825/diverseqa-hard-neg-think: 13 shards ===
  vm2825/diverseqa-hard-neg-think: 441,691 rows on disk (target 400,000)
=== vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b: 11 shards ===
  vm2825/triviaqa-...-hard-neg-sft4b: 509,858 rows on disk (target 400,000)
=== ragrawal36/multihop_qa_sft: 66 shards ===
  ragrawal36/multihop_qa_sft: 502,988 rows on disk (target 400,000)
DONE -> /home/rohunagrawal/hf_parquet

13G  /home/rohunagrawal/hf_parquet
  5.3G  vm2825__diverseqa-hard-neg-think        4.3G  vm2825__science-qa-hard-neg-think
  2.0G  ragrawal36__multihop_qa_sft             909M  vm2825__triviaqa-...-hard-neg-sft4b
/dev/root  97G  66G  32G  68% /
```

**~1.91M rows in 13 GB** (vs 75 GB for the mix). Actual came in under the 15.15 GB estimate —
the plan sizes from *average* shard bytes and the sampled shards ran smaller. The verify path
confirmed every source from local footers; no top-up was needed.

## Repro

```bash
TPU_NAME=rohun-v6e-8-2 RUN_SCRIPT_PATH=scripts/misc/run_download_hf_data_checks.sh DETACH=0 \
  bash scripts/infrastructure/multi-vm-tpu-run.sh      # v6e-8, europe-west4-a
```

## Known gaps

- **The top-up branch specifically is unexercised**: at 1.25× headroom every source cleared its
  target on the first pass, so `download_source`'s "fetch more shards" loop never ran. The
  verification itself did run (exact rows from local footers, all four sources).
- **Pre-tokenized storage (2.3×) is unused**; needs a loader for the
  `input_ids/pos_doc_ids/neg_doc_ids` schema.
- **`precache_hf.sh` is untouched** and still the only thing that pre-pulls model checkpoints.
- The `except Exception → rebuild` 429 livelock in `qa.py` is **not fixed** — this change routes
  around it. See [../data/hf-rate-limits.md](../data/hf-rate-limits.md) "Known gaps".

Reference pages updated: [data/epoch-sized-data.md](../data/epoch-sized-data.md) (new),
[data/hf-rate-limits.md](../data/hf-rate-limits.md) (new),
[data/data-preparation.md](../data/data-preparation.md), [data/README.md](../data/README.md).
