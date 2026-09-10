# MuSiQue → multi-hop SFT finetuning set

**Date:** 2026-07-18 · **Author:** rohunagrawal · **Status:** done (dataset built; not yet trained on) · **Commit:** _TBD_

**Result: `ragrawal36/musique-sft` — 13,153 rows (65.97% of 19,938), 10 shards.**

| Metric | Value |
|---|---|
| Yield, 2-hop | 10,187 / 14,376 = **70.9%** |
| Yield, 3-hop | 2,381 / 4,387 = **54.3%** |
| Yield, 4-hop | 585 / 1,175 = **49.8%** |
| `hop_grounding` = 1.0 | 79.9% (only 1.1% below 0.5) |
| Training target length | median 13 chars (gold answer) |
| CoT length | median 882 chars |
| `pos_doc` / `neg_doc` per row | 2.27 / 17.73 |
| Rows citing `[Document N]` | 1 of 13,153 |
| Empty answers · duplicate ids | 0 · 0 |

**Yield falls with hop count** (70.9% → 54.3% → 49.8%), so the accepted set is mildly skewed
toward easier questions: 77.4% 2-hop vs 72.1% in the source. Expected — harder questions need
longer chains and fail the judge more — and far milder than a 400-token budget would have made
it. Correctable with source weighting or a recovery pass over the rejected 3/4-hop rows.

A two-stage pipeline turning MuSiQue into a `QADataset`-consumable SFT set with generated CoT:
`datagen/musique/prepare_musique_sft_base.py` flattens MuSiQue's 20-paragraph rows into
`{pos_doc[], neg_doc[]}` and `datagen/musique/generate_musique_sft.py` adds a Qwen3-4B chain of
thought on a TPU box. Consumed via `configs/dataset/sources/musique_sft.yaml`. **No loader
changes were needed** — the generic normalizer already handles list-valued docs.

## What a row looks like

A 4-hop example (`4hop1__101723_221169_833580_61459`, `hop_grounding=0.75`). The prompt at training
time is the **question only** — the 4 gold and 16 distractor paragraphs go into the memory bank,
and loss runs on the target span alone:

```
QUESTION: What year did Italy enter the same conflict as did Albert I of the country hosting the 1920 Summer Olympics where the country Don Kulick lives in sent a team?

GOLD DOCS: Military history of Italy during World War I, Sweden at the 1920 Summer Olympics, Don Kulick, Queen Elisabeth Medical Foundation

TARGET (loss on this span only):
<think>

Okay, let's break this down. The question is asking for the year Italy entered the same conflict as Albert I of the country hosting the 1920 Summer Olympics, where Don Kulick lives.

First, I need to figure out which country hosted the 1920 Summer Olympics. The 1920 Olympics were in Antwerp, Belgium. So the host country is Belgium. Then, the question mentions Don Kulick, who is from Sweden. Wait, the document says Don Kulick is a professor at Uppsala University in Sweden. So the country where Don Kulick lives is Sweden.

Now, Albert I of Belgium is mentioned in the Queen Elisabeth Medical Foundation document. The 1920 Olympics were hosted by Belgium, so Albert I was the king of Belgium. The ...</think>

1915
```

More samples (9, spanning 2/3/4-hop, with full documents) can be regenerated into
`results/musique_sft_samples.json` — `results/` is gitignored, so they are not committed.

Two caveats visible in the samples: the CoT carries Qwen3's thinking-mode voice ("Wait," "Hmm,"
self-correction, dead ends), and `hop_grounding` cannot distinguish coherent reasoning from a
confused ramble that happens to contain the gold intermediate strings — one sampled row scores
1.0 while explicitly saying it cannot find the fact it needs.

## Motivation

MuSiQue was already in the repo, but **only as a held-out eval benchmark**: 7 tasks under
`configs/eval/tasks/gen_*_musique_*.yaml` and 4 eval sets including the active
`configs/eval_set/hard_neg_think_c512.yaml`. `datagen/musique/prepare_musique_supporting_list.py`
built an eval-only artifact (`split: validation`) that keeps the 2–4 supporting paragraphs and
**discards the 16–18 distractors**.

Those distractors are the interesting part. The `vm2825/*` hard-neg sources in the current mix
have to *mine* negatives with an embedding model and then defensively drop anything scoring
>0.95 (`data/utils.py::make_normalizer`, `neg_score_threshold`) because high-similarity mined
negatives are frequently false negatives. MuSiQue's distractors are human-annotated
(`is_supporting=false`), so they need no scoring and carry no false-negative risk — strictly
cleaner retrieval supervision than anything else in the mix.

Prior state: [data/dataset-configs](../data/dataset-configs.md), [data/qa-dataset](../data/qa-dataset.md).

## Contamination guard

Training on MuSiQue changes what the existing MuSiQue evals measure (zero-shot generalization →
in-domain). The split hygiene is clean — train 19,938 / validation 2,417, and every eval reads
validation — but the guard is structural rather than conventional: **stage 1 only emits the
train split**, and `--split validation` warns loudly. The finetuning artifact therefore cannot
carry eval data.

Note that split *names* are not a reliable filter here: `gen_large_mem_musique_supporting.yaml`
loads `musique-supporting-with-ids-validation` with `split: train` — a validation-derived repo
whose split is named `train`. Key off MuSiQue `id`s, not split names.

## Options weighed

| Decision | Chosen | Rejected, and why |
|---|---|---|
| CoT source | Generate with Qwen3-4B over the full 20-paragraph context | **Templated from `question_decomposition`** — free and correct-by-construction, but produces uniform robotic phrasing unlike the rest of the mix. **Supporting-only context** — the memory model must discriminate gold from distractor at retrieval time, so the CoT should demonstrate that, not read pre-filtered text. |
| Training target | Gold `answer` (no `field_map`) | **`generated_answer`**, the convention the `vm2825/*` sources use — but those are synthetic and have no gold. MuSiQue does, and gold is 12 chars vs 151 for the model's paraphrase, matching what the official evaluator and the existing MuSiQue eval tasks score. The judge has already confirmed the generation agrees with gold, so `think` stays consistent with the target. |
| Reasoning fidelity | Record `hop_grounding`, don't filter on it | Filtering during generation bakes in a threshold before any evidence for it exists. One pass is expensive; the column can be thresholded on the parquets afterwards. (In the event: 79.9% at 1.0 and only 1.1% below 0.5 — no filtering needed. Deciding upfront would have been a guess.) |
| Doc encoding | Native list columns | Separator-joined strings (the `vm2825/*` convention) — `make_normalizer` passes lists through unchanged (`data/utils.py:130,139`), so the separator is pure overhead, and the misspelled `<doc_seperator>` is a live footgun. |
| Negative filtering | None (`neg_score_threshold` unset) | Thresholding is meaningless on gold labels. |

## How it was built

**Stage 1** — `prepare_musique_sft_base.py::process_row` splits `paragraphs` on `is_supporting`
into `pos_doc`/`neg_doc` (formatted `**{title}**\n{text}`, matching the existing
`prepare_musique_supporting_list.py`), carries `answer_aliases` and `question_decomposition`,
and writes 10 sharded parquets to `{HF_USERNAME}/musique-sft-base`. Rows with no gold paragraph
are dropped rather than falling back to "all paragraphs positive", which would poison
`pos_doc_mask`.

**Stage 2** — `generate_musique_sft.py`, modelled on `datagen/generate_multihop_sft.py`
(generate → token filter → LLM-judge filter → upload matching parquet; resumable by listing the
output repo). Four deliberate differences:

1. `format_paragraphs()` re-scatters gold among distractors with a per-row seed from the MuSiQue
   `id`. Stage 1's pos/neg split destroys the original ordering and would otherwise leave all
   gold first — a positional shortcut the CoT could exploit. Deterministic across reruns.
2. The judge sees `answer_aliases`; MuSiQue answers are entities with common surface variants.
3. `hop_grounding` — fraction of gold intermediate answers appearing verbatim in the think
   block. The judge only checks the *final* answer, so it passes a lucky shortcut that skipped
   a hop; this measures whether the chain was actually walked. Substring match, so it
   under-counts paraphrase — a floor, not a measure.
4. `detect_tpu_chips()` sizes tensor parallelism from `/dev/vfio` instead of hardcoding 8.

**The think+answer ceiling must sit below the training `seq_len`, not at it.**
`_StreamingQAFilter` (`data/qa.py:105-122`) drops any row whose *full* formatted sequence
exceeds `seq_len`, and that sequence also carries the chat template and question. Measured on
Qwen3-4B the prefix is 33 tokens, so:

| training `seq_len` | think+answer budget | worst-case total | budget = `seq_len` would give |
|---|---|---|---|
| 512 | 400 | 434 ✓ | 554 ✗ |
| 1024 | 950 | 984 ✓ | 1066 ✗ |

Inheriting `generate_multihop_sft.py`'s 512 budget under `seq_len` 512 would have silently lost
rows at training time, long after the TPU cost was paid. **The shipped data is generated at 950
(`seq_len` 1024)**; the 512 variant is a filter away. Both pairings are asserted in the tests.

### New config

`configs/dataset/sources/musique_sft.yaml` (no `field_map` — train on the gold `answer`;
`think_field: think`) and standalone `configs/dataset/musique_sft.yaml`. Three non-default
values there matter:

- **`seq_len: 1024`** — must match the budget the CoT was generated at. Lowering it to 512 does
  not shorten the data, it makes `_StreamingQAFilter` silently drop every over-length row (~35%,
  disproportionately 3/4-hop). To train at 512, filter the parquets first.

- **`min_doc_length: 0`** — the default 64-token floor is applied to `pos_doc[0]` only
  (`data/qa.py:85-91`), and 16.1% of MuSiQue gold paragraphs are shorter than that. The default
  would have silently dropped ~1/6 of the dataset. (`multihop_qa_sft_midtraining.yaml` already
  sets this for the same reason.)
- **`num_chunks_per_doc: 20`** — one chunk per paragraph at ~152 tokens each. `pack_docs` packs
  positives first, so gold always survives and a smaller M costs only distractors; 20 keeps the
  full annotated set at the price of a `16×20×256 = 81,920`-slot bank (vs 65,536 at M=16, +25%).

**Folding into the main mix:** `min_doc_length` is dataset-level, not per-source, so adding
`musique_sft` to `qa_hard_neg_think_sft4b.yaml` and relaxing the floor there also relaxes it for
the `vm2825/*` sources, which currently rely on it. Hence the standalone config.

## The numbering defect — read this before generating CoT for a memory model

The first two generation passes numbered the paragraphs in the prompt (`[Document 1]`,
`[Document 2]`, …) so the gold-vs-distractor task was legible. Measured on the output:

| | numbered | unnumbered |
|---|---|---|
| `think` citing `[Document N]` | **99.6%** | **0.0%** |
| `generated_answer` citing `[Document N]` | 88.2% | 0.0% |
| accepted | 54.1% | 54.6% |

**At training time the prompt holds only the question** — the paragraphs are in the memory bank
as retrieved vectors, unnumbered and unordered. So a CoT that says "Document 3 states…" refers
to a scheme that cannot exist at inference: it trains the model to emit citations it can never
ground. Every aggregate metric looked healthy while this was happening (54% yield, 85%
grounding, judge-verified) — it was visible only by reading rows.

Generalises to any generated-CoT data for this framework: **the generator's prompt shows
documents inline, the trained model's prompt does not.** Anything the CoT says about the
*presentation* of the documents — numbers, positions, "the first passage" — is unlearnable.
Paragraph `**Title**` headers are safe because they are part of the memorised content.

Fixing it cost nothing in yield. A 0.9% residue still refers to "the first document" in prose;
the `think` column is in the parquet, so that can be filtered without regenerating.

## MuSiQue's train split is ordered by hop count — shuffle before sharding

`dgslibisey/MuSiQue`'s train split is sorted by hop type. Sharded in source order, the base
repo came out **shards 0-6 pure 2hop, 7-8 pure 3hop, 9 the 4hop tail**. Three consequences,
all of which bit before it was spotted:

- **Every yield number measured on shards 0-2 was a 2-hop-only number.** The 400-vs-950 token
  A/B below is still valid (all three shards were 2hop, so it is like-for-like) but it is *not*
  a whole-dataset yield — 3/4-hop questions need longer chains and will accept at a lower rate.
- **A partial or interrupted stage-2 run silently produces a hop-skewed dataset.** Stage 2 is
  resumable per shard, so stopping early is normal, not exceptional.
- **A tight token budget truncates the hard questions preferentially**, so the bias compounds:
  the rows lost are disproportionately the multi-hop ones the dataset exists to teach.

`prepare_musique_sft_base.py` now shuffles with `--shuffle-seed` (default 42) before sharding,
and logs each shard's hop mix so a recurrence is visible in the log. Post-fix every shard is
~71% 2hop / ~23% 3hop / ~6% 4hop.

## Token budget: 400 (seq_len 512) vs 950 (seq_len 1024)

Measured on 2-hop-only shards, so like-for-like but not a whole-dataset rate:

| | 400 tok (`seq_len` 512) | 950 tok (`seq_len` 1024) |
|---|---|---|
| accepted | 52.4% (mean of 2 shards) | **68.2%** |
| token-filtered | 8.5% | 3.2% |
| no-think (over output budget) | 29.5% | 14.8% |
| judge-rejected | 9.8% | 13.8% |
| `hop_grounding` = 1.0 | 82% | 83% |

The gain is smaller than the recovered length-filter buckets suggest because judge rejection
*rises*: rows that previously died on length now reach the judge and some fail there. Quality
of the accepted set is unchanged.

**Generate at the loose budget regardless of the training `seq_len` you intend.** A row
accepted under a 400-token ceiling is a length-filtered subset of what is accepted at 950, and
`think` is in the parquet — so the `seq_len` 512 dataset is a one-line filter away, with no
regeneration. Generating at 400 is the irreversible direction. `--max-think-ans-tokens`,
`--max-output-tokens` and `--max-model-len` are flags for this reason.

Raising the output budget also forces `--max-model-len` up: vLLM rejects a request when
`prompt + max_tokens > max_model_len`, and MuSiQue's longest contexts tokenise to ~7.5k, so an
1100-token output budget overflows the default 8192. Run at 12288.

Cost note for `seq_len` 1024 at training time: `padding="max_length"` pads every row, so it
looks like 2×, but the memory bank dominates token volume — `16 × 20 × 256 = 81,920` tokens
through the 0.6B embed tower vs `16 × 512 = 8,192` through the 4B main model. FLOP-weighting by
parameters, doubling the LM sequence is roughly **1.4×** total, not 2×.

## Prompt: brevity A/B

The first shard yielded only 43.5%. The loss was **not** accuracy — the judge rejected just 3.8%
— but verbosity: 38.4% exceeded the token budget and 14.3% blew past the output cap entirely.
Tightening the instruction to a hard 100-word budget plus "never quote or restate a document":

| | shard 0 (verbose) | shard 1 (tightened) |
|---|---|---|
| accepted | 870 (43.5%) | **1081 (54.1%)** |
| token-filtered | 767 (38.4%) | **214 (10.7%)** |
| no-think | 286 (14.3%) | 531 (26.6%) |
| judge-rejected | 75 (3.8%) | 174 (8.7%) |
| input-too-long | 2 | 0 |
| `hop_grounding` = 1.0 | 84% | 85% |

`no-think` rose because `MAX_OUTPUT_TOKENS` also dropped to 512, truncating more responses
before `</think>` closes — but anything truncated at 512 exceeds the 400 ceiling and would have
been filtered regardless, so no accepted row was lost. `judge-rejected` rose genuinely (terser
reasoning errs more), but rejected rows are discarded; the accepted set stays judge-verified at
unchanged grounding.

Not a clean ablation — prompt and output budget changed together — but the confound is
conservative (a smaller output budget can only suppress yield), so the prompt's contribution is
at least the observed +10.6pp. Shard 0 was deleted and regenerated so the set is homogeneous.

## Repro

```bash
# stage 1 (no TPU needed, but run on the box for the uv env + HF token)
uv run python datagen/musique/prepare_musique_sft_base.py
# stage 2, per shard-batch
uv run python datagen/musique/generate_musique_sft.py            # all pending shards
uv run python datagen/musique/generate_musique_sft.py --limit 1  # one shard (probe)

# both, via the launcher (PHASE=probe|smoke|full)
TPU_NAME=tpu-v6e-vm ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce \
RUN_SCRIPT_PATH=scripts/data/musique_sft_box_run.sh RUN_ENV="PHASE=full" \
  bash scripts/infrastructure/multi-vm-tpu-run.sh
```

**TPU:** `v6e-4` (`ct6e-standard-4t`, project `memory-layers`, zone `europe-west4-a`).
**Generator/judge:** `Qwen/Qwen3-4B` via vLLM, temp 0.6 (gen) / 0.0 (judge).
**Artifacts:** `ragrawal36/musique-sft-base` (19,938 rows, 10 shards) →
`ragrawal36/musique-sft`. No wandb run (datagen, not training); no checkpoints.
**Throughput:** ~2,000 rows / 6.5 min / shard.

The A/B numbers below were taken before the numbering fix and are **not** reproducible from the
current tree — the prompt has changed twice since (brevity limit, then unnumbering). They are
recorded for the reasoning, not as a rerunnable result; the post-fix baseline is the
unnumbered column in the table above.

## Infrastructure

`tpu-v6e-vm` is a **Compute Engine VM with an attached TPU** (`ct6e-standard-4t`), not a Cloud
TPU node — invisible to `gcloud compute tpus tpu-vm`, so `multi-vm-tpu-run.sh` could not address
it at all. Added an opt-in `TRANSPORT=gce` mode swapping the two `gc_ssh`/`gc_scp` bodies;
`TRANSPORT=tpu` remains the default and existing launches are untouched.

Killing a generation run leaves `EngineCore` workers holding `/dev/vfio/*`; `pkill -f "vllm
serve"` matches only the server parent, so the next launch downloads the whole model and *then*
dies on `Device or resource busy`. `generate_musique_sft.py` now calls
`evals/vllm.py::VLLMInference.free_tpu_devices()` before starting the server.
**`datagen/vllm_inference.py::start_server` still lacks the `free_tpu` parameter its `evals/`
counterpart has** — the two have drifted, and `generate_multihop_sft.py` is exposed to the same
failure. Not fixed here (out of scope).

(Aside, since it cost time twice: `pkill -f <pattern>` run over ssh matches the `bash -c`
wrapper carrying that same pattern in its argv, so the command kills its own session and gcloud
reports a bare `rc=255` that reads like a network fault. Use `[p]attern`.)

`datagen/vllm_inference.py::start_server` launched vLLM with a plain `uv run`, which re-syncs the
venv and reverts the pinned `fastapi`/`starlette`/`prometheus-fastapi-instrumentator` that
`vllm-tpu==0.12.0`'s HTTP server needs (see the `pyproject.toml` note); `evals/vllm.py` already
used `--no-sync` for exactly this reason. Added `--no-sync` to match.

Kept `datagen/vllm_inference.py`'s client rather than `evals/vllm.py`'s: the latter reads
`reasoning_content`, which requires a `--reasoning-parser` flag its `start_server` never passes,
so it would return `think=None` and leave `<think>` inline in the answer.

Box gotchas (10 GB root disk, system Python 3.10 → `uv venv` picking 3.14) are recorded in
[experiment-launch-instructions](../infrastructure/experiment-launch-instructions.md).

## Reference pages updated

- [infrastructure/experiment-launch-instructions](../infrastructure/experiment-launch-instructions.md)
  — GCE-vs-Cloud-TPU box shapes, `TRANSPORT=gce`, disk resize, Python pinning.
- [data/dataset-configs](../data/dataset-configs.md) — the new source and dataset configs.

## Tests

`uv run python tests/test_musique_sft.py` (on the box; tests 4–5 need the Qwen3-4B tokenizer):

```
[1] stage-1 process_row              9 PASS
[2] hop_grounding_score              9 PASS
[3] format_paragraphs                6 PASS
[4] token budget 400 vs seq_len=512
  PASS  worst-case row fits seq_len=512 (got 434)
       prefix=33 tok, think+answer budget=400, total=434
  PASS  a budget of 512 would overflow seq_len=512 (got 554) — justifies 400
[4] token budget 950 vs seq_len=1024
  PASS  worst-case row fits seq_len=1024 (got 984)
       prefix=33 tok, think+answer budget=950, total=984
  PASS  a budget of 1024 would overflow seq_len=1024 (got 1066) — justifies 950
[5] normalizer round-trip with musique_sft field_map   9 PASS
======================================================================
37 passed, 0 failed
```

Both shipped budget pairings are covered, because the generated data is at 950 while the
`seq_len` 512 dataset is derived by filtering — a regression in either would silently drop rows
at training time rather than fail loudly here.

Stage 1 end-to-end: `19938 raw rows → 19938 kept | 0 skipped`, shuffled, 10 shards uploaded,
each ~71% 2hop / ~23% 3hop / ~6% 4hop.

## Midtraining run — completed 2026-07-18

**`musique_sft_midtrain_topk64_seq1024_chunks20_bs32-2026-07-18-17-35-52`** — 1250 steps in
**1 h 01 m** (~2.4 s/step) on a **4-chip v5p** (`tpu-v5p-4-uscentral1a`), `rc=0`, 5 checkpoints
(250…1250) in `gs://memory-layers-training-usc1/…`.
[wandb](https://wandb.ai/memory-layers/memory-layers/runs/musique_sft_midtrain_topk64_seq1024_chun-2026-07-18-17-35-52).

| step | Loss | CE | implied `doc_access_loss` |
|---|---|---|---|
| ~410 | 0.7655 | 0.6058 | ~1.60 |
| 502 | 0.2998 | 0.2874 | ~0.12 |
| 822 | 0.2771 | 0.2688 | ~0.08 |
| 1206 | 0.2380 | 0.2283 | ~0.10 |

CE 0.61 → 0.23. **Batch 32 at `seq_len` 1024 (163,840 bank slots) fits a 4-chip v5p** — v5p is
~95 GB HBM/chip vs v6e's ~31 GB, so 4 chips give ~380 GB against a v6e-8's ~248 GB. Throughput
was also far better than a naive FLOPs ratio predicts (~2.4 s/step, not the ~4× slowdown
estimated): the memory-bank work dominates, and it suits v5p's per-chip HBM.

⚠️ **Do not read the `doc_access_loss` collapse as learned retrieval.** It falls from ~1.6 to
~0.1 within 100 steps, but the in-batch bank only ever holds that row's own 20 documents plus
other rows' documents, so it **cannot distinguish learned retrieval from memorising 13k rows over
3 epochs**. Only a corpus-scale eval can (`gen_large_mem`, gold among 512 documents). If corpus
`doc_access_acc` does not move while in-batch loss collapses, it memorised. And since MuSiQue is
now in-domain, **msmarco/hotpotqa in `hard_neg_think_c512` are the honest read.**

### Four blockers, all specific to a fresh flex-start box

None were visible on the pre-existing TPU boxes, which had accumulated state (HF cache, working
venv) that masked them. Any new box hits all four:

1. **Launcher could not address it** — a GCE-attached TPU is invisible to the Cloud TPU API.
   Fixed by `TRANSPORT=gce` (§1 of the launch runbook).
2. **`jax.distributed.initialize()` → `IndexError`** — JAX parses `worker-network-endpoints` as
   Cloud TPU `id:id:ip` triples; a GCE-attached TPU publishes the bare instance name. Guarded in
   `train.py::_init_jax_distributed`.
3. **`HF_HUB_OFFLINE=1` blocked the model download** — the flag is required for the
   offline-parquet data path but also gates `snapshot_download` of the weights. Fixed with
   `MODELS_ONLY=1 bash scripts/misc/precache_hf.sh`.
4. **Provisioning** — 9 flex requests before one was granted, and only for v5p in
   `us-central1-a`. See the capacity table in the launch runbook.

Also: the launcher's log relay can lag the box by hundreds of steps. `~/runs/<session>.log` on
the box is the source of truth, not the local tail.

## Midtraining run — configuration

`scripts/embed/train_musique_sft_midtrain.sh` fine-tunes the hard-neg-think 4B checkpoint on
this dataset alone. Config verified to compose with `scripts/misc/check_train_config.sh`.

| Setting | Value | Why |
|---|---|---|
| `resume_from` | `gs://…/qwen3_mem_embed/**100000**` | The trailing step is load-bearing: `utils.py::setup_checkpointing` parses it into `resume_step`, selecting the *weights-only, fresh optimizer, step 0* branch. Without it, a full resume at step 100000 exits immediately against `steps=1250`. |
| `trainer` | `midtraining_telemetry` (new) | `midtraining` + the weight-0 telemetry block. NOT `staged*` — its stage 0 re-freezes the main model and zeroes CE for 10k steps, a warmup for a *fresh* model that would undo a converged checkpoint. |
| trainable | `mem_*`, `embed_model`, main layers 13/14/15 | From `midtraining`. Memory layer is 14, so this is that layer and its neighbours. |
| `learning_rate` | 1e-4 | The `standard` default, chosen over the more cautious 1e-5. `staged_sim` uses 1e-5 for stage 3 "to protect the 4B"; drift would show first in the out-of-domain msmarco/hotpotqa tasks. |
| `warmup_frac` | 0.1 (added) | `midtraining` ships with none. A warm start restores weights but not the optimizer, so Adam's moments begin at zero. |
| `batch_size` / `steps` | 32 / 1250 | 40,000 samples = 3.04 epochs. These move together — batch 32 at 2500 steps would be 6.1 epochs, i.e. memorisation on 13k rows. |
| `seq_len` | 1024 | Must match the 950-token generation budget; see above. |
| `mem_top_k` | 64 | Matches the checkpoint (config default is 128). |
| `checkpoint_interval` / `max_to_keep` | 250 / 12 | 3000-step window, so the whole run stays evaluable. |
| in-loop evals | off | Same rationale as `train_hard_neg_think.sh`; score from a separate eval box. |

**Needs a v6e-8.** The bank is `32 × 20 × 256 = 163,840` slots at `seq_len` 1024, vs `65,536`
at `seq_len` 512 / batch 16 for the run that produced the checkpoint — 2.5× the bank and 2× the
sequence. Fallback ladder if it OOMs: `dataset.batch_size=16 trainer.steps=2500` (same sample
budget), then `dataset.num_chunks_per_doc=16` (costs 4 distractors/row; gold always survives
because `pack_docs` packs positives first).

⚠️ **MuSiQue evals become in-domain for any model from this run.** `hard_neg_think_c512`
includes musique@512; after midtraining those numbers no longer measure zero-shot
generalisation and are not comparable to the pre-midtraining checkpoint. msmarco and hotpotqa
in that suite stay out-of-domain and are the honest read.

## Follow-ups & risks

- **Yield falls with hop count** (70.9% / 54.3% / 49.8%), leaving the set mildly 2-hop-skewed
  (77.4% vs 72.1% in source). A recovery pass over rejected 3/4-hop rows — feeding
  `question_decomposition` as a hint — would both raise yield and correct the skew. Not needed
  for a usable set, but it is the highest-value follow-up.
- **Judge rejection (14.9%) is now the largest loss bucket**, ahead of over-budget generations
  (14.4%). At the 400-token budget length dominated; raising it moved the bottleneck to
  correctness, so further budget increases will return little.
- **0.9% of `think` blocks still say "the first document"** in prose, even unnumbered.
  Filterable on the parquets via the `think` column; not worth a regeneration pass.
- **`datagen/vllm_inference.py` has drifted from `evals/vllm.py`** — no `free_tpu` parameter,
  and until this change no `--no-sync`. Worth reconciling the two rather than patching callers.
- **Not yet trained on.** No claim about downstream effect; that needs an experiment write-up.
- **Contamination is a live decision**, not a solved problem. Training on MuSiQue train makes
  every existing MuSiQue eval in-domain. Re-baseline before comparing across that boundary.
- **`hop_grounding` is recorded but unused.** If terser CoT later proves to skip hops, threshold
  the parquets — no regeneration needed.
- The repo has **no `.python-version`**, so every fresh box will hit the 3.14 trap. Adding one
  would make it self-correcting.
