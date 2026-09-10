# LongHealth eval pipeline (built, currently returns 0 samples)

**Status: NOT WORKING. The eval runs to completion, exits 0, and evaluates zero questions.**
Two fixes are required before it produces data — one mine, one a latent defect in
`data/qa.py`. Both are diagnosed below; neither is applied yet.

## Motivation

The long-document serving-cost work
([experiment](../experiments/2026-07-19-long-document-serving-cost.md)) measures latency from a
shape-matched microbenchmark rather than a real run. Wiring LongHealth up as an actual eval gives
(a) end-to-end latency on real data, and (b) `doc_hit_rate` at long-document scale, which is a
genuine retrieval signal even though generation accuracy on this benchmark is out-of-distribution
and meaningless.

LongHealth (Bressem et al., [github.com/kbressem/LongHealth](https://github.com/kbressem/LongHealth))
is 20 fictional patient records — 133 clinical documents, 233,041 tokens — with 400 multiple-choice
questions in three categories: information extraction, negation, and sorting.

## What was built

| file | role |
|------|------|
| `datagen/longhealth/prepare_longhealth.py` | benchmark JSON → docs + QA datasets on HF |
| `configs/dataset/sources/longhealth_docs.yaml` | 133 clinical documents, column `pos_doc` |
| `configs/dataset/sources/longhealth_qa.yaml` | 400 questions, `pos_doc_ids` = gold docs |
| `configs/eval/tasks/gen_large_mem_longhealth.yaml` | eval task |
| `scripts/embed/longhealth_eval.sh` | runner |
| `datagen/longhealth/longhealth_stats.py` | corpus token statistics |

Datasets are pushed **private** (`ragrawal36/longhealth-{docs,qa}-with-ids`): LongHealth is a
third-party benchmark and re-hosting it publicly under our account is not ours to decide.

**Gold labels are exact.** Each question carries `answer_location`, naming the specific `text_N`
holding the answer, so `pos_doc_ids` is ground truth rather than a heuristic. All 400 questions
have a usable location, so `doc_hit_rate` would be exact across the whole set.

**Chunking was sized against the real distribution.** `max_chunks_per_doc: 32` (8,192-token cap)
covers 100% of the corpus; the default of 4 (1,024-token cap) would silently drop ~30% of corpus
tokens and shrink the bank without any error — the same silent-truncation failure mode this repo
has hit before.

## Why it returns 0 samples

`data/qa.py:105` drops any row whose answer text appears inside the question, unless the answer
carries a multiple-choice letter prefix:

```python
if answer and question and answer.lower() in question.lower():
    if not any(opt in answer for opt in ["A)", "B)", "C)", "D)"]):
        return False
```

This is an anti-leakage guard with a deliberate multiple-choice carve-out. It fires on every
LongHealth row because:

1. **My bug.** `format_question()` embeds the five options in the question text (correct for
   multiple choice), while `answer` is the bare option text (`"Vincristine"`). So
   `answer in question` is always true, and the letter-prefix escape never fires because the bare
   answer contains no `"C)"`. **All 400 rows dropped.**
2. **A latent repo defect.** The whitelist stops at `"D)"`. LongHealth has five options, so even
   after fix 1 the **45 questions whose answer is option E** would still be silently dropped —
   a real defect for any 5-way multiple-choice dataset, not just this one. (Answer-letter
   distribution: A 57, B 104, C 121, D 73, **E 45**.)

The failure is silent end to end: zero rows → no golds to inject → the corpus builder reports
`0 gold + 975 distractor chunks`, generation runs `0/128`, and the job exits 0.

## The fixes (not yet applied)

1. In `prepare_longhealth.py`, set `answer` to the letter-prefixed form (`"C) Vincristine"`) so the
   existing carve-out fires.
2. In `data/qa.py:105`, extend the whitelist to include `"E)"` — or better, make it a regex on a
   leading `^[A-Z]\)` so it generalises to any number of options.

A third change is worth considering independently: **the eval should fail loudly when it evaluates
zero samples.** A run that filters every row currently produces `generated_count: 0` alongside
`llm_judge_accuracy: 0.0` and reports success.

## Collateral to clean up

The failed run wrote `llm_judge_accuracy: 0.0` to
`gs://memory-layers-training-usc1/musique_ground4layer_midtrain_…-2026-07-19-00-52-42/eval/step_1500/longhealth.json`
and logged it to the wandb run at `train_step=1500`. Nothing in that record indicates zero rows
were evaluated, so it reads as a model that scored zero on LongHealth. It should be overwritten
with an explicit invalid marker or removed — **left in place pending a decision**, since deleting
logged artifacts is outward-facing.

## Test record

No passing test. The eval was run once and produced:

```
  corpus (scanned 975): 0 gold + 975 distractor chunks = 975 total (target=133)
  Generating (large mem):   0%|          | 0/128 [00:04<?, ?it/s]
[log_eval] outputs/.../longhealth.json  (0 samples, 3 metrics)
    generated_count: 0
    lexical_grounding: 0.0
    llm_judge_accuracy: 0.0
```

Corpus statistics *are* verified (`datagen/longhealth/longhealth_stats.py`):

```
EXACT corpus tokens: 233,041   docs 133
  tokens/doc  min 157  med 1534  max 8582
  tokens/word ratio: 2.12
  chunks@256: 977
correct-option distribution: {'A': 57, 'B': 104, 'C': 121, 'D': 73, 'E': 45}
```

The dry-run path is also verified: `prepare_longhealth.py --dry-run` reports 133 docs / 400
questions / 0 missing gold locations.

## Repro

```bash
uv run --no-sync python datagen/longhealth/prepare_longhealth.py --dry-run
uv run --no-sync python datagen/longhealth/prepare_longhealth.py --private
bash scripts/embed/longhealth_eval.sh     # currently yields 0 samples
```
