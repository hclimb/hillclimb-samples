# Dataset Configs

`configs/dataset/*.yaml` — selected with `dataset=<name>`. Top-level configs set the
`QADataset`/etc. fields; `configs/dataset/sources/*.yaml` define individual sources composed
into a mix.

## Common fields (`qa.yaml`)
`name` · `seq_len` (Q/A length, 512) · `doc_chunk_seq_len` (per-chunk, 256) ·
`num_chunks_per_doc` (4) · `batch_size` · `shuffle` · `chat_template` · `provide_docs` ·
`mask_prefix` · `split` · `num_workers` · `distill` · `hf_name` (single) **or** `sources`
(interleave).

## Multi-source mixes
A mix composes sources via Hydra defaults, e.g. `pretraining.yaml`:
```yaml
defaults:
  - /dataset/sources@sources.science_qa: science_qa
  - /dataset/sources@sources.diverse_qa: diverse_qa
  - ...
```
Each `sources.<key>` is one source config (see [normalizers.md](normalizers.md) for its keys).
> The default training dataset is `pretraining_cot`, which is **empty on this branch** — real
> runs point `dataset=` at a concrete mix (e.g. `pretraining`, `qa_sim40`, `msmarco_triplets_sft4b`).

## Source families (`configs/dataset/sources/`)
| Family | What |
|--------|------|
| `science_*`, `diverse_*`, `multihop_qa_sft`, `squad_qa` | core pretraining QA (with/without CoT, hard-neg, think) |
| `musique_sft` | MuSiQue multi-hop QA + generated CoT, **train split only** (see below) |
| `etd_*` | text-pair similarity sources (`mask_ce` retrieval-only rows) |
| `msa_*_{qa,docs}` | the 9 MSA benchmark datasets ([data-preparation.md](data-preparation.md)) |
| `flashrag_*`, `hotpotqa_*` | FlashRAG / HotpotQA (`normalizer_type`) |
| `hotpotqa_hard_neg_reasoning_modified` | `vm2825/hotpotqa-hard-neg-reasoning-embedding-modified-SFT-4B-parts-0-1` — HotpotQA, vm2825 hard-neg-SFT4B schema. `neg_score_threshold` is a **constant placeholder**, not a real score — see below |
| `*_with_ids` | id-tagged QA variants that carry `pos_doc_ids` → `doc_access_acc` |
| `*_docs` | raw-document configs for `DocumentsDataset` (bank building) |
| `longhealth_{docs,qa}` | LongHealth clinical QA — 133 docs / 233k tokens / 400 MC questions. **Currently unusable: every row is filtered out**, see [implementation note](../implementations/2026-07-19-longhealth-eval-pipeline.md) |

## `musique_sft` — two things that differ from every other mix

Built by `datagen/musique/{prepare_musique_sft_base,generate_musique_sft}.py`
([implementation note](../implementations/2026-07-18-musique-sft-dataset.md)). Its negatives are
MuSiQue's human-annotated distractors, not embedding-mined ones, so it sets **no**
`neg_score_threshold` and needs **no** `doc_separator` (`pos_doc`/`neg_doc` are native list
columns, which `make_normalizer` passes through unchanged).

⚠️ **Train split only.** MuSiQue validation is a live eval benchmark
([eval-configs](../evaluation/eval-configs.md)); `prepare_musique_sft_base.py` refuses the
validation split without an explicit flag so the finetuning artifact cannot carry eval data.
Training on it makes every existing MuSiQue eval in-domain — re-baseline before comparing
across that boundary.

⚠️ **The CoT is generated at a 950-token think+answer budget** (`seq_len` 1024). To train at
`seq_len` 512 instead, filter the parquets on `think` length rather than regenerating — the
loose budget is a superset. Stage 1 shuffles before sharding because MuSiQue's train split is
ordered by hop count; without it a partial stage-2 run yields a 2-hop-only dataset.

⚠️ **`min_doc_length: 0` is required, not stylistic.** The default 64-token floor is applied to
`pos_doc[0]` only (`data/qa.py:85-91`) and 16% of MuSiQue gold paragraphs are shorter, so the
default silently drops ~1/6 of the dataset. Because `min_doc_length` is **dataset-level, not
per-source**, folding `musique_sft` into `qa_hard_neg_think_sft4b.yaml` would relax the floor for
the `vm2825/*` sources too — which is why it ships as a standalone `configs/dataset/musique_sft.yaml`.

Override fields inline: `dataset.batch_size=64`, `dataset.num_chunks_per_doc=8`. The wider
Hydra composition mechanics belong to the (future) configuration section; trainer configs are
in [../training/trainer-configs.md](../training/trainer-configs.md).

## `hotpotqa_hard_neg_reasoning_modified` — `neg_scores` is a constant placeholder, not a real score

Unlike the sibling `vm2825/*` hard-neg sources (`combined_hard_neg_sft4b`,
`diverse_qa_hard_neg_think`, `science_qa_hard_neg*`), whose `neg_score_threshold: 0.95` drops
mined negatives above a real ~0-1 cosine-similarity score (near-duplicates of the positive risk
being false negatives), this dataset's `neg_scores` column is the literal string
`"0.0<doc_seperator>0.0<doc_seperator>0.0"` on **every** one of its 78,755 rows — confirmed via
the HF `datasets-server` `/rows` endpoint, not just a schema/sample check. There is no real score
to threshold on, so its source config (`configs/dataset/sources/
hotpotqa_hard_neg_reasoning_modified.yaml`) sets `neg_score_threshold: 0.0` explicitly (keeps all
3 negs/row, since `make_normalizer` keeps `score <= threshold` and `0.0 <= 0.0`) rather than
inheriting the siblings' `0.95`, which happens to also pass 0.0 but is tuned for a different,
absent, signal — and would silently stop passing if this dataset ever gets real scores added.

Standalone finetune config: `configs/dataset/hotpotqa_hard_neg_reasoning_modified_finetune.yaml`
(mirrors `qa_hard_neg_think_sft4b.yaml`'s `seq_len`/`doc_chunk_seq_len`/`num_chunks_per_doc`/
`batch_size` so the memory bank matches the checkpoint being warm-started). Launch:
`scripts/embed/train_hotpotqa_hard_neg_reasoning_finetune.sh`; stage data first with
`scripts/misc/download_hotpotqa_hard_neg_reasoning_data.sh`.

## `multihop_qa_sft` / `multihop_qa_sft_hard_neg` — synthetically generated, NOT HotpotQA-derived

⚠️ **This is not a HotpotQA-family dataset, despite training on it being aimed at multihop
QA/retrieval improvement.** `ragrawal36/multihop_qa` (source of `multihop_qa_sft` →
`mihir-1999/multihop_qa_sft-hard-neg-train`, used by `multihop_hard_neg_full` and every
`multihop_*` training recipe) is generated end-to-end by `datagen/multihop_qa/`: random walks
over a DBpedia 2022 KG snapshot (uniform-random start entity, ~55 whitelisted biographical/
institutional predicates), Wikipedia paragraphs fetched per hop (falling back to an
**LLM-fabricated** grounding sentence when no real paragraph states the fact), and an LLM
(not a human) phrasing the walked relation chain into a question. It shares no construction
method, entity-curation process, or hop-count distribution with HotpotQA (human-authored,
2-paragraph bridge/comparison questions over manually curated bridge entities — see
[Yang et al. 2018](https://arxiv.org/abs/1809.09600)). Full comparison against both HotpotQA and MuSiQue's actual construction methods (including
MuSiQue's formally-verified "connected reasoning" condition, which our data makes no attempt at):
[2026-08-16-multihop-training-data-vs-hotpotqa-eval-mismatch.md](../experiments/2026-08-16-multihop-training-data-vs-hotpotqa-eval-mismatch.md).
Treat further training on this data as changing the model's fit to a *DBpedia-relation-chain*
question distribution, not a HotpotQA/MuSiQue-like one — hotpotqa/musique hybrid evals of
checkpoints trained heavily on it are out-of-domain evals, not in-domain ones.
