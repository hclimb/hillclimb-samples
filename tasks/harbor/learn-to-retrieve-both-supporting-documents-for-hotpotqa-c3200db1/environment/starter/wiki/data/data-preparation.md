# Data Preparation

One-time scripts in `data/utils/` that build/upload processed corpora to HuggingFace under
`{HF_USERNAME}/...`. All are **idempotent** — they skip the upload if the target dataset
already exists. Run before the evals/training that consume them.

> **Getting training data onto a box** is a different job, covered by
> [epoch-sized-data.md](epoch-sized-data.md) (`data/download_hf_data.py`): download only the rows
> a run consumes, balanced across interleave sources. Required in practice — live-HF streaming
> livelocks on HF's 429 quota ([hf-rate-limits.md](hf-rate-limits.md)).

## Two script families
- **`prepare_*_docs.py`** — flatten a source corpus into a single `text` column (one doc per
  row) for `DocumentsDataset` / large-static-memory evals. Variants:
  `prepare_musique_docs.py`, `prepare_hotpotqa_docs.py`, `prepare_flashrag_docs.py`,
  `prepare_msa_docs.py`.
- **`prepare_*_qa_with_ids.py`** — augment a QA split with **`pos_doc_ids`** referencing the
  id-tagged corpus, so retrieval accuracy (`doc_access_acc`) can be computed. Must run **after**
  the matching `*_docs.py`. Variants: `prepare_musique_qa_with_ids.py`,
  `prepare_hotpotqa_qa_with_ids.py`, `prepare_flashrag_msmarco_qa_with_ids.py`,
  `prepare_msa_qa_with_ids.py`.

## The id-tagging flow (why it matters)
`*_docs.py` stamps a stable integer `id` on each corpus doc → `DocumentsDataset` reads it as
`doc_id`. `*_qa_with_ids.py` then tags each question's gold docs with those same ids as
`pos_doc_ids`. At eval, retrieved slot → doc_id is compared against `pos_doc_ids` for
document-level hit rate. Without this pass, `pos_doc_ids` are all −1 and accuracy isn't computed.

## MSA corpora
`data/msa_prep_common.py :: DATASET_SPECS` — the 9 MSA datasets, each `{doc_repo, qa_repo,
doc_separator: "\n||||\n"}` (`2wikimultihopqa`, `dureader`, `hotpotqa`, `msmarco_v1`, `musique`,
`narrativeqa`, `natural_questions`, `popqa`, `triviaqa_10m`). `split_pos_doc` splits packed docs
on the separator.

## Other
- `prepare_novelhopqa.py` — fetch full novel texts from Project Gutenberg → `novelhopqa-books`.
- `process_nemotron.py` — filter/format the Nemotron pretraining corpus.
- `persona_creation.py` — synthetic persona generation.

Usage/prereqs (HF creds, which eval needs which prep) live in
[../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md).
Pre-cache these datasets for offline runs with `scripts/misc/precache_hf.sh`.
