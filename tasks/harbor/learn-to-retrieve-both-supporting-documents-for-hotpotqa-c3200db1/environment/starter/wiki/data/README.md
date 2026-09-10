# Data

How documents + Q/A become the batches the model consumes, and how to add or prepare
datasets. Streaming pipeline built on HuggingFace `datasets` + Google `grain`.

## Factory — `data/__init__.py :: get_dataset(cfg, model)`
| `dataset.name` | Class | Use |
|----------------|-------|-----|
| `qa` | `QADataset` / `QADatasetIndexed` (`data/qa.py`) | **primary** — Q/A + supporting docs → memory; streaming (default) or indexed ArrayRecord (opt-in via `storage: arrayrecord`, see [storage-modes.md](storage-modes.md)) |
| `documents` | `DocumentsDataset` (`data/documents.py`) | raw doc chunks (build a static memory bank / corpus evals) |
| `novelhopqa` | `NovelHopQADataset` (`data/novelhopqa.py`) | long-context multi-hop QA over full novels |
| `doc_copy` | `DocCopyDataset` (`data/doc_copy.py`) | removable grounding experiment (target = the document) |

## Flow
`QADataset` loads/normalizes HF sources → a `grain` pipeline **source → filter → transform →
mp_prefetch → batch** → `generator()` yields `(x, masks)` per step. The trainer's
`process_train_pairs` turns that into `(inputs, targets, input_masks, loss_masks, ce_enable)`
and calls `model.forward`.

## Pages
- [batch-format.md](batch-format.md) — exactly what `generator()` yields (the model/trainer contract)
- [qa-dataset.md](qa-dataset.md) — the primary dataset: pipeline, tokenization/masking, modes, resume
- [storage-modes.md](storage-modes.md) — streaming vs. `arrayrecord` (indexed) — preprocess step, config keys, O(1) resume tradeoffs
- [indexed-loader-resume.md](indexed-loader-resume.md) — warm-start from a ckpt + mid-run rescue: how to synthesize `dataloader_state.json` for a given step, exact Grain state format, verification signs
- [normalizers.md](normalizers.md) — mapping arbitrary HF schemas → `question/answer/pos_doc/neg_doc`
- [other-datasets.md](other-datasets.md) — `documents`, `novelhopqa`, `doc_copy`
- [dataset-configs.md](dataset-configs.md) — `configs/dataset/*.yaml` + source configs
- [hf-rate-limits.md](hf-rate-limits.md) — **read before streaming live-HF**: the per-account 429 quota, why `num_workers × sources` blows it, and the rebuild-on-429 livelock that keeps a run at step 0 forever
- [epoch-sized-data.md](epoch-sized-data.md) — `data/download_hf_data.py`: download only the rows a run consumes, balanced across sources (a 100k-step epoch in ~15 GB vs 75 GB for the full mix)
- [data-preparation.md](data-preparation.md) — `data/utils/` one-time corpus prep + id-tagging

## Key fields (`configs/dataset/qa.yaml`)
`seq_len` (Q/A length) · `doc_chunk_seq_len` (per-chunk length) · `num_chunks_per_doc` ·
`batch_size` · `provide_docs` · `mask_prefix` · `chat_template` · `distill` · `split` ·
`num_workers` · `shuffle` · `hf_name`/`sources`.
