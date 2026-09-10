# Other Datasets

The non-`qa` dataset classes.

## `DocumentsDataset` (`data/documents.py`)
Streams **raw document chunks** — no Q/A. Used to pre-build a static memory bank for
`gen_large_mem` / corpus evals.
- `generator()` yields `(ids[B,chunk_size], masks[B,chunk_size], doc_ids[B])`.
- Tokenizes each doc into `chunk_size`-token chunks (≤ `max_chunks_per_doc`), buffers to
  `batch_size`, stops at `max_docs`.
- **`doc_id`**: the dataset's `id` column when present (from a `*_with_ids` prep script), else a
  running counter — all chunks of a doc share it, enabling document-level retrieval accuracy.
- `normalizer_type=flashrag`/`hotpotqa` expands each row's selected passages / context
  paragraphs into individual docs. Config: `configs/dataset/documents.yaml`.

## `NovelHopQADataset` (`data/novelhopqa.py`)
Long-context multi-hop QA. Subclasses `QADataset` (reuses its pipeline + resume) but builds
`self.dataset` itself: joins QA rows (`abhaygupta1266/novelhopqa`) with book texts
(`{HF_USERNAME}/novelhopqa-books`) on the `book` title → `pos_doc` = the **full book text**.
`split` selects hop depth (`hop_1`…`hop_4`). Books uploaded by
`data/utils/prepare_novelhopqa.py`.

## `DocCopyDataset` (`data/doc_copy.py`)
**Self-contained, removable grounding experiment.** Target is the **positive document**
(reproduce it), not the answer — forcing the memory read to source the doc's high-information
tokens. Overrides only the item transform; emits identical batch keys, so the trainer/loss are
unchanged. Caveat: partial context leak (teacher-forced doc lets connectives be induction-copied
from the visible prefix — watch `mem_pos_weight_mass` / `mem_top1_weight`). Removal steps are
documented in the file's header.

## Prompt templates (`data/prompts/*.txt`)
Prefix templates referenced by `prompt_path` / `teacher_prompt_path` / `student_prompt_path`:
`default`, `doc_completion`, `summarization_qa`, `similarity_recall`, `doc_copy`,
`novelhopqa`, `musique`, and `teacher_*` variants for distillation.
