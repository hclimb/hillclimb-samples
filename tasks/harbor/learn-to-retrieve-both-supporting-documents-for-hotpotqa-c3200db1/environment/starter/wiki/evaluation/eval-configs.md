# Eval Configs

Four layers compose an eval run. `eval.py` loads `configs/eval.yaml`; you pick a suite with an
**eval_set**.

## The layers
1. **`configs/eval.yaml`** — top level: `defaults: /eval_set@evals: pretraining`,
   `model: qwen3_mem_embed`; fields `checkpoint_dir`, `seed`, `tp_devices`, `use_wandb`,
   `aux_losses` (override training aux config, e.g. enable `doc_access_acc`).
2. **`configs/eval_set/<name>.yaml`** — a named group of tasks, composed via
   `/eval/tasks@<key>: <task>`. Each `<key>` becomes an entry in `cfg.evals`.
3. **`configs/eval/tasks/<name>.yaml`** — one task: composes the eval **type**
   (`/eval@eval: generation_large_mem`), a `dataset` (+ source), an `eval.doc_dataset` (for
   large-mem), and sets `eval.type`, `num_samples`, `output_file`, `max_new_tokens`, `metrics:`
   (e.g. `llm_judge_accuracy`), plus a `rag:` block for [rag-eval.md](rag-eval.md).
4. **`configs/eval/<type>.yaml`** — the base config per evaluator type (`nll`, `generation`,
   `generation_embed`, `generation_large_mem`, `generation_base`, `ruler`, `niah`).

## Eval sets (`configs/eval_set/`)
`pretraining` (NLL + gen_embed per source) · `corpus_evals` · `multihop_evals` ·
`long_context_evals` · `msa_evals` (9 MSA tasks) · `ruler` · `ground_evals` ·
`embed_*`/`hotpotqa_corpus`/`msmarco_valce` · `hard_neg_think_c512` (msmarco/hotpotqa/musique
@512-doc corpus + scienceQA NLL, n=128; driven by an [eval box](eval-boxes.md)) ·
`none` (empty — for training with in-loop eval off).

### Corpus sizing: `max_docs` vs `target_docs`
For a `generation_large_mem` task with `inject_query_gold: true`, **size the corpus with
`target_docs` and leave `max_docs: null`.** `data/documents.py` stops the doc generator at
`max_docs`, so setting it caps the gold **scan** — any query whose gold doc sits past the cap is
silently absent from the bank, which is the opposite of what `inject_query_gold` is for. With
`max_docs: null` the scan sees the whole corpus, then `target_docs` trims to
`all gold chunks + distractors` (`gen_large_mem.py:398-416`, which prints
`corpus (scanned N): X gold + Y distractor chunks = Z total`). `target_docs` defaults to `max_docs`
when absent. With `max_chunks_per_doc: 1`, N docs == N chunks.

## Swapping the eval set
`eval.yaml` loads `pretraining` by default. Swap with the two-line Hydra group override (quotes
required — `~`/`+` are shell-special):
```bash
'~eval_set@evals=pretraining'    # remove the default
'+eval_set@evals=corpus_evals'   # add another
'+eval_set@evals2=multihop_evals'  # run multiple sets at once
```

## Overriding a task
Dotted keys reach into a task: `eval.gen_large_mem_msa_hotpotqa.eval.num_samples=64`,
`eval.<key>.eval.lookup_chunk_size=4096`. The canonical run scripts (`eval_*.sh`,
`rag_*.sh`) live in [../infrastructure/experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md).

## Multiple-choice sources and the anti-leakage filter

`data/qa.py` drops rows whose answer text appears inside the question, to stop trivially leaked
answers reaching training. Multiple-choice datasets legitimately trip this — the options are *in*
the question — so the filter carves out answers carrying a letter prefix:

```python
if answer.lower() in question.lower():
    if not any(opt in answer for opt in ["A)", "B)", "C)", "D)"]):
        return False
```

Two consequences when adding a multiple-choice source:

- The dataset's `answer` field must be the **letter-prefixed** form (`"C) Vincristine"`), not the
  bare option text, or every row is dropped.
- The whitelist stops at `"D)"`, so **any question whose answer is option E or beyond is silently
  dropped** even with correct formatting. This bit LongHealth (45 of 400 questions); see the
  [implementation note](../implementations/2026-07-19-longhealth-eval-pipeline.md).

A source that trips either case fails silently: zero rows survive, generation runs `0/N`, and the
eval still exits 0 while reporting `llm_judge_accuracy: 0.0`. Check `generated_count` before
believing any eval result.
