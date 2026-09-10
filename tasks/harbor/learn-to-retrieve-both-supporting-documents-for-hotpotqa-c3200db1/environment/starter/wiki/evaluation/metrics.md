# Metrics

Metrics run in the **parent process** after the JAX worker exits (so vLLM can use the freed
TPU/GPU). `evals/metrics/__init__.py :: run_metrics(samples, metrics_cfg)` → annotated samples +
`{metric: mean}`. Registered in `METRICS`; a task enables them under its `metrics:` block.

> **These are eval-time only.** `run_metrics` is called from `evals/shared.py::run_metrics_pipeline`
> (i.e. `eval.py`), never by the trainer — `trainer.py::_run_evals` keeps only
> `evaluator.evaluate()`'s `inference_metrics`. A task's `metrics:` block is **inert in-loop**: a
> training run pointed at a task listing `llm_judge_accuracy` gets no judge number and no error.
> Score a training run from a box instead — [eval-boxes.md](eval-boxes.md).

## Metrics the *evaluators* return (not in `METRICS`)
Distinct from the above: these come back in `inference_metrics` from the JAX worker, so they DO
work in-loop (subject to the `val/*` logging gate noted in [eval-boxes.md](eval-boxes.md)).
- `nll` (`NLLEvaluator`), `doc_access_acc`, `doc_hit_rate`, `doc_token_hit_rate`.
- **`mem_pos_weight_mass`** — softmax weight landing on positive-doc slots, the "right answer" read
  vs `doc_access_acc`'s "right doc" argmax count. Emitted by `NLLEvaluator` and
  `GenLargeMemEvaluator`, gated on the aux config enabling `mem_pos_weight_mass` (see
  [../training/auxiliary-losses.md](../training/auxiliary-losses.md)); force it at eval with
  `+aux_losses.mem_pos_weight_mass.enabled=true`. The two evaluators measure different things and
  are not comparable to each other:
  | Evaluator | Bank | Comparable to |
  |---|---|---|
  | `NLLEvaluator` | that batch's own pos+hard-neg docs | `train/mem_pos_weight_mass` (same in-batch setting) |
  | `gen_large_mem` | the shared corpus (e.g. 512 docs), on the generated answer span | other corpus evals at the same corpus size |

## `llm_judge_accuracy` / `llm_judge_score` (`metrics/llm_judge.py`)
Answer-equivalence judging via a vLLM server. The judge sees the **question (context only),
ground truth, and generated answer** and must emit `<judgement>match</judgement>` or
`<judgement>not match</judgement>` (a with-`document` template variant exists). System prompt:
*ground truth is the sole source of truth; match iff the generated answer covers its core
point(s); don't penalize extra info or paraphrase.*
- `llm_judge_accuracy` → fraction matched. `llm_judge_score` → graded variant.
- Judge model is a vLLM-served Qwen3 (e.g. `Qwen3-8B`/`32B`).
- ⚠️ `_parse_score` takes the **last** `<judgement>` tag in the judge's output, not the first —
  the judge runs with `thinking=True`, and sometimes writes a tentative tag while reasoning
  through a hypothetical before landing on a different final verdict after `</think>`. Taking
  the first tag (the pre-2026-08-17 behavior) silently inflated `llm_judge_accuracy` on every
  affected sample; see
  [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md).

## `lexical_grounding` (`metrics/lexical_grounding.py`)
Surface **faithfulness**, not correctness: fraction of an answer's *content words* literally
copied from the gold document. A token counts as grounded if it's in a ≥2-token verbatim span
or is a single non-stopword present in the doc; stopwords excluded; `None` if the answer has no
content words. A hallucinated number → ungrounded (lowers score); a correct paraphrase also
scores low — **report alongside `llm_judge_accuracy`**.

## The vLLM server (`evals/vllm.py`)
`VLLMInference` wraps an OpenAI-compatible client against a local vLLM server
(`http://localhost:8000/v1`) and manages its lifecycle (sync + async completion).
> Install caveat: the `vllm-tpu==0.12.0` HTTP judge needs pinned `fastapi`/`starlette` launched
> via `uv run --no-sync` (details in the `pyproject.toml` note).

## Adding a metric
Write `fn(results, **kwargs) → scores` (optionally `(scores, outputs)`), register it in
`METRICS`, reference it under a task's `metrics:` block. `None` scores are dropped from the mean.
