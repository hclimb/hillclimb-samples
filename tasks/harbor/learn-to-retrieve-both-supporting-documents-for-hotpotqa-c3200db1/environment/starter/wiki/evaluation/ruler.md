# RULER Long-Context Benchmark

`evals/ruler.py` (`RULEREvaluator`) + the standalone `eval_ruler.py` Fire CLI. Tests
long-context ability across context lengths (4K–131K tokens) with synthetic tasks.

## Task types
| Task | What it tests |
|------|---------------|
| `niah_s` | Needle-in-a-haystack, single needle — retrieve one fact from long context |
| `niah_mk` | Multi-key needles |
| `niah_mv` | Multi-value (same key, several values) |
| `niah_mq` | Multi-query |
| `vt` | Value tracking — follow a value through a chain of updates |
| `cwe` | Common word extraction |
| `fwe` | Frequent word extraction (above a threshold) |
| `qa_squad` | QA over a long SQuAD passage |
| `qa_hotpot` | Multi-hop QA over a long HotpotQA doc |

## Running
```bash
# Via the eval set (two-process path):
uv run eval.py checkpoint_dir=… '~eval_set@evals=pretraining' '+eval_set@evals=ruler'
# Or the standalone Fire CLI:
uv run eval_ruler.py --checkpoint_dir=… --tasks=niah_s,niah_mk,vt \
  --context_lengths=4096,16384,65536 --num_samples=100
```
Config: `configs/eval/ruler.yaml`, `configs/eval/niah.yaml`, `configs/eval_set/ruler.yaml`.
Override inline: `'evals.ruler.eval.tasks=[niah_s,vt]'`,
`'evals.ruler.eval.context_lengths=[4096,32768]'`, `'evals.ruler.eval.num_samples=100'`.

## Note
The RULER path forces `main_model.mem_lookup_chunk_size=8192` (chunked retrieval) for the long
contexts, restoring it afterward — see [../architecture/sharded-retrieval.md](../architecture/sharded-retrieval.md).
