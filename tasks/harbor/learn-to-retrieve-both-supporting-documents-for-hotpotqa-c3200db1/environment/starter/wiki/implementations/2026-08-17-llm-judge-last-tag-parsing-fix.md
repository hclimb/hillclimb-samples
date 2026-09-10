# `llm_judge_accuracy`'s `_parse_score` takes the last `<judgement>` tag, not the first

**Date:** 2026-08-17 · **Author:** claude (session with rohunagrawal) · **Status:** done ·
**Commit/PR:** none yet (working tree on `multihop-finetuning`).

## What changed

`evals/metrics/llm_judge.py::_parse_score` now scores off the **last** `<judgement>` tag found
in the judge's output (`JUDGEMENT_PATTERN.findall(text)[-1]`), instead of the **first**
(`JUDGEMENT_PATTERN.search(text)`). Also applied the identical fix to
`scripts/misc/judge_cot_accuracy.py::parse_verdict`, a new script added this session that copied
the same (buggy) pattern.

## Motivation & context

While building `scripts/misc/judge_cot_accuracy.py` (a script to re-judge whether a model's raw
chain-of-thought, not its extracted final answer, reaches the ground truth — see
[2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md](../experiments/2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md)'s
follow-up), rohunagrawal had flagged that the model's CoT often reasons to the correct answer
while its separately-extracted "Generated Answer" field is wrong or unrelated. Cross-checking a
flipped sample by hand (question about the 2013 6 Hours of Silverstone) turned up something
unrelated to that investigation: the ORIGINAL `llm_judge_accuracy_output` for that sample
contained the judge's actual, unambiguous final verdict — `<judgement>not match</judgement>` —
yet the stored `llm_judge_accuracy` was `1.0`.

Root cause: `llm_judge_accuracy` calls the judge with `thinking=True` (extended reasoning before
the final tag), and the judge occasionally writes a **tentative** `<judgement>` tag while
reasoning through a hypothetical ("if the generated answer had said X, this would be
`<judgement>match</judgement>`... but it doesn't, so...") before landing on a different, real
final verdict after `</think>`. `_parse_score`'s `JUDGEMENT_PATTERN.search(text)` returns the
**first** match in the string — i.e. exactly the tentative, hypothetical tag, not the model's
actual conclusion.

Measured impact, re-parsing already-stored judge output (no re-inference needed — pure text
reparsing) across the 4 hybrid-eval JSONs from this session's and the prior session's write-ups:

| eval | reported `llm_judge_accuracy` | corrected (last-tag-wins) | samples flipped |
|---|---|---|---|
| hotpotqa hybrid, ground4layer batched-iso, step 20200 | 0.4609 (59/128) | **0.4297** (55/128) | 4 |
| hotpotqa hybrid, pf32/indexed/lr_masked, step 100000 | 0.4609 (59/128) | **0.4297** (55/128) | 4 |
| hotpotqa hybrid, CE-only arm, step 42800 | 0.3828 (49/128) | **0.3438** (44/128) | 5 |
| musique hybrid, pf32/indexed/lr_masked, step 100000 | 0.3203 (41/128) | **0.2812** (36/128) | 5 |
| musique **plain** (non-hybrid), pf32/indexed/lr_masked, step 100000 | 0.1953 (25/128) | 0.1953 (25/128) | 0 |

Every affected sample's tag sequence was identical: `['match', 'not match', 'not match']` — i.e.
the bug **only ever inflates** the reported score (a tentative "match" is overridden by a real
"not match"); no case ran the other direction in any of the four affected files. The plain
(non-hybrid) musique eval had zero multi-tag samples and was unaffected — this may be
incidental to this particular sample set rather than a property of the plain-eval path; not
established either way.

## Options weighed

1. **Restrict parsing to the `content` field only** (the text after `</think>`, i.e. what
   `llm_judge_accuracy`'s own code already separates out as `content` vs. `reasoning` before
   concatenating them into the stored `_output` string). Rejected as the primary fix: the stored
   `..._output` field IS the concatenated `<think>{reasoning}</think>\n\n{content}` string (see
   `llm_judge_accuracy`'s own `explanations` construction), and a from-scratch reparse of
   historical files only has that concatenated string to work with — a fix that assumed access to
   `content` alone wouldn't apply to correcting the historical data.
2. **Take the last `<judgement>` tag in the full concatenated string** (chosen). Works identically
   whether parsing fresh judge output inside `llm_judge_accuracy` (where `content` naturally comes
   last, after `reasoning`, once concatenated) or reparsing an already-stored `_output` string
   after the fact. Simple, minimal, and the model's own convention already treats the *last*
   `<judgement>` tag as authoritative in every one of the observed cases (the final line, after
   `</think>`, is always the intended answer).

## How it was built & integrated

- `evals/metrics/llm_judge.py::_parse_score`: `JUDGEMENT_PATTERN.search(text)` →
  `JUDGEMENT_PATTERN.findall(text)`, score off `matches[-1]`. The no-tag fallback (substring
  check for "not match"/"match") is unchanged — it was never the affected path.
- `scripts/misc/judge_cot_accuracy.py::parse_verdict`: identical fix, since it copied the same
  regex pattern. Confirmed empirically this script's own judge calls (Qwen3-8B, `temperature=0`,
  `max_completion_tokens=512`, judging only the extracted CoT) never produced more than one
  `<judgement>` tag in 128 samples — the CoT-accuracy numbers already reported are unaffected by
  this bug — but the fix is applied for correctness going forward regardless.
- No change to `llm_judge_score`'s `_parse_score_05`: it already takes `digits[-1]` (the *last*
  standalone 0-5 digit), so it was never subject to this class of bug.
- No change to `scripts/misc/judge_grounding.py`: its own SUPPORTED/UNSUPPORTED parser already
  takes `toks[-1]`, same as `_parse_score_05`.

## Reference pages updated

- [evaluation/metrics.md](../evaluation/metrics.md): `llm_judge_accuracy` section now documents
  the last-tag-wins parsing rule and why it matters.

## Tests

`uv run python tests/test_llm_judge_parse_score.py` (run via the launcher on `rohun-v6e-8-0`, no
local accelerator) — includes the real regression case (a trimmed version of the actual
Silverstone-sample judge output) alongside synthetic single-tag/multi-tag/no-tag cases:

```
[PASS] single 'match' tag: expected=1.0 got=1.0
[PASS] single 'not match' tag: expected=0.0 got=0.0
[PASS] multi-tag: first=match, last=not match (the bug case): expected=0.0 got=0.0
[PASS] multi-tag: first=not match, last=match: expected=1.0 got=1.0
[PASS] real regression text (Silverstone sample): expected=0.0 got=0.0
[PASS] no tags, fallback 'not match' text: expected=0.0 got=0.0
[PASS] no tags, fallback 'match' text: expected=1.0 got=1.0
[PASS] no tags, no match/not-match text at all: expected=0.0 got=0.0

ALL TESTS PASSED
```

## Follow-ups & risks

- **Only the 5 files checked this session were reparsed.** Every other historical
  `wiki/experiments/*.md` write-up reporting `llm_judge_accuracy` from the hybrid-eval path
  (`gen_large_mem_msa_*_hybrid`) is a candidate for the same inflation and has not been rechecked
  — this fix only prevents new occurrences; it doesn't retroactively correct every prior number
  in the wiki. Reparsing is cheap (pure text processing over each result JSON's stored
  `llm_judge_accuracy_output`, no re-inference) if a specific historical number needs auditing.
- **The magnitude (4-5/128, ~3-4%) is only measured across these 4 files.** Whether it's a stable
  rate or varies by task/checkpoint/judge-model is unknown.
- **Why the plain musique eval had zero multi-tag samples** wasn't investigated — could be
  chance (small n=128) or a real difference in how that eval path prompts/constrains the judge.
