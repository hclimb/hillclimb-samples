"""Regression test: evals/metrics/llm_judge.py::_parse_score must take the LAST <judgement>
tag, not the first. The judge runs with thinking=True, and sometimes writes a tentative
<judgement> tag while reasoning through a hypothetical ("if the answer were X, that would be
a match...") before landing on a different final verdict after </think>. `.search()` (the
pre-fix behavior) silently grabs whichever tag comes first in the string.

Found via a real sample from the hotpotqa hybrid CE-only eval (step 42800): judge output
contained the tags ['match', 'not match', 'not match'] verbatim -- the model's real, final
verdict was "not match", but the stored `llm_judge_accuracy` was 1.0. Confirmed the same
pattern (5/128 samples, always inflating the score) in the musique hybrid pf32/lr_masked eval
(step 100000) too.

Run: uv run python tests/test_llm_judge_parse_score.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics.llm_judge import _parse_score

# Real judge output, trimmed, from the hotpotqa hybrid CE-only eval's "Silverstone" sample --
# the model second-guesses itself mid-thought ("Oh, maybe... the answer would be a match")
# before its actual final tag says otherwise.
REAL_REGRESSION_TEXT = """<think>
Okay, let's see. The generated answer isn't provided here... wait, maybe there's a mistake.
Hmm, if the generated answer said BBC Formula One, this would be <judgement>match</judgement>
but let me reconsider -- the generated answer is actually empty, so it can't cover the point.
</think>

Therefore, the verdict is not match.
<judgement>not match</judgement>
"""


def run_cases():
    cases = [
        ("single 'match' tag", "<judgement>match</judgement>", 1.0),
        ("single 'not match' tag", "<judgement>not match</judgement>", 0.0),
        ("multi-tag: first=match, last=not match (the bug case)",
         "<judgement>match</judgement> ... <judgement>not match</judgement>", 0.0),
        ("multi-tag: first=not match, last=match",
         "<judgement>not match</judgement> ... <judgement>match</judgement>", 1.0),
        ("real regression text (Silverstone sample)", REAL_REGRESSION_TEXT, 0.0),
        ("no tags, fallback 'not match' text", "I think this is not match overall.", 0.0),
        ("no tags, fallback 'match' text", "I think this is a match overall.", 1.0),
        ("no tags, no match/not-match text at all", "I have no idea what to say here.", 0.0),
    ]
    all_ok = True
    for name, text, expected in cases:
        got = _parse_score(text)
        ok = got == expected
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected={expected} got={got}")
    return all_ok


def main():
    all_ok = run_cases()
    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
