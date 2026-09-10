"""
Tests for the MuSiQue → SFT conversion (datagen/musique/*).

Covers:
  1. Stage-1 process_row — pos/neg split, hop_type, decomposition, drop rules
  2. hop_grounding_score — the reasoning-fidelity metric
  3. format_paragraphs — deterministic, and gold is actually scattered
  4. Token budget — the think+answer ceiling must leave a row that survives
     _StreamingQAFilter (the silent-drop failure mode). Checked for BOTH shipped
     pairings: 400/seq_len 512 and 950/seq_len 1024. The data is generated at 950;
     the 512 dataset is derived by filtering, so both must hold.
  5. Normalizer round-trip — configs/dataset/sources/musique_sft.yaml's
     field_map actually produces the schema QADataset expects

Tests 4 and 5 need the Qwen3-4B tokenizer; the rest are pure logic.

    uv run python tests/test_musique_sft.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datagen.musique.prepare_musique_sft_base import process_row
from datagen.musique.generate_musique_sft import (
    MAX_THINK_ANS_TOKENS,
    format_paragraphs,
    hop_grounding_score,
)
from data.utils import build_prefix_text, make_normalizer

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))


def make_musique_row(n_supporting=2, n_distractor=18, answerable=True):
    paragraphs = [
        {"idx": i, "title": f"Title {i}", "paragraph_text": f"Body text for paragraph {i}. " * 20,
         "is_supporting": i < n_supporting}
        for i in range(n_supporting + n_distractor)
    ]
    return {
        "id": f"{n_supporting}hop__111_222",
        "question": "When was the institute that owned The Collegian founded?",
        "answer": "1960",
        "answer_aliases": ["1960s"],
        "answerable": answerable,
        "paragraphs": paragraphs,
        "question_decomposition": [
            {"id": i, "question": f"sub-question {i}", "answer": f"intermediate{i}",
             "paragraph_support_idx": i}
            for i in range(n_supporting)
        ],
    }


# ---------------------------------------------------------------------------
def test_process_row():
    print("\n[1] stage-1 process_row")
    rec = process_row(make_musique_row(2, 18))
    check("splits pos/neg by is_supporting", len(rec["pos_doc"]) == 2 and len(rec["neg_doc"]) == 18,
          f"got {len(rec['pos_doc'])}/{len(rec['neg_doc'])}")
    check("formats title in bold", rec["pos_doc"][0].startswith("**Title 0**\n"))
    check("hop_type parsed from id", rec["hop_type"] == "2hop", rec["hop_type"])
    check("aliases preserved", rec["answer_aliases"] == ["1960s"])
    decomp = json.loads(rec["decomposition"])
    check("decomposition is valid JSON with all steps", len(decomp) == 2)
    check("decomposition keeps intermediate answers", decomp[0]["answer"] == "intermediate0")

    check("drops unanswerable", process_row(make_musique_row(answerable=False)) is None)
    check("drops rows with no gold paragraph", process_row(make_musique_row(n_supporting=0)) is None)

    rec4 = process_row(make_musique_row(4, 16))
    check("4-hop handled", rec4["hop_type"] == "4hop" and len(rec4["pos_doc"]) == 4)


def test_hop_grounding():
    print("\n[2] hop_grounding_score")
    decomp = json.dumps([{"answer": "Houston Baptist University"}, {"answer": "1960"}])
    check("all hops present → 1.0",
          hop_grounding_score("owned by Houston Baptist University, founded 1960", decomp) == 1.0)
    check("half present → 0.5",
          hop_grounding_score("It was founded in 1960.", decomp) == 0.5)
    check("none present → 0.0", hop_grounding_score("no idea", decomp) == 0.0)
    check("case-insensitive",
          hop_grounding_score("houston baptist university ... 1960", decomp) == 1.0)
    check("empty decomposition → 0.0", hop_grounding_score("anything", "[]") == 0.0)
    # "" is a substring of everything; blank gold answers must not score as grounded.
    blank = json.dumps([{"answer": ""}, {"answer": "  "}])
    check("all-blank intermediate answers → 0.0", hop_grounding_score("unrelated", blank) == 0.0)
    mixed = json.dumps([{"answer": ""}, {"answer": "1960"}])
    check("blank steps ignored, not counted as hits",
          hop_grounding_score("founded in 1960", mixed) == 1.0)
    check("blank steps ignored when genuinely absent",
          hop_grounding_score("no year here", mixed) == 0.0)
    check("malformed JSON → 0.0", hop_grounding_score("anything", "not json") == 0.0)


def test_format_paragraphs():
    print("\n[3] format_paragraphs")
    pos = [f"POS{i}" for i in range(3)]
    neg = [f"NEG{i}" for i in range(17)]
    a = format_paragraphs(pos, neg, "2hop__1_2")
    b = format_paragraphs(pos, neg, "2hop__1_2")
    c = format_paragraphs(pos, neg, "2hop__9_9")
    check("deterministic for a given id", a == b)
    check("differs across ids", a != c)
    check("all 20 docs present", all(d in a for d in pos + neg))
    # Numbering must NOT appear: it made the CoT cite "[Document 3]" in 99.6% of rows,
    # references that are meaningless once the docs move into the memory bank.
    check("no document numbering", re.search(r"[Dd]ocument\s*\d", a) is None,
          "found a document index in the formatted context")
    # Gold must not be clustered at the front, or the CoT learns position not content.
    slots = a.split("\n\n")
    check("one slot per doc", len(slots) == 20, f"got {len(slots)}")
    gold_slots = [i for i, s in enumerate(slots) if s in pos]
    check("gold is scattered, not all in the first 5 slots",
          not all(i < 5 for i in gold_slots), f"gold at {gold_slots}")


def test_token_budget(tok, seq_len=512, budget=None):
    budget = budget if budget is not None else MAX_THINK_ANS_TOKENS
    print(f"\n[4] token budget {budget} vs seq_len={seq_len}")
    SEQ_LEN = seq_len
    # Worst case: a long question (MuSiQue p95 ≈ 162 chars) and a think+answer
    # block filling the entire generation budget.
    question = "When was the institution that owned the publication founded by the person " \
               "who also established the rival newspaper in that same state first opened?"
    filler = tok.decode(tok.encode("The relevant document states a fact. " * 400)[:budget - 8])
    think, answer = filler, "1960"

    spliced = f"<think>\n\n{think}</think>\n\n{answer}"
    prefix = build_prefix_text(tok, question, None, chat_template=True, has_thinking=True)
    full = prefix + spliced + (tok.eos_token or "<|im_end|>")
    n = len(tok(full, truncation=False, return_tensors="np")["input_ids"][0])

    check(f"worst-case row fits seq_len={seq_len} (got {n})", n <= SEQ_LEN, f"{n} > {SEQ_LEN}")
    print(f"       prefix={len(tok.encode(prefix))} tok, think+answer budget={budget}, total={n}")

    # Confirm the headroom is real: a budget equal to seq_len would NOT have been safe.
    over_filler = tok.decode(tok.encode("The relevant document states a fact. " * 500)[:seq_len])
    over_full = prefix + f"<think>\n\n{over_filler}</think>\n\n{answer}" + (tok.eos_token or "")
    n_over = len(tok(over_full, truncation=False, return_tensors="np")["input_ids"][0])
    check(f"a budget of {seq_len} would overflow seq_len={seq_len} (got {n_over}) — justifies {budget}",
          n_over > SEQ_LEN)


def test_normalizer_roundtrip(tok):
    print("\n[5] normalizer round-trip with musique_sft field_map")
    # Exactly what configs/dataset/sources/musique_sft.yaml declares.
    normalizer = make_normalizer(
        field_map=None,   # musique_sft.yaml declares no field_map: train on gold `answer`
        think_field="think",
        doc_separator=None,
        neg_score_threshold=None,
        min_neg_docs=2,
    )
    stage2_row = {
        "id": "2hop__1_2",
        "question": "When was it founded?",
        "answer": "1960",
        # Deliberately the verbose, citation-laden shape the model actually produced,
        # so the assertion below fails if the target ever reverts to this column.
        "generated_answer": "Per Document 3, HBU was founded in 1960.",
        "think": "The Collegian is owned by HBU. HBU was founded in 1960.",
        "pos_doc": ["**A**\nalpha", "**B**\nbravo"],
        "neg_doc": [f"**N{i}**\nnoise" for i in range(18)],
        "hop_type": "2hop",
        "hop_grounding": 1.0,
    }
    out = normalizer(stage2_row)

    check("pos_doc stays a list of 2", isinstance(out["pos_doc"], list) and len(out["pos_doc"]) == 2)
    check("neg_doc stays a list of 18", isinstance(out["neg_doc"], list) and len(out["neg_doc"]) == 18)
    check("question passthrough", out["question"] == "When was it founded?")
    check("target is the GOLD answer, not the paraphrase",
          out["answer"].endswith("1960") and "Document" not in out["answer"])
    check("think spliced into answer", out["answer"].startswith("<think>\n\n"))
    check("think block closed", "</think>\n\n" in out["answer"])
    check("_min_neg_docs attached", out["_min_neg_docs"] == 2)
    check("_ce_enable on", out["_ce_enable"] == 1.0)

    # The filter's answer-leak rule must not fire on entity answers.
    leaks = out["answer"].lower() in out["question"].lower()
    check("answer does not leak into question", not leaks)


def main():
    print("=" * 70)
    print("MuSiQue SFT conversion tests")
    print("=" * 70)

    test_process_row()
    test_hop_grounding()
    test_format_paragraphs()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception as e:
        print(f"\n[4][5] SKIPPED — could not load Qwen3-4B tokenizer: {e}")
        tok = None

    if tok is not None:
        # Both shipped pairings. The data is generated at 950 (seq_len 1024); the 512 dataset
        # is derived by filtering, so both must be known-safe against _StreamingQAFilter.
        test_token_budget(tok, seq_len=512, budget=400)
        test_token_budget(tok, seq_len=1024, budget=950)
        test_normalizer_roundtrip(tok)

    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for f in FAILED:
            print(f"  FAILED: {f}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
