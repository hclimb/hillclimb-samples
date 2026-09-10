#!/usr/bin/env python3
"""LongHealth -> the repo's docs + QA dataset pair, for memory-layer evaluation.

LongHealth (Bressem et al., https://github.com/kbressem/LongHealth) is 20 fictional patient
records -- 133 clinical documents, ~110k words (~149k tokens) -- with 400 multiple-choice
questions in three categories: information extraction, negation, and sorting.

WHY THIS BENCHMARK. Its documents are long (median ~948 tokens, max ~5,900) while the corpus is
small (133 docs). That is exactly the "few, long documents" regime where a memory layer's flat
prefill can beat RAG's k*doc_len prefill -- see
wiki/experiments/2026-07-19-musique-corpus-scaling-and-throughput-pareto.md. MuSiQue was the
opposite shape (many 256-token docs) and RAG won there.

WHAT THIS SCRIPT DOES NOT CLAIM. No checkpoint here was trained on long documents, so accuracy on
LongHealth is out-of-distribution and is not a model-quality measurement. The throughput axis is
independent of that -- prefill/decode at a given shape does not care whether the answer is right.

Two outputs, mirroring the msa-musique-{docs,qa}-with-ids pair:
  <user>/longhealth-docs-with-ids : one row per clinical document, with a stable integer `id`
  <user>/longhealth-qa-with-ids   : one row per question, `pos_doc_ids` naming the gold documents

GOLD LABELS ARE EXACT. Each LongHealth question carries `answer_location`, naming the specific
text_N that contains the answer, so `pos_doc_ids` is ground truth rather than a heuristic. That
makes doc_hit_rate meaningful even when generation accuracy is not.

Usage (run on a box, where HF_TOKEN lives):
    uv run --no-sync python datagen/longhealth/prepare_longhealth.py --private
"""
import argparse
import json
import logging
import os
import urllib.request

from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SOURCE_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json"
OPTION_KEYS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]


def format_question(q: dict) -> str:
    """Multiple-choice prompt. Options are presented in their stored order -- LongHealth already
    fixes which letter holds the correct answer, and reshuffling here would silently desync the
    `correct` field from any letter-based scoring downstream."""
    lines = [q["question"], ""]
    for letter, key in zip("ABCDE", OPTION_KEYS):
        if q.get(key):
            lines.append(f"{letter}) {q[key]}")
    lines.append("")
    lines.append("Answer with the text of the correct option.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=os.environ.get("HF_USERNAME"))
    ap.add_argument("--docs-repo", default=None)
    ap.add_argument("--qa-repo", default=None)
    ap.add_argument("--private", action="store_true",
                    help="Push as private. Default for a third-party benchmark: do not re-host "
                         "someone else's dataset publicly under our account.")
    ap.add_argument("--dry-run", action="store_true", help="Build and report, do not push.")
    args = ap.parse_args()

    if not args.user:
        raise SystemExit("set HF_USERNAME or pass --user")
    docs_repo = args.docs_repo or f"{args.user}/longhealth-docs-with-ids"
    qa_repo = args.qa_repo or f"{args.user}/longhealth-qa-with-ids"

    log.info(f"fetching {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as r:
        bench = json.loads(r.read().decode())
    log.info(f"loaded {len(bench)} patients")

    # ---- documents: one row per clinical text, stable id shared with the QA side -------------
    doc_rows, key_to_id = [], {}
    for patient_id in sorted(bench):
        for text_key in sorted(bench[patient_id]["texts"]):
            text = bench[patient_id]["texts"][text_key]
            if not text or not text.strip():
                continue
            key_to_id[(patient_id, text_key)] = len(doc_rows)
            doc_rows.append({"pos_doc": text, "id": len(doc_rows),
                             "patient_id": patient_id, "text_key": text_key})

    # ---- questions: gold ids come from answer_location, so they are ground truth --------------
    qa_rows, missing_loc = [], 0
    for patient_id in sorted(bench):
        for q in bench[patient_id]["questions"]:
            loc = q.get("answer_location") or {}
            gold_keys = [k for k in loc if (patient_id, k) in key_to_id]
            if not gold_keys:
                # No usable location: keep the question (it is still a valid query) but record it,
                # because a silently unlabelled gold would corrupt doc_hit_rate rather than the
                # generation metric, and that is the harder error to notice later.
                missing_loc += 1
            gold_ids = [key_to_id[(patient_id, k)] for k in gold_keys]
            qa_rows.append({
                "question": format_question(q),
                "answer": q["correct"],
                "pos_doc": [bench[patient_id]["texts"][k] for k in gold_keys],
                "pos_doc_ids": gold_ids,
                "neg_doc": [],
                "patient_id": patient_id,
                "question_no": q.get("No", -1),
                "has_gold_location": bool(gold_keys),
            })

    words = sum(len(r["pos_doc"].split()) for r in doc_rows)
    log.info(f"docs {len(doc_rows)}  words {words:,}  est tokens {words * 1.35:,.0f}")
    log.info(f"questions {len(qa_rows)}  (without a usable answer_location: {missing_loc})")
    if missing_loc:
        log.warning(f"{missing_loc} question(s) have no gold document id -- doc_hit_rate is "
                    f"computed over the remaining {len(qa_rows) - missing_loc}")

    if args.dry_run:
        log.info("--dry-run: not pushing")
        return

    vis = "private" if args.private else "PUBLIC"
    log.info(f"pushing {docs_repo} ({vis})")
    Dataset.from_list(doc_rows).push_to_hub(docs_repo, private=args.private)
    log.info(f"pushing {qa_repo} ({vis})")
    Dataset.from_list(qa_rows).push_to_hub(qa_repo, private=args.private)
    log.info("done")


if __name__ == "__main__":
    main()
