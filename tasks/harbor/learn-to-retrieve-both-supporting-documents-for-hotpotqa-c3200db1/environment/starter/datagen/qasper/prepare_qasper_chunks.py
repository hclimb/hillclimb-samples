#!/usr/bin/env python3
"""QASPER -> chunk-doc corpus + evidence-gold QA repos, msmarco-hybrid protocol compatible.

Design (2026-07-22, for the MSA sweep's QASPER arm): QASPER papers are ~5.4k tokens — far
past the eval's 256-token doc slot — so each paper is packed into PARAGRAPH-ALIGNED chunks
of <=256 Qwen3-4B tokens and every chunk becomes its own corpus doc (one chunk per doc =
gather_bank compatible). Gold ids for a question are the chunks holding its evidence
paragraphs — paragraph-aligned packing makes that mapping exact, no substring fuzz.
Questions kept: answerable, with >=1 mappable evidence paragraph; answer text = free-form
if present, else joined extractive spans, else Yes/No.

Corpus = train+dev papers (bigger haystack; ~25k chunk-docs, trimmed to target_docs by
inject_query_gold at eval time). QA = dev questions only. Schema mirrors the msa-*-with-ids
repos ({question, answer, pos_doc, pos_doc_ids}, pos_doc packed with "\n||||\n").

    uv run python datagen/qasper/prepare_qasper_chunks.py [--dry-run] [--max-questions 400]
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

import dotenv

dotenv.load_dotenv()

from datasets import Dataset
from transformers import AutoTokenizer

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
SEP = "\n||||\n"
_USER = os.environ.get("HF_USERNAME", "")


def norm(s):
    return " ".join(str(s).split())


def paper_paragraphs(rec):
    """Ordered (text) paragraphs: title, abstract, then section names + paragraphs."""
    out = []
    if rec.get("title"):
        out.append(rec["title"])
    if rec.get("abstract"):
        out.append(rec["abstract"])
    for sec in rec.get("full_text") or []:
        if sec.get("section_name"):
            out.append(sec["section_name"])
        for p in sec.get("paragraphs") or []:
            if p and p.strip():
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--max-questions", type=int, default=400)
    ap.add_argument("--docs-repo", default=f"{_USER}/qasper-chunks-docs-with-ids")
    ap.add_argument("--qa-repo", default=f"{_USER}/qasper-chunks-qa-with-ids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not _USER:
        sys.exit("set HF_USERNAME in .env")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    path = os.path.expanduser("~/qasper.tgz")
    if not os.path.exists(path):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    train, dev = {}, {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".json"):
                obj = json.load(tf.extractfile(m))
                (dev if "dev" in m.name else train).update(obj)
    print(f"papers: train {len(train)}, dev {len(dev)}")

    # Chunk all papers (dev first so its golds survive any corpus trim), track
    # (paper_id, para_norm) -> [chunk ids] for evidence mapping.
    docs, para_to_chunks = [], {}
    for split in (dev, train):
        for pid, rec in split.items():
            paras = paper_paragraphs(rec)
            cur, cur_tokens, cur_paras = [], 0, []
            def flush():
                nonlocal cur, cur_tokens, cur_paras
                if not cur:
                    return
                cid = len(docs)
                docs.append({"text": "\n".join(cur), "id": cid})
                for pn in cur_paras:
                    para_to_chunks.setdefault((pid, pn), []).append(cid)
                cur, cur_tokens, cur_paras = [], 0, []
            for p in paras:
                n = len(tok(p, add_special_tokens=False)["input_ids"])
                if n > args.chunk_tokens:            # oversized paragraph: hard-split
                    flush()
                    ids = tok(p, add_special_tokens=False)["input_ids"]
                    for i in range(0, len(ids), args.chunk_tokens):
                        piece = tok.decode(ids[i:i + args.chunk_tokens])
                        cid = len(docs)
                        docs.append({"text": piece, "id": cid})
                        para_to_chunks.setdefault((pid, norm(p)), []).append(cid)
                    continue
                if cur_tokens + n > args.chunk_tokens:
                    flush()
                cur.append(p)
                cur_tokens += n
                cur_paras.append(norm(p))
            flush()
    print(f"chunk-docs: {len(docs)}")

    # Dev questions with mappable evidence.
    id_to_text = {d["id"]: d["text"] for d in docs}
    rows, skipped = [], {"unanswerable": 0, "no_answer_text": 0, "no_evidence": 0}
    for pid, rec in dev.items():
        for qa in rec.get("qas") or []:
            if len(rows) >= args.max_questions:
                break
            ans = next((a["answer"] for a in (qa.get("answers") or [])
                        if a.get("answer") and not a["answer"].get("unanswerable")), None)
            if ans is None:
                skipped["unanswerable"] += 1
                continue
            if ans.get("free_form_answer"):
                answer = ans["free_form_answer"]
            elif ans.get("extractive_spans"):
                answer = ", ".join(ans["extractive_spans"])
            elif ans.get("yes_no") is not None:
                answer = "Yes" if ans["yes_no"] else "No"
            else:
                skipped["no_answer_text"] += 1
                continue
            evid = [e for e in (ans.get("evidence") or []) if e and not e.startswith("FLOAT SELECTED")]
            gold = sorted({c for e in evid for c in para_to_chunks.get((pid, norm(e)), [])})
            if not gold:
                skipped["no_evidence"] += 1
                continue
            rows.append({
                "question": qa["question"],
                "answer": answer,
                "pos_doc": SEP.join(id_to_text[c] for c in gold),
                "pos_doc_ids": gold,
            })
    print(f"qa rows: {len(rows)}  skipped: {skipped}")
    gpq = [len(r["pos_doc_ids"]) for r in rows]
    print(f"golds/question: mean {sum(gpq)/len(gpq):.2f}  max {max(gpq)}")

    if args.dry_run:
        print("dry run — not uploading")
        return
    Dataset.from_list(docs).push_to_hub(args.docs_repo, token=token, private=False)
    Dataset.from_list(rows).push_to_hub(args.qa_repo, token=token, private=False)
    print(f"Done -> {args.docs_repo} ({len(docs)}), {args.qa_repo} ({len(rows)})")


if __name__ == "__main__":
    main()
