"""Count doc tokens encoded into memory per eval, at the eval cap.

Replicates the eval's corpus loading (data/documents.py: max_docs=4000,
max_chunks_per_doc=3, chunk_size=256 -> caps at 12,000 chunks) and sums valid
(non-pad) tokens. Same docs + chunk config for both MSA and membed, so this is
one number per dataset. CPU-only (tokenization), no TPU needed.

    HF_USERNAME=ragrawal36 python scripts/embed/count_doc_tokens.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transformers import AutoTokenizer
from data.documents import DocumentsDataset

HF_USER = os.environ["HF_USERNAME"]
TOK = os.environ.get("BENCH_TOKENIZER", "EverMind-AI/MSA-4B")
MAX_DOCS, MAX_CPD, CHUNK = 4000, 3, 256

DATASETS = ["popqa", "natural_questions", "narrativeqa", "2wikimultihopqa",
            "hotpotqa", "musique", "dureader", "msmarco_v1", "triviaqa_10m"]
# hf slug uses dashes
HF_SLUG = {d: d.replace("_", "-") for d in DATASETS}

tok = AutoTokenizer.from_pretrained(TOK)
print(f"tokenizer={TOK}  cap: max_docs={MAX_DOCS} x {MAX_CPD} chunks x {CHUNK} = "
      f"{MAX_DOCS*MAX_CPD} chunks max\n", flush=True)

print(f"{'dataset':20s} {'docs':>7s} {'chunks':>8s} {'valid_tokens':>14s}")
results = {}
for d in DATASETS:
    ds = DocumentsDataset(
        tokenizer=tok, hf_name=f"{HF_USER}/msa-{HF_SLUG[d]}-docs-with-ids",
        split="train", column="text",
        batch_size=256, max_docs=MAX_DOCS, chunk_size=CHUNK, max_chunks_per_doc=MAX_CPD,
    )
    n_tokens = n_chunks = 0
    doc_ids = set()
    for ids_b, masks_b, docid_b in ds.generator():
        n_tokens += int(masks_b.sum())
        n_chunks += ids_b.shape[0]
        doc_ids.update(docid_b.tolist())
    results[d] = {"docs": len(doc_ids), "chunks": n_chunks, "valid_tokens": n_tokens}
    print(f"{d:20s} {len(doc_ids):>7d} {n_chunks:>8d} {n_tokens:>14,d}", flush=True)

import json
print("\nJSON:", json.dumps(results))
print("COUNT_DONE", flush=True)
