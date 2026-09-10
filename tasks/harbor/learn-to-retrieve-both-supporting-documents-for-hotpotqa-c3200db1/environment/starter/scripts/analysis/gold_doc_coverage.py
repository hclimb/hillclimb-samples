"""Measure gold-doc coverage under the eval's corpus cap.

For each dataset: build the set of doc IDs actually encoded into memory at the eval
cap (max_docs=4000 -> 12,000 chunks, via data/documents.py), then check the first
NQ evaluated questions' pos_doc_ids against it.

  - any_cov: question has >=1 gold doc in the encoded corpus (answer plausibly present)
  - all_cov: ALL of the question's gold docs are present (full evidence; matters for multi-hop)

This bounds how much of the low absolute scores is "answer wasn't even in memory"
vs "model failed". CPU-only.

    HF_USERNAME=ragrawal36 python scripts/embed/gold_doc_coverage.py [NQ]
"""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from transformers import AutoTokenizer
from datasets import load_dataset
from data.documents import DocumentsDataset

HF_USER = os.environ["HF_USERNAME"]
TOK = os.environ.get("BENCH_TOKENIZER", "EverMind-AI/MSA-4B")
NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
MAX_DOCS, MAX_CPD, CHUNK = 4000, 3, 256

DATASETS = ["popqa", "natural_questions", "narrativeqa", "2wikimultihopqa",
            "hotpotqa", "musique", "dureader", "msmarco_v1", "triviaqa_10m"]
SLUG = {d: d.replace("_", "-") for d in DATASETS}

tok = AutoTokenizer.from_pretrained(TOK)
hf_token = os.environ.get("HF_TOKEN")

print(f"NQ={NQ} questions/dataset, cap=12,000 chunks\n")
print(f"{'dataset':20s} {'enc_docs':>9s} {'any_cov':>8s} {'all_cov':>8s}")
res = {}
for d in DATASETS:
    ds = DocumentsDataset(tokenizer=tok, hf_name=f"{HF_USER}/msa-{SLUG[d]}-docs-with-ids",
                          split="train", column="text", batch_size=512,
                          max_docs=MAX_DOCS, chunk_size=CHUNK, max_chunks_per_doc=MAX_CPD)
    enc = set()
    for _, _, docid_b in ds.generator():
        enc.update(int(x) for x in docid_b.tolist())

    qa = load_dataset(f"{HF_USER}/msa-{SLUG[d]}-qa-with-ids", split="train",
                      streaming=True, token=hf_token)
    any_hit = all_hit = n = 0
    for row in qa:
        if n >= NQ:
            break
        pids = row.get("pos_doc_ids")
        if isinstance(pids, str):
            try: pids = json.loads(pids)
            except Exception: pids = []
        pids = [int(x) for x in (pids or [])]
        n += 1
        if not pids:
            continue
        present = [p in enc for p in pids]
        any_hit += int(any(present))
        all_hit += int(all(present))
    res[d] = {"enc_docs": len(enc), "any_cov": any_hit / n if n else 0,
              "all_cov": all_hit / n if n else 0, "n": n}
    print(f"{d:20s} {len(enc):>9d} {res[d]['any_cov']:>8.3f} {res[d]['all_cov']:>8.3f}", flush=True)

print("\nJSON:", json.dumps(res))
print("COVERAGE_DONE", flush=True)
