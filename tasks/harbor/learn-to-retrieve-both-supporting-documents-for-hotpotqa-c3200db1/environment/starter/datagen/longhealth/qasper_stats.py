import json, tarfile, io, urllib.request, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
p = os.path.expanduser("~/qasper.tgz")
if not os.path.exists(p):
    urllib.request.urlretrieve(URL, p)
papers = {}
with tarfile.open(p) as tf:
    for m in tf.getmembers():
        if m.name.endswith(".json"):
            papers.update(json.load(tf.extractfile(m)))
def paper_text(r):
    parts=[r.get("title") or "", r.get("abstract") or ""]
    for sec in r.get("full_text") or []:
        if sec.get("section_name"): parts.append(sec["section_name"])
        parts.extend(sec.get("paragraphs") or [])
    return "\n".join(x for x in parts if x)
tot=0; per=[]; nq=0; words=0
for pid,r in papers.items():
    t=paper_text(r); n=len(tok(t)["input_ids"])
    tot+=n; per.append(n); words+=len(t.split()); nq+=len(r.get("qas") or [])
per.sort()
print(f"QASPER papers: {len(per):,}   questions: {nq:,}")
print(f"corpus tokens: {tot:,}")
print(f"tokens/paper  min {per[0]:,}  med {per[len(per)//2]:,}  mean {tot//len(per):,}  p90 {per[int(.9*len(per))]:,}  max {per[-1]:,}")
print(f"tokens/word ratio: {tot/words:.2f}")
print(f"whole-corpus bank: {tot:,} slots")
print(f"RAG@5 whole-paper prompt: {5*(tot//len(per)):,} tokens")
