import json, urllib.request, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
b = json.loads(urllib.request.urlopen("https://raw.githubusercontent.com/kbressem/LongHealth/main/data/benchmark_v5.json").read().decode())
tot=0; per=[]; words=0
for pid,p in b.items():
    for k,t in p["texts"].items():
        n=len(tok(t)["input_ids"]); tot+=n; per.append(n); words+=len(t.split())
per.sort()
print(f"EXACT corpus tokens: {tot:,}   docs {len(per)}")
print(f"  tokens/doc  min {per[0]}  med {per[len(per)//2]}  max {per[-1]}")
print(f"  tokens/word ratio: {tot/words:.2f}")
print(f"  chunks@256: {sum((n+255)//256 for n in per)}")
OPT=["answer_a","answer_b","answer_c","answer_d","answer_e"]
c=collections.Counter()
for p in b.values():
    for q in p["questions"]:
        hit=None
        for L,k in zip("ABCDE",OPT):
            if q.get(k)==q["correct"]: hit=L; break
        c[hit or "?"]+=1
print("correct-option distribution:", dict(sorted(c.items())))
print(f"  option-E questions the D-capped filter still drops: {c['E']} of 400")
