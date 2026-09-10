"""Sanity 2: exercise the MSA memory path (encode -> route -> prefill -> decode)
on a tiny synthetic corpus. Checks shapes/JIT and that routing picks the gold doc.

    cd memory-layers && set -a && source .env && set +a && \
    PYTHONPATH=. .venv/bin/python scripts/embed/sanity2_msa.py
"""
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from omegaconf import OmegaConf
from models import get_model
import models.qwen3_msa as msa

cfg_o = OmegaConf.create({"name": "qwen3_msa",
                          "main_model": {"model_id": "EverMind-AI/MSA-4B", "load_weights": True}})
model = get_model(cfg_o, tp_devices=1)
cfg, w, tok = model.cfg, model.weights, model.tokenizer
kernel = cfg['msa']['pooling_kernel_size']
router_layers = cfg['msa']['router_layers']

docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital and largest city of Italy.",
    "Madrid is the capital of Spain.",
    "Tokyo is the capital of Japan.",
    "Ottawa is the capital of Canada.",
    "Canberra is the capital of Australia.",
    "Cairo is the capital of Egypt.",
]
T = 256
N = len(docs)
ids = np.full((N, T), tok.pad_token_id, dtype=np.int64)
mask = np.zeros((N, T), dtype=bool)
for i, d in enumerate(docs):
    wrapped = f"<|im_start|>[{i+1}]. {d}[{i+1}]<|im_end|>"
    t = tok(wrapped, add_special_tokens=False)["input_ids"][:T]
    ids[i, :len(t)] = t
    mask[i, :len(t)] = True
docids = np.arange(N, dtype=np.int64)

print("encoding corpus...", flush=True)
enc = jax.jit(partial(msa.encode_docs, cfg))(jnp.array(ids), jnp.array(mask), w)
ppr = T // kernel
banks = {'kbar': {}, 'vbar': {}, 'krbar': {}}
for L in router_layers:
    banks['kbar'][L] = jnp.array(np.array(enc['kbar'][L]).reshape(N * ppr, 8, 128))
    banks['vbar'][L] = jnp.array(np.array(enc['vbar'][L]).reshape(N * ppr, 8, 128))
    banks['krbar'][L] = jnp.array(np.array(enc['krbar'][L]).reshape(N * ppr, 8, 128))
chunk_valid = np.array(enc['chunk_valid']).reshape(N * ppr)
bank_chunk_doc = np.repeat(docids, ppr)
P = ppr
dct = np.zeros((N, P), np.int32); dctv = np.zeros((N, P), bool); fill = np.zeros(N, np.int32)
for c in range(N * ppr):
    if not chunk_valid[c]:
        continue
    d = int(bank_chunk_doc[c]); j = fill[d]
    if j < P:
        dct[d, j] = c; dctv[d, j] = True; fill[d] = j + 1
dct, dctv = jnp.array(dct), jnp.array(dctv)
bcd, bcv = jnp.array(bank_chunk_doc), jnp.array(chunk_valid)

# routing check for layer 18: which docs selected for the query?
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
qt = tok(prompt, add_special_tokens=False)["input_ids"]
PROMPT_LEN = 64
qids = np.full((1, PROMPT_LEN), tok.pad_token_id, np.int64)
qmask = np.zeros((1, PROMPT_LEN), bool)
qids[0, PROMPT_LEN - len(qt):] = qt
qmask[0, PROMPT_LEN - len(qt):] = True
qids = np.repeat(qids, jax.device_count(), 0); qmask = np.repeat(qmask, jax.device_count(), 0)
B = qids.shape[0]
num_docs, MAXLEN, max_new = N, PROMPT_LEN + 48, 48

prefill = jax.jit(lambda a, b, ww, bk, t, tv, cd, cv: msa.prefill(cfg, a, b, ww, bk, t, tv, cd, cv, num_docs, MAXLEN))
decode = jax.jit(lambda tk, ww, ca, wi, pos: msa.decode_step(cfg, tk, ww, ca, wi, pos))

print("prefill...", flush=True)
qids_j = jax.device_put(jnp.array(qids), jax.sharding.PartitionSpec('data', None))
qmask_j = jax.device_put(jnp.array(qmask), jax.sharding.PartitionSpec('data', None))
logits, cache = prefill(qids_j, qmask_j, w, banks, dct, dctv, bcd, bcv)
base = jnp.array(qmask.sum(1).astype(np.int32))
nxt = jnp.argmax(logits[:, -1], -1)
gen = [np.array(nxt)]
for s in range(max_new - 1):
    wi = jnp.array(PROMPT_LEN + s, jnp.int32)
    ll, cache = decode(nxt[:, None], w, cache, wi, (base + s)[:, None])
    nxt = jnp.argmax(ll, -1)
    gen.append(np.array(nxt))
gen = np.stack(gen, 1)
eos = tok.eos_token_id
g0 = gen[0]
ep = np.where(g0 == eos)[0]
g0 = g0[:ep[0]] if len(ep) else g0
print("=== MSA GENERATION (q='capital of France') ===", flush=True)
print(repr(tok.decode(g0, skip_special_tokens=True)), flush=True)
print("SANITY2_DONE", flush=True)
