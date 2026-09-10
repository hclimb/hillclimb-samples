"""Streaming SnapKV baseline: off-the-shelf Qwen3-4B, whole corpus in context, KV compressed.

Adds the "context stuffing + KV-cache compression" point to the Pareto plots
(wiki/experiments/2026-07-21-snapkv-baseline.md). SnapKV (FasterDecoding/SnapKV) scores prompt
KV positions per KV head by the attention they receive from an observation window (the
question) and keeps the top-C per head. Two deviations from upstream, both forced and
documented:

  * Our JAX stack has plain RoPE (no YaRN), so a 131k+ single-window prefill is out of reach.
    The corpus is prefilled in SEGMENTS under a <=32k position budget: prefill segment ->
    probe with the question (models/qwen3.py::forward_window_scores) -> keep top-C of the
    segment per (layer, kv-head) -> compact the cache -> next segment starts at the compacted
    position. Kept keys retain their (segment-local) RoPE phases; cross-segment relative
    geometry is approximate — inherent to any streaming compression.
  * QASPER mode (--probe self) scores each segment with its own last tokens instead of the
    question (question-agnostic), so ONE compressed cache serves every question — 6.34M
    tokens/query would otherwise be infeasible. That variant is SnapKV-inspired (TOVA/H2O
    flavored), not faithful SnapKV; label it accordingly.

Positions are "compressed-space": each segment prefills at pos = current compacted length, so
total positions never exceed the budget and stay within Qwen3-4B's native 32,768 context.
Stale cache beyond the write frontier is never exposed (the causal mask only reveals
[0, pos+T)).

Run (both slice workers, via the launcher):
  RUN_SCRIPT wrapper sets MODE/COMP/etc; see scripts/embed/snapkv_musique.sh.
"""

import argparse
import json
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", default=None, help="HF dataset with a text column (doc corpus)")
    p.add_argument("--corpus_column", default="text")
    p.add_argument("--query_dataset", default=None)
    p.add_argument("--query_column", default="question")
    p.add_argument("--answer_column", default="answer")
    p.add_argument("--num_queries", type=int, default=128)
    p.add_argument("--comp", type=int, default=8, help="compression ratio per segment")
    p.add_argument("--seg", type=int, default=4096, help="segment length (tokens)")
    p.add_argument("--chunk", type=int, default=1024, help="prefill chunk length")
    p.add_argument("--win", type=int, default=96, help="probe window length (question padded to this)")
    p.add_argument("--pool", type=int, default=7, help="SnapKV 1D pooling kernel over scores")
    p.add_argument("--max_new", type=int, default=1280)
    p.add_argument("--probe", choices=["question", "self"], default="question",
                   help="question = faithful per-query SnapKV; self = query-agnostic (shared cache)")
    p.add_argument("--budget", type=int, default=0, help="cache budget; 0 = auto")
    p.add_argument("--out", default="outputs/snapkv_results.json")
    p.add_argument("--time_queries", type=int, default=8,
                   help="how many queries to wall-clock for the throughput point")
    p.add_argument("--max_corpus_tokens", type=int, default=0, help="debug: truncate corpus")
    p.add_argument("--no_compact", action="store_true",
                   help="debug: skip probe+compaction entirely (plain streaming prefill)")
    p.add_argument("--corpus_format", choices=["text", "qasper"], default="text",
                   help="qasper: assemble papers from allenai/qasper and take Q/A pairs from it")
    return p.parse_args()


def pad_to(ids, n, pad_tok):
    return ids + [pad_tok] * (n - len(ids)) if len(ids) < n else ids[:n]


def main():
    args = parse_args()
    from utils import init_jax_distributed
    init_jax_distributed()
    from models.qwen3 import load as load_qwen3, forward_window_scores
    from datasets import load_dataset
    from evals.utils import global_device_put, split_thinking

    if jax.process_index() == 0:
        import hashlib, inspect
        import models.qwen3 as _MQ
        _src = inspect.getsource(_MQ.forward_window_scores)
        print(f"[snapkv] qwen3 file={_MQ.__file__} fws-sha1={hashlib.sha1(_src.encode()).hexdigest()[:12]}")
        print(f"[snapkv] einsum: {[l.strip() for l in _src.splitlines() if 'einsum(' in l and 'bskh' in l]}")

    model = load_qwen3(args.model, tp_devices=1, load_weights=True)
    tok = model.tokenizer
    cfg, weights = model.cfg, model.weights
    mesh = next(v.sharding.mesh for v in weights.values()
                if hasattr(getattr(v, "sharding", None), "mesh"))
    B = int(mesh.shape["data"])
    NL = tok("\n", add_special_tokens=False)["input_ids"][-1]   # benign filler token
    proc0 = jax.process_index() == 0

    def load_dataset_retry(name, split, attempts=4):
        # transient HF 504s on one host kill the whole slice run (peer death cascades)
        for a in range(attempts):
            try:
                return load_dataset(name, split=split)
            except Exception as e:
                if a == attempts - 1:
                    raise
                if proc0:
                    print(f"[snapkv] HF load {name} failed ({e}); retry {a + 1}/{attempts}")
                time.sleep(20 * (a + 1))

    # ---- corpus token stream (tokenized once) --------------------------------------------
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    doc_ids, queries = [], []
    if args.corpus_format == "qasper":
        # papers -> one stream; Q/A pairs from the same split (answerable only). The 07-19
        # longdoc page's QASPER corpus is these full texts (~6.3M tokens over 1,169 papers).
        # allenai/qasper is a script dataset (unsupported by modern `datasets`); read the
        # Hub's auto-converted parquet branch. Corpus = ALL splits (the 07-19 longdoc page's
        # 1,169-paper / ~6.3M-token haystack); questions come from validation only.
        def _load_split(split):
            for a in range(4):
                try:
                    return load_dataset(
                        "parquet",
                        data_files=f"hf://datasets/allenai/qasper@refs/convert/parquet/qasper/{split}/0000.parquet",
                        split="train")
                except Exception as e:
                    if a == 3:
                        raise
                    if proc0:
                        print(f"[snapkv] qasper {split} load failed ({e}); retry {a + 1}/4")
                    time.sleep(20 * (a + 1))
        val = _load_split("validation")
        papers = list(val) + list(_load_split("train")) + list(_load_split("test"))
        val_ids = {p["id"] for p in val} if "id" in val.column_names else None
        n_docs = 0
        for paper in papers:
            parts = [paper["title"], paper["abstract"]]
            for sec in paper["full_text"]["paragraphs"]:
                parts += list(sec)
            doc_ids += tok("\n\n".join(p for p in parts if p), add_special_tokens=False)["input_ids"] + sep
            n_docs += 1
            if val_ids is not None and paper.get("id") not in val_ids:
                continue                                   # questions from validation only
            for q, ans in zip(paper["qas"]["question"], paper["qas"]["answers"]):
                golds = [a["free_form_answer"] or " ".join(a["extractive_spans"])
                         for a in ans["answer"]]
                golds = [g for g in golds if g]
                if golds and len(queries) < args.num_queries * 4:
                    queries.append((str(q), str(golds[0])))
        queries = queries[:args.num_queries]
        docs = list(range(n_docs))
    else:
        docs = load_dataset_retry(args.corpus, "train")
        for r in docs:
            doc_ids += tok(str(r[args.corpus_column]), add_special_tokens=False)["input_ids"] + sep
        qa = load_dataset_retry(args.query_dataset, "train")
        queries = [(str(qa[i][args.query_column]), str(qa[i][args.answer_column]))
                   for i in range(min(args.num_queries, len(qa)))]
    if args.max_corpus_tokens:
        doc_ids = doc_ids[:args.max_corpus_tokens]
    if proc0:
        print(f"[snapkv] corpus: {len(docs)} docs -> {len(doc_ids)} tokens"
              f"{' (truncated)' if args.max_corpus_tokens else ''}; {len(queries)} queries")

    # ---- prompt scaffolding ---------------------------------------------------------------
    # chat template split around the docs: [head + instr][DOCS][question tail + assistant]
    MARK = "\x00DOCS\x00"
    shell = tok.apply_chat_template(
        [{"role": "user", "content": f"Read the documents and answer the question.\n\n{MARK}\n\nQuestion: \x00Q\x00\nAnswer briefly."}],
        tokenize=False, add_generation_prompt=True)
    head_txt, rest = shell.split(MARK)
    head_ids = tok(head_txt, add_special_tokens=False)["input_ids"]

    def tail_ids(q):
        return tok(rest.replace("\x00Q\x00", q), add_special_tokens=False)["input_ids"]

    # ---- budget ----------------------------------------------------------------------------
    stream_base = head_ids + doc_ids
    n_seg = (len(stream_base) + args.seg - 1) // args.seg
    C = max(args.seg // args.comp, 8)   # QASPER-scale corpora need C well below the probe win
    budget = args.budget or (n_seg * C + args.seg + args.win + args.max_new + 64)
    if budget > 32768:
        raise ValueError(f"budget {budget} exceeds Qwen3 native 32768 — raise --comp "
                         f"(need >= {len(stream_base) // (32768 - args.seg - args.max_new - 160) + 1}x)")
    if proc0:
        print(f"[snapkv] segments={n_seg} seg={args.seg} C/seg={C} comp={args.comp}x "
              f"budget={budget} probe={args.probe}")

    L = cfg["num_hidden_layers"]
    rope_theta = float(cfg["rope_theta"])

    # ---- jitted pieces (fixed shapes: chunk, win, budget) ----------------------------------
    fwd = model.forward

    # weights are jit ARGUMENTS (closing over multi-host global arrays is not allowed)
    @partial(jax.jit, donate_argnums=(2,))
    def _prefill_chunk(w, ids, kv, pos):
        out = fwd(ids, w, kv=kv, pos=pos)
        return out.kv, out.logits[:, -1, :]

    @partial(jax.jit, donate_argnums=(2,))
    def _probe_scores(w, ids, kv, pos):
        kv, scores = forward_window_scores(cfg, ids, w, kv, pos)
        print(f"[probe-trace] per-layer score shape: {scores[0].shape}")
        # mean over the (data-sharded) batch axis: rows are identical query copies, and
        # slicing s[0] on a sharded dim is not allowed
        return kv, jnp.stack([s.mean(axis=0) for s in scores])   # [L, Kh, S]

    def prefill_chunk(ids, kv, pos):
        return _prefill_chunk(weights, ids, kv, pos)

    def probe_scores(ids, kv, pos):
        return _probe_scores(weights, ids, kv, pos)

    @partial(jax.jit, donate_argnums=(1,), static_argnames=("keep",))
    def compact(kv_l, scores_l, seg_start, seg_end, dst, keep):
        """One layer: pool scores, top-`keep` inside [seg_start, seg_end), sorted ascending,
        gather those cache rows and write them at dst. scores_l: [Kh, S]."""
        print(f"[compact-trace] kv_l={kv_l.shape} scores_l={scores_l.shape} keep={keep}")
        if scores_l.ndim == 3:
            # window dim survived upstream (stale-bytecode paranoia): reduce it here — summing
            # attention mass over window positions is exactly the intended SnapKV aggregation
            scores_l = scores_l.sum(axis=-2)
        S = scores_l.shape[-1]
        # SnapKV's local smoothing via shifted means (reduce_window's padding spec fought
        # back; edge wrap from roll is negligible for a smoothing kernel)
        offs = range(-(args.pool // 2), args.pool // 2 + 1)
        pooled = jnp.mean(jnp.stack([jnp.roll(scores_l, o, axis=-1) for o in offs]), axis=0)
        pos_idx = jnp.arange(S)[None, :]
        in_seg = (pos_idx >= seg_start) & (pos_idx < seg_end)
        pooled = jnp.where(in_seg, pooled, -jnp.inf)
        _, idx = jax.lax.top_k(pooled, keep)                     # [Kh, keep]
        idx = jnp.sort(idx, axis=-1)
        # gather via one-hot einsum: explicit-sharding mode rejects the ambiguous
        # take_along_axis gather ("provide out_sharding"), and einsum contracts S cleanly
        onehot = jax.nn.one_hot(idx, S, dtype=kv_l.dtype)        # [Kh, keep, S]
        # pin shardings around the gather (tp=1: the 'model' axis is size 1, so these
        # reshards are metadata-only, but explicit mode wants consistent batch-dim specs)
        onehot = jax.sharding.reshard(onehot, P(None, None, None))
        kvr = jax.sharding.reshard(kv_l, P(None, 'data', None, None, None))
        g = jnp.einsum('kcs,zbskh->zbckh', onehot, kvr,
                       out_sharding=P(None, 'data', None, None, None))
        # Re-rope gathered keys to their DESTINATION phases (position repacking). Fresh
        # segment writes have phase == slot, so kept keys carry phase == original slot idx;
        # after compaction their slot becomes dst+j. Without this, kept keys keep phases up
        # to seg_end while the next content starts at dst+keep < seg_end — queries would sit
        # at LOWER phases than cached keys, which degenerates decoding. RoPE composes, so a
        # delta rotation is exact. V is phase-free.
        k_g, v_g = g[0], g[1]                                    # [B, keep, Kh, H]
        Hd = k_g.shape[-1]
        src = idx.T.astype(jnp.float32)                          # [keep, Kh]
        dstp = (dst + jnp.arange(keep, dtype=jnp.int32)).astype(jnp.float32)[:, None]
        delta = dstp - src                                       # [keep, Kh]
        freqs = 1.0 / (rope_theta ** (jnp.arange(0, Hd, 2, dtype=jnp.float32) / Hd))
        ang = delta[..., None] * freqs                           # [keep, Kh, Hd/2]
        sn, cs = jnp.sin(ang), jnp.cos(ang)
        k1, k2 = k_g[..., :Hd // 2].astype(jnp.float32), k_g[..., Hd // 2:].astype(jnp.float32)
        k_rot = jnp.concatenate([k1 * cs - k2 * sn, k2 * cs + k1 * sn], axis=-1).astype(k_g.dtype)
        k_rot = jax.sharding.reshard(k_rot, P('data', None, None, None))
        v_g = jax.sharding.reshard(v_g, P('data', None, None, None))
        g = jnp.stack([k_rot, v_g])
        out = jax.lax.dynamic_update_slice(kvr, g, (0, 0, dst, 0, 0))
        return jax.sharding.reshard(out, P(None, 'data', None, 'model', None))

    @partial(jax.jit, donate_argnums=(2,))
    def _decode_step(w, tok_ids, kv, pos):
        out = fwd(tok_ids, w, kv=kv, pos=pos)
        return out.kv, jnp.argmax(out.logits[:, -1, :], axis=-1)

    def decode_step(tok_ids, kv, pos):
        return _decode_step(weights, tok_ids, kv, pos)

    def put(ids_np):
        return global_device_put(np.tile(np.asarray(ids_np, np.int32)[None, :], (B, 1)),
                                 mesh, P("data", None))

    # ---- streaming compression for one token stream ----------------------------------------
    def build_cache(stream, probe_ids_np):
        """probe_ids_np=None -> query-agnostic mode: each segment is scored by its own last
        `win` tokens (TOVA-style), so the compressed cache is query-independent."""
        kv = model.init_kv(B, budget)
        cur = 0
        for s0 in range(0, len(stream), args.seg):
            seg = pad_to(stream[s0:s0 + args.seg], args.seg, NL)
            for c0 in range(0, args.seg, args.chunk):
                kv, _ = prefill_chunk(put(seg[c0:c0 + args.chunk]), kv, jnp.int32(cur + c0))
            if args.no_compact:
                cur += args.seg
                continue
            seg_end = cur + args.seg
            probe = probe_ids_np if probe_ids_np is not None else seg[-args.win:]
            kv, sc = probe_scores(put(probe), kv, jnp.int32(seg_end))
            for l in range(L):
                kv[l] = compact(kv[l], sc[l], jnp.int32(cur), jnp.int32(seg_end),
                                jnp.int32(cur), keep=C)
            cur += C
        return kv, cur

    # ---- eval loop --------------------------------------------------------------------------
    results, timings = [], []
    eos = tok.eos_token_id
    shared_cache = None
    for qi, (q, gold) in enumerate(queries):
        t0 = time.perf_counter()
        probe_np = pad_to(tok(f"Question: {q}", add_special_tokens=False)["input_ids"],
                          args.win, NL) if args.probe == "question" else None

        if args.probe == "question":
            kv, cur = build_cache(stream_base, probe_np)
        else:
            if shared_cache is None:
                t_idx = time.perf_counter()
                kv0, cur0 = build_cache(stream_base, None)   # segment-tail (query-agnostic)
                if proc0:
                    print(f"[snapkv] shared cache built in {time.perf_counter() - t_idx:.0f}s "
                          f"({cur0} slots)", flush=True)
                shared_cache = (kv0, cur0)
            kv = [jnp.copy(x) for x in shared_cache[0]]
            cur = shared_cache[1]

        # question tail + decode. Pad at the FRONT (between docs and "Question:"): padding
        # after the assistant tag poisons greedy decoding into an endless newline run.
        tl = tail_ids(q)
        want = ((len(tl) + 63) // 64) * 64                       # 64-multiples: few jit shapes
        tl = [NL] * (want - len(tl)) + tl
        kv, last_logits = prefill_chunk(put(tl), kv, jnp.int32(cur))
        pos = cur + len(tl)
        nxt = jnp.argmax(last_logits, axis=-1)
        # Decode in on-device blocks: the python loop only enqueues async dispatches; the
        # host syncs once per BLOCK (EOS check), not once per token.
        BLOCK = 128
        out_toks = []
        done = False
        while not done and len(out_toks) < args.max_new:
            block = []
            for i in range(BLOCK):
                block.append(nxt)
                kv, nxt = decode_step(nxt[:, None].astype(jnp.int32), kv,
                                      jnp.int32(pos + len(out_toks) + i))
            host_block = np.asarray(jax.experimental.multihost_utils.process_allgather(
                jnp.stack(block, axis=1), tiled=True))          # [B, BLOCK], rows identical
            for t in host_block[0].tolist():
                out_toks.append(int(t))
                if int(t) == eos:
                    done = True
                    break
            else:
                continue
        timings.append(time.perf_counter() - t0)

        generated = tok.decode(out_toks, skip_special_tokens=True)
        thinking, answer = split_thinking(generated)
        results.append({"prompt": q, "generated": generated, "thinking": thinking,
                        "generated_answer": answer or generated.strip(), "ground_truth": gold,
                        "doc": ""})
        if proc0:
            print(f"[snapkv] q{qi}: {timings[-1]:.1f}s  ans={answer[:80]!r}", flush=True)

    metrics = {
        "generated_count": len(results), "comp": args.comp, "seg": args.seg,
        "probe": args.probe, "budget": budget, "corpus_tokens": len(stream_base),
        "median_query_s": float(np.median(timings)),
        "mean_query_s": float(np.mean(timings)),
    }
    if proc0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "samples": results}, f, indent=2)
        print(f"[snapkv] wrote {args.out}\n{json.dumps(metrics, indent=1)}")


if __name__ == "__main__":
    main()
