"""MSA large-static-memory evaluator.

Adapts the gen_large_mem flow to the MSA architecture (models/qwen3_msa.py):
  Phase 1 — encode the whole doc corpus with the 4B backbone into per-router-layer
            pooled banks (K̄, V̄, K̄ᴿ) + a doc->pooled-chunk table.
  Phase 2 — for each QA batch: route top-k docs per router layer, concat their pooled
            KV ahead of the query's local KV, generate the answer (greedy), and emit
            judge-ready records.

Selected via eval type `generation_large_mem_msa`.
"""
import json
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm import tqdm

from .base import Evaluator
from .utils import split_thinking
import models.qwen3_msa as msa


def _extract_answer(text):
    """Pull the final answer out of MSA's generative-retrieval-style output.

    MSA-4B emits e.g.:
      "the document number related to the above issue is:\n[12][34]<End-of-Retrieve>
       The user's question is: <q>?The answer to the question is: <ANSWER>"
    Prefer the span after "answer to the question is:"; else after <End-of-Retrieve>
    (dropping the echoed question); else fall back to the <think> split / raw text.
    """
    for marker in ("answer to the question is:", "The answer is:", "answer:"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    if "<End-of-Retrieve>" in text:
        tail = text.split("<End-of-Retrieve>", 1)[1]
        if "?" in tail:                       # drop "The user's question is: ...?"
            tail = tail.split("?", 1)[1]
        return tail.strip()
    _, ans = split_thinking(text)
    return (ans or text).strip()


class GenLargeMemMSAEvaluator(Evaluator):
    def evaluate(self, model, dataset, step=None, doc_dataset=None, **kwargs):
        if doc_dataset is None:
            raise ValueError("GenLargeMemMSAEvaluator requires a doc_dataset.")
        cfg = model.cfg
        weights = model.weights
        # REPLICATE_WEIGHTS=1: replicate the base transformer weights (P()) so decode
        # incurs no per-matmul all-reduce. qwen3.load shards the contraction dim on the
        # 8-chip 'data' axis at tp=1, which cripples small-batch decode (the §2b bug).
        # Needed for a fair same-engine MSA throughput number (membed restores weights
        # replicated already; MSA loads fresh from HF, so it must be replicated here).
        if __import__("os").environ.get("REPLICATE_WEIGHTS"):
            from jax.sharding import NamedSharding, PartitionSpec as _P
            _ref = next(v for k, v in weights.items() if "q_proj" in k)
            _rep = NamedSharding(_ref.sharding.mesh, _P())
            weights = {k: jax.device_put(v, _rep) for k, v in weights.items()}
            jax.block_until_ready(weights)
            if jax.process_index() == 0:
                print("[MSA] REPLICATED weights for decode", flush=True)
        tok = model.tokenizer
        kernel = cfg['msa']['pooling_kernel_size']
        router_layers = cfg['msa']['router_layers']
        p0 = jax.process_index() == 0

        # ---------------- Phase 1: collect + encode corpus ----------------
        max_docs = self.cfg.doc_dataset.get("max_docs", None)
        max_cpd = self.cfg.doc_dataset.get("max_chunks_per_doc", None)
        max_chunks = (max_docs * max_cpd if (max_docs and max_cpd) else None)
        # cap: inherited generation_large_mem default (2048) is sized for the 0.6B
        # embedder; the 4B MSA backbone needs a much smaller doc-encode batch.
        ebs = min(int(self.cfg.get("msa_embed_batch_size", 32)), 64)

        all_ids, all_masks, all_docids, n = [], [], [], 0
        for ids_b, masks_b, docid_b in doc_dataset.generator():
            all_ids.append(ids_b); all_masks.append(masks_b); all_docids.append(docid_b)
            n += ids_b.shape[0]
            if max_chunks is not None and n >= max_chunks:
                break
        all_ids = np.concatenate(all_ids); all_masks = np.concatenate(all_masks)
        all_docids = np.concatenate(all_docids)
        if max_chunks is not None:
            all_ids, all_masks, all_docids = all_ids[:max_chunks], all_masks[:max_chunks], all_docids[:max_chunks]
        R, T = all_ids.shape
        ppr = T // kernel                       # pooled chunks per row
        if p0:
            print(f"[MSA] corpus rows={R}, chunk_len={T}, pooled/row={ppr}", flush=True)

        BENCH = os.environ.get("MEMBENCH")
        bench = {"model": "qwen3_msa", "dataset": self.cfg.get("output_file", "")}
        doc_tokens_valid = int(all_masks.astype(bool).sum())
        t_enc0 = time.perf_counter()
        encode_jit = jax.jit(partial(msa.encode_docs, cfg))
        kbar = {L: [] for L in router_layers}
        vbar = {L: [] for L in router_layers}
        krbar = {L: [] for L in router_layers}
        cvalid = []
        # pad R up to a multiple of ebs for static-shape encode
        for s in range(0, R, ebs):
            ids_np = all_ids[s:s + ebs]; m_np = all_masks[s:s + ebs].astype(bool)
            b = ids_np.shape[0]
            if b < ebs:  # pad last batch
                ids_np = np.concatenate([ids_np, np.zeros((ebs - b, T), ids_np.dtype)])
                m_np = np.concatenate([m_np, np.zeros((ebs - b, T), bool)])
            out = encode_jit(jnp.array(ids_np), jnp.array(m_np), weights)
            for L in router_layers:
                kbar[L].append(np.array(out['kbar'][L])[:b])
                vbar[L].append(np.array(out['vbar'][L])[:b])
                krbar[L].append(np.array(out['krbar'][L])[:b])
            cvalid.append(np.array(out['chunk_valid'])[:b])

        # concat rows -> [R, ppr, 8,128] -> flatten chunks -> [C,8,128]; C = R*ppr
        banks = {'kbar': {}, 'vbar': {}, 'krbar': {}}
        for L in router_layers:
            banks['kbar'][L] = jnp.array(np.concatenate(kbar[L]).reshape(R * ppr, *kbar[L][0].shape[2:]))
            banks['vbar'][L] = jnp.array(np.concatenate(vbar[L]).reshape(R * ppr, *vbar[L][0].shape[2:]))
            banks['krbar'][L] = jnp.array(np.concatenate(krbar[L]).reshape(R * ppr, *krbar[L][0].shape[2:]))
        chunk_valid = np.concatenate(cvalid).reshape(R * ppr)          # [C]

        # dense doc-id remap + chunk->doc + doc->chunk table
        row_docids = all_docids.astype(np.int64)
        uniq, inv = np.unique(row_docids, return_inverse=True)
        num_docs = len(uniq)
        bank_chunk_doc = np.repeat(inv, ppr)                           # [C] dense doc id
        C = R * ppr
        # P = max pooled chunks per doc
        counts = np.bincount(bank_chunk_doc[chunk_valid], minlength=num_docs)
        P = int(counts.max()) if num_docs > 0 else ppr
        doc_chunk_table = np.zeros((num_docs, P), np.int32)
        doc_chunk_table_valid = np.zeros((num_docs, P), bool)
        fill = np.zeros(num_docs, np.int32)
        for c in range(C):
            if not chunk_valid[c]:
                continue
            d = bank_chunk_doc[c]
            j = fill[d]
            if j < P:
                doc_chunk_table[d, j] = c
                doc_chunk_table_valid[d, j] = True
                fill[d] = j + 1
        if p0:
            print(f"[MSA] num_docs={num_docs}, C={C}, P(max chunks/doc)={P}", flush=True)

        dct = jnp.array(doc_chunk_table)
        dctv = jnp.array(doc_chunk_table_valid)
        bcd = jnp.array(bank_chunk_doc)
        bcv = jnp.array(chunk_valid)
        jax.block_until_ready((banks['kbar'][router_layers[0]], dct))
        t_encode = time.perf_counter() - t_enc0
        top_k = int(cfg['msa'].get('top_k', 16))
        bench.update({
            "num_docs": int(num_docs), "bank_vectors_per_layer": int(C),
            "router_layers": len(router_layers), "pooling_kernel": int(kernel),
            "doc_tokens_in_memory": doc_tokens_valid,
            "top_k_docs_per_layer": top_k, "max_chunks_per_doc_pooled": int(P),
            # raw doc-tokens the model conditions on in ONE forward (per router layer):
            # top_k docs x P pooled chunks x kernel tokens/chunk
            "attended_doc_tokens_per_layer": int(top_k * P * kernel),
            "encode_corpus_s": round(t_encode, 3),
        })

        # ---------------- Phase 2: generation ----------------
        max_new = int(self.cfg.get("max_new_tokens", 64))
        PROMPT_LEN = int(self.cfg.get("prompt_len", getattr(self.cfg, "seq_len", 512)))
        MAXLEN = PROMPT_LEN + max_new
        num_samples = self.cfg.get("num_samples", None)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id

        prefill_jit = jax.jit(lambda qids, qmask, w, bk, t, tv, cd, cv:
                              msa.prefill(cfg, qids, qmask, w, bk, t, tv, cd, cv, num_docs, MAXLEN))
        decode_jit = jax.jit(lambda tk, w, ca, wi, pos: msa.decode_step(cfg, tk, w, ca, wi, pos))

        results, count = [], 0
        pbar = tqdm(total=num_samples or 0, desc="MSA gen", disable=not p0)
        for batch_tokens, batch_masks in dataset.generator(num_epochs=1):
            if num_samples is not None and count >= num_samples:
                break
            if not isinstance(batch_tokens, dict):
                batch_tokens = {"batch": batch_tokens}
            raw_batch = np.array(batch_tokens["batch"])
            batch_mask = np.array(batch_masks["batch_mask"])
            loss_mask = np.array(batch_masks["loss_mask"])
            B, Tb = raw_batch.shape

            prompt_ends, gt_answer_ids = [], []
            for i in range(B):
                ap = np.where(loss_mask[i] > 0)[0]
                pe = int(ap[0]) if len(ap) > 0 else Tb
                prompt_ends.append(pe)
                gt_answer_ids.append(raw_batch[i, ap].tolist() if len(ap) > 0 else [])

            # left-pad prompts to PROMPT_LEN (truncate left if longer)
            qids = np.full((B, PROMPT_LEN), pad_id, dtype=raw_batch.dtype)
            qmask = np.zeros((B, PROMPT_LEN), dtype=bool)
            for i in range(B):
                pe = min(prompt_ends[i], PROMPT_LEN)
                src = raw_batch[i, :prompt_ends[i]][-PROMPT_LEN:]
                m = batch_mask[i, :prompt_ends[i]][-PROMPT_LEN:].astype(bool)
                qids[i, PROMPT_LEN - len(src):] = src
                qmask[i, PROMPT_LEN - len(m):] = m

            from jax.sharding import PartitionSpec as P
            qids_j = jax.device_put(jnp.array(qids), P('data', None))
            qmask_j = jax.device_put(jnp.array(qmask), P('data', None))
            t_pf = time.perf_counter()
            logits, cache = prefill_jit(qids_j, qmask_j, weights, banks, dct, dctv, bcd, bcv)
            jax.block_until_ready(logits)
            t_prefill = time.perf_counter() - t_pf
            base_pos = jnp.array(qmask.sum(axis=1).astype(np.int32))   # [B]
            nxt = jnp.argmax(logits[:, -1], axis=-1)                   # [B]

            t_dec = time.perf_counter()
            n_steps = 1
            gen_cols = [nxt]                                           # device arrays
            for stp in range(max_new - 1):
                widx = jnp.array(PROMPT_LEN + stp, dtype=jnp.int32)
                pos = (base_pos + stp)[:, None]
                logits_l, cache = decode_jit(nxt[:, None], weights, cache, widx, pos)
                nxt = jnp.argmax(logits_l, axis=-1)
                gen_cols.append(nxt)
                n_steps += 1
                # amortized early-stop: sync every 32 steps, break once all rows EOS'd
                if (stp + 1) % 32 == 0:
                    col = np.array(jnp.stack(gen_cols, axis=1))
                    if np.all(np.any(col == eos_id, axis=1)):
                        break
            gen = np.array(jnp.stack(gen_cols, axis=1))                # [B, <=max_new]
            jax.block_until_ready(nxt)
            t_decode = time.perf_counter() - t_dec
            if BENCH:
                bench.setdefault("batches", []).append({
                    "B": int(B), "prefill_s": round(t_prefill, 4),
                    "decode_s": round(t_decode, 4), "decode_steps": int(n_steps),
                })

            actual = min(B, num_samples - count) if num_samples is not None else B
            for i in range(actual):
                eosp = np.where(gen[i] == eos_id)[0]
                gi = gen[i, :eosp[0]] if len(eosp) > 0 else gen[i]
                generated = tok.decode(gi, skip_special_tokens=True)
                thinking, _ = split_thinking(generated)
                gen_answer = _extract_answer(generated)
                results.append({
                    "prompt": tok.decode(raw_batch[i, :prompt_ends[i]], skip_special_tokens=False),
                    "generated": generated,
                    "thinking": thinking,
                    "generated_answer": gen_answer,
                    "ground_truth": tok.decode(gt_answer_ids[i], skip_special_tokens=True),
                })
            count += actual
            pbar.update(actual)
        pbar.close()

        if BENCH and p0:
            os.makedirs(BENCH, exist_ok=True)
            bpath = os.path.join(BENCH, "bench_msa.json")
            with open(bpath, "w") as f:
                json.dump(bench, f, indent=2)
            print(f"[MSA][BENCH] wrote {bpath}", flush=True)

        out = {"metrics": {"generated_count": count}, "samples": results}
        if self.cfg.get("output_file", None) and p0:
            path = self._get_output_path(step, self.cfg.output_file)
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[MSA] wrote {len(results)} samples -> {path}", flush=True)
        return {"output_file": self.cfg.get("output_file", None),
                "metrics_cfg": self.cfg.get("metrics", None),
                "samples": results}
