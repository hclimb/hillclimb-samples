"""RAG->memory-layer HYBRID evaluator: a retrieval-filtered memory bank over a shared corpus.

Motivation (wiki/experiments/2026-07-19-musique-corpus-scaling-and-throughput-pareto.md): the
memory layer's accuracy decay with corpus size is NOT a retrieval failure — `doc_hit_rate` holds
at 0.94+ while `mem_pos_weight_mass` collapses 4x — the gold slots are found and then out-voted
by distractor slots (attention dilution). This evaluator intervenes on exactly that mechanism:
a cheap dense retrieval (the SAME vanilla Qwen3-Embedding tower and settings as the classic-RAG
baseline, evals/rag/single_embedding_retrieval.py, so retrieval parity is by construction) picks
the top-k docs per query, and the memory bank is restricted to those docs' slots during
generation. Same checkpoint, same bank, same queries; only the candidate set changes.

How the restriction works. GenLargeMemEvaluator builds ONE bank for the whole corpus and every
query generates against it. Here the bank is also built once, but before each query the global
`main_model.mem_mask` is rewritten to (token-validity AND slot-belongs-to-a-top-k-doc).
retrieval_ops applies mem_mask as float32.min BEFORE top-k selection (models/retrieval_ops.py,
`_matmul_top_k` / `_scan_chunks`), so a masked slot can never be retrieved — masking is
equivalent to physically shrinking the bank, with no flat-index remapping. The mask keeps its
[M] shape and dtype across queries, so the jitted generation step never recompiles.

Deliberate simplifications vs the parent evaluator (this targets c512..c8192 corpora):
  * The bank is REPLICATED on every chip (the parent's MEM_REPLICATE_BANK branch): no data-axis
    sharding, no CPU mem_v callback, no CPU pre-norm. Guarded by _MAX_BANK_SLOTS — past that,
    use GenLargeMemEvaluator's sharded path instead.
  * Two batching modes (2026-07-21; was B=1-only). dataset.batch_size == 1 (legacy): the
    query is TILED to the mesh's data-axis size with a shared [M] mem_mask; rows are
    greedy-identical by construction, and cross-row divergence is counted
    (`row_divergence_rate`) as a free determinism check. dataset.batch_size == B > 1
    (data-parallel): B DISTINCT queries per forward with a per-example [B, M] mem_mask —
    models/retrieval_ops.py broadcasts either mask rank before top-k (chunked/replicated
    path only; other lookup variants raise). Decode is bandwidth-bound, so a B-row step
    costs about the same wall clock as B=1 — B=8 is ~8x query throughput. B must be a
    multiple of the mesh data-axis size; row_divergence_rate is not reported in this mode
    (no redundant rows to compare). Approx top-k is ALWAYS preferred per policy
    (wiki/architecture/retrieval-modes.md) — much faster, shifts well under the judge
    noise floor.
  * Prompts are left-padded to a FIXED `prompt_pad_len`, so the run compiles once. The parent
    pads to the per-batch max, which at one query per step would re-JIT every single query.

Telemetry: `doc_hit_rate` / `mem_pos_weight_mass` are computed on the generated answer span
against the per-query restricted mask, via the parent's numpy helpers. A gold doc the retrieval
step missed is masked out and can never be hit, so recall failures surface in those metrics and
in the per-sample `rag_all_golds_covered` flag. `rag_any_gold@k` / `rag_all_golds@k` are
reported over the full similarity ranking — MuSiQue questions carry 2-4 gold docs, and
*all-golds* coverage is the recall that actually bounds a multi-hop answer.

Oracle mode (`rag.oracle: true`): the mask is the query's own pos_doc_ids, optionally filled
with distractors to `rag.oracle_fill_to` in corpus order, and no encoder is loaded — the
perfect-retrieval ceiling that separates recall misses from residual dilution.
"""

import gc
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

from .base import Evaluator
from .gen_large_mem import (
    _mean_defined,
    _numpy_doc_hit_rate_gen_per_example,
    _numpy_pos_weight_mass_per_example_layers,
)
from .utils import embed_documents_tokenized, global_device_put, split_thinking
from inference import _generate_tokens

# Replicated-bank HBM guard. mem_k/mem_v are bf16 [M, ~2048]-ish; at 4M slots that is ~16 GB
# per side per chip, which does not leave room for the 4B model on 32 GB v6e HBM.
_MAX_BANK_SLOTS = 4_000_000


def rank_unique_doc_ids(chunk_ranking, chunk_doc_ids):
    """Collapse a ranked list of chunk indices into doc ids, keeping first-hit order.

    chunk_ranking:  1-D iterable of chunk indices, best first.
    chunk_doc_ids:  1-D array mapping chunk index -> corpus doc id.
    Returns the full list of unique doc ids in rank order (slice [:k] for a top-k).
    """
    out, seen = [], set()
    for ci in chunk_ranking:
        did = int(chunk_doc_ids[int(ci)])
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


def build_doc_slot_mask(doc_ids, doc_id_to_flat_indices, validity):
    """[M] mask restricting the bank to `doc_ids`: (slot belongs to a doc) AND validity.

    ANDing with the embed-produced token-validity mask matters — replacing it would turn the
    padding slots inside selected docs valid. Returned in validity's dtype so the device-side
    mem_mask keeps one dtype across queries (no recompile).
    """
    validity = np.asarray(validity)
    allowed = np.zeros(validity.shape[0], dtype=bool)
    for did in doc_ids:
        flat = doc_id_to_flat_indices.get(int(did))
        if flat:
            allowed[np.fromiter(flat, dtype=np.int64)] = True
    return (allowed & validity.astype(bool)).astype(validity.dtype)


def build_gather_plan(doc_ids, doc_id_to_flat_indices, docs_per_row, eff_doc_len):
    """Global flat slot indices for one row's fixed-size block of a gathered bank.

    Each of the row's first `docs_per_row` docs contributes its eff_doc_len contiguous
    slots, in ranked order. Rows with fewer docs are padded with slot 0 — the caller MUST
    mask the pad region off. Returns (flat_idx [docs_per_row * eff_doc_len] int64,
    n_valid_docs).
    """
    flat = np.zeros(docs_per_row * eff_doc_len, dtype=np.int64)
    n = 0
    for did in doc_ids:
        if n >= docs_per_row:
            break
        idxs = doc_id_to_flat_indices.get(int(did))
        if not idxs:
            continue
        block = np.fromiter(sorted(idxs), dtype=np.int64)
        flat[n * eff_doc_len:(n + 1) * eff_doc_len] = block[:eff_doc_len]
        n += 1
    return flat, n


def gold_coverage(ranked_doc_ids, gold_ids, ks):
    """{k: (any_gold, all_golds)} over a ranked doc-id list; (None, None) when no golds."""
    gold = {int(g) for g in gold_ids if int(g) >= 0}
    out = {}
    for k in ks:
        if not gold:
            out[k] = (None, None)
            continue
        hits = len(gold & set(ranked_doc_ids[:k]))
        out[k] = (hits > 0, hits == len(gold))
    return out


class GenLargeMemRagHybridEvaluator(Evaluator):

    def evaluate(self, model, dataset, step=None, aux_loss_config=None, doc_dataset=None, **kwargs):
        if doc_dataset is None:
            raise ValueError("GenLargeMemRagHybridEvaluator requires eval.doc_dataset.")
        if not bool(self.cfg.get("inject_query_gold", False)):
            raise NotImplementedError(
                "The hybrid eval is defined relative to the injected-gold corpus "
                "(inject_query_gold: true), matching gen_large_mem_musique_c512."
            )

        rag_cfg = self.cfg.get("rag", {}) or {}
        rag_top_k = int(rag_cfg.get("top_k", 50))
        oracle = bool(rag_cfg.get("oracle", False))
        oracle_fill_to = int(rag_cfg.get("oracle_fill_to", 0) or 0)
        prompt_pad_len = int(self.cfg.get("prompt_pad_len", 512))
        num_samples = self.cfg.get("num_samples", None)
        # GRPO-readiness diagnostic mode: keep every tiled row's completion (temperature>0
        # makes them independent samples, not redundant determinism checks) instead of
        # collapsing to row 0. See wiki/implementations/2026-08-24-grpo-readiness-multisample-hybrid-eval.md.
        multi_sample = bool(self.cfg.get("multi_sample", False))

        from models.memory_utils import resolve_approx_topk
        use_approx, _ = resolve_approx_topk(model.cfg.get("main_model", {}))
        if jax.process_index() == 0:
            print(f"Starting GenLargeMemRagHybrid eval "
                  f"(top_k={rag_top_k}, oracle={oracle}, approx_topk={use_approx})...")

        # ----------------------------------------------------------------
        # Queries: load the QA rows directly (question text for retrieval, pos_doc_ids for
        # golds). Same source, order and prefix as the eval dataset (shuffle: false), and the
        # same protocol as build_musique_c512_rag_corpus.py. A per-query containment assert
        # against the decoded prompt below makes the by-index join safe.
        # ----------------------------------------------------------------
        from datasets import load_dataset
        query_repo = rag_cfg.get("query_dataset", None)
        if not query_repo:
            raise ValueError("eval.rag.query_dataset is required (HF repo with question + pos_doc_ids).")
        query_split = rag_cfg.get("query_split", "train")
        query_column = rag_cfg.get("query_column", "question")
        # Load the FULL split and match batches to rows by question text: the data pipeline
        # drops some rows (length/format filters), so eval order != HF row order past the
        # first dropped row — a by-index join breaks (observed at query 16 on MuSiQue).
        qa_rows = load_dataset(query_repo, split=query_split)
        cand_questions = [str(r[query_column]) for r in qa_rows]
        cand_norm = [" ".join(q.split()) for q in cand_questions]
        cand_golds = [[int(v) for v in (r.get("pos_doc_ids") or []) if int(v) >= 0] for r in qa_rows]

        tokenizer = model.tokenizer
        # Collect ALL rows of every consumed batch (stop only at batch boundaries): the
        # generation pass below consumes the same generator in the same batch order, so every
        # row it sees must have a matched question here even when num_samples isn't a
        # multiple of the batch size.
        questions, row_gold_ids = [], []
        for batch_tokens, batch_masks in dataset.generator(num_epochs=1):
            if num_samples is not None and len(questions) >= num_samples:
                break
            if not isinstance(batch_tokens, dict):
                batch_tokens = {"batch": batch_tokens}
            raw = np.array(batch_tokens["batch"])
            lm_all = np.array(batch_masks["loss_mask"])
            for r in range(raw.shape[0]):
                ans_pos = np.where(lm_all[r] > 0)[0]
                pe = int(ans_pos[0]) if len(ans_pos) > 0 else raw.shape[1]
                prompt_norm = " ".join(tokenizer.decode(raw[r, :pe], skip_special_tokens=False).split())
                hits = [i for i, qn in enumerate(cand_norm) if qn and qn in prompt_norm]
                if not hits:
                    raise ValueError(
                        f"query {len(questions)}: no {query_repo} question found in the eval "
                        f"prompt — dataset/source mismatch.\n  prompt: {prompt_norm[:400]!r}"
                    )
                hit = max(hits, key=lambda i: len(cand_norm[i]))   # longest wins substring ties
                questions.append(cand_questions[hit])
                row_gold_ids.append(cand_golds[hit])
        gold_ids = set(g for row in row_gold_ids for g in row)
        if jax.process_index() == 0:
            print(f"  {len(questions)} queries matched against {query_repo} "
                  f"({len(qa_rows)} rows), {len(gold_ids)} gold doc ids")

        # ----------------------------------------------------------------
        # Corpus: scan the FULL doc corpus, keep every gold chunk, fill with distractors in
        # corpus order up to target_docs — the same selection as the parent's
        # inject_query_gold path, so the haystack matches the non-hybrid eval exactly.
        # ----------------------------------------------------------------
        max_chunks_per_doc = self.cfg.doc_dataset.get("max_chunks_per_doc", None)
        target_docs = self.cfg.doc_dataset.get("target_docs", self.cfg.doc_dataset.get("max_docs", None))
        target_chunks = (target_docs * (max_chunks_per_doc or 1)) if target_docs is not None else None

        di, dm, dd = [], [], []
        for ids_batch, masks_batch, doc_id_batch in doc_dataset.generator():
            di.append(ids_batch); dm.append(masks_batch); dd.append(np.array(doc_id_batch))
        ids = np.concatenate(di); masks = np.concatenate(dm); dids = np.concatenate(dd)
        is_gold = np.isin(dids, np.array(sorted(gold_ids), dtype=dids.dtype)) if gold_ids else np.zeros(len(dids), bool)
        gi = np.where(is_gold)[0]; oi = np.where(~is_gold)[0]
        n_other = max(0, (target_chunks - len(gi)) if target_chunks is not None else len(oi))
        sel = np.concatenate([gi, oi[:n_other]])
        all_doc_ids = ids[sel]; all_doc_masks = masks[sel]; all_corpus_doc_ids = dids[sel]
        total_corpus_chunks = all_doc_ids.shape[0]
        missing = gold_ids - set(int(d) for d in all_corpus_doc_ids)
        if missing:
            raise ValueError(f"{len(missing)} gold doc ids absent from the doc corpus: {sorted(missing)[:10]}")
        if jax.process_index() == 0:
            print(f"  corpus (scanned {len(dids)}): {len(gi)} gold + {min(n_other, len(oi))} "
                  f"distractor chunks = {total_corpus_chunks} (target={target_docs})")

        # Chunk texts: used by the retrieval encoder and for the samples' "doc" field
        # (lexical_grounding reads it). Token->text round-trip of what the bank actually
        # indexes — closer to ground truth than re-reading raw text from the hub.
        tokenizer = model.tokenizer
        chunk_texts = [
            tokenizer.decode(all_doc_ids[i][all_doc_masks[i] > 0], skip_special_tokens=True)
            for i in range(total_corpus_chunks)
        ]

        # Mesh of the loaded memory model — needed before the retrieval pre-pass so every
        # host-array placement below is multi-host safe (bare-P device_put is not).
        some_w = next(v for v in model.weights.values()
                      if hasattr(getattr(v, 'sharding', None), 'mesh'))
        mesh = some_w.sharding.mesh
        n_rows = int(mesh.shape['data'])
        if multi_sample and jax.process_index() == 0:
            print(f"  multi_sample=true: group size = mesh data-axis size = {n_rows} "
                  f"independent completions/query (temperature={self.cfg.get('temperature', 0.0)})")

        # ----------------------------------------------------------------
        # Retrieval pre-pass -> per-query ranked doc ids. Oracle mode skips the encoder.
        # ----------------------------------------------------------------
        # auto-K: pick the smallest candidate k whose mean all-golds coverage clears the
        # threshold (else the largest candidate = the cap). Chosen AFTER the pre-pass ranks
        # the full corpus, so it costs nothing extra.
        auto_k_threshold = rag_cfg.get("auto_k_threshold", None)
        auto_k_candidates = sorted(int(x) for x in
                                   (rag_cfg.get("auto_k_candidates", None) or (5, 10, 25, 50, 100, 150, 200)))

        ranked_ids_per_query, coverage_per_query = [], []
        ks = {5, 10, 25, 50, 100} | {rag_top_k}
        if auto_k_threshold is not None:
            ks |= set(auto_k_candidates)
        ks = sorted(k for k in ks if k <= total_corpus_chunks)
        if oracle:
            fill_to = max(oracle_fill_to, 0)
            corpus_order_ids = list(dict.fromkeys(int(d) for d in all_corpus_doc_ids))
            for golds in row_gold_ids:
                picked = list(golds)
                for did in corpus_order_ids:
                    if fill_to and len(picked) >= fill_to:
                        break
                    if did not in picked:
                        picked.append(did)
                ranked_ids_per_query.append(picked if fill_to else list(golds))
                coverage_per_query.append(None)
        else:
            from types import SimpleNamespace
            from .rag.single_embedding_retrieval import build_encoder, encode_texts
            enc_args = SimpleNamespace(
                model_name=rag_cfg.get("encoder", "Qwen/Qwen3-Embedding-0.6B"),
                tp_devices=1,
                hf_ckpt_dir=os.path.expanduser(rag_cfg.get("hf_ckpt_dir", "~/weights/huggingface")),
                # 256/128: parity with scripts/misc/rag_only.py (the RAG baseline this is read against).
                max_doc_length=int(rag_cfg.get("max_doc_length", 256)),
                max_query_length=int(rag_cfg.get("max_query_length", 128)),
            )
            t0 = time.perf_counter()
            enc_tokenizer, enc_embed_fn = build_encoder(enc_args)
            enc_bs = int(rag_cfg.get("encode_batch_size", 512))
            # This encoder is always built at tp_devices=1 (enc_args above), independent of
            # this eval's own `mesh` — placing its inputs onto `mesh` instead of its OWN weight
            # mesh mismatches whenever the eval's tp_devices != 1 (create_mask then raises
            # "context mesh ... should match the aval mesh ...").
            enc_mesh = getattr(enc_embed_fn, "mesh", None) or mesh
            doc_embs = encode_texts(chunk_texts, enc_tokenizer, enc_embed_fn,
                                    enc_args.max_doc_length, enc_bs, desc="RAG docs", mesh=enc_mesh)
            q_embs = encode_texts(questions, enc_tokenizer, enc_embed_fn,
                                  enc_args.max_query_length, enc_bs, desc="RAG queries", mesh=enc_mesh)
            del enc_embed_fn, enc_tokenizer
            gc.collect()
            jax.effects_barrier()
            sims = q_embs @ doc_embs.T                                # (Q, C) float32
            chunk_rankings = np.argsort(-sims, axis=1)
            ranked_full_per_query = []
            for qi in range(len(questions)):
                full = rank_unique_doc_ids(chunk_rankings[qi], all_corpus_doc_ids)
                ranked_full_per_query.append(full)
                coverage_per_query.append(gold_coverage(full, row_gold_ids[qi], ks))
            if auto_k_threshold is not None:
                usable = [c for c in auto_k_candidates if c <= total_corpus_chunks]
                chosen = None
                for cand in usable:
                    allg = _mean_defined([c[cand][1] for c in coverage_per_query if c])
                    if allg is not None and allg >= float(auto_k_threshold):
                        chosen = cand
                        break
                rag_top_k = chosen if chosen is not None else max(usable)
                if jax.process_index() == 0:
                    print(f"  auto-K: rag.top_k={rag_top_k} "
                          f"({'smallest candidate with all_golds >= ' + str(auto_k_threshold) if chosen is not None else 'CAP — no candidate cleared ' + str(auto_k_threshold)}; "
                          f"candidates={usable})")
            ranked_ids_per_query = [full[:rag_top_k] for full in ranked_full_per_query]
            del ranked_full_per_query
            if jax.process_index() == 0:
                print(f"  retrieval pre-pass done in {time.perf_counter() - t0:.1f}s")
                for k in ks:
                    anyg = _mean_defined([c[k][0] for c in coverage_per_query if c])
                    allg = _mean_defined([c[k][1] for c in coverage_per_query if c])
                    if anyg is not None:
                        print(f"    rag_any_gold@{k}={anyg:.4f}  rag_all_golds@{k}={allg:.4f}")

        # ----------------------------------------------------------------
        # Bank: embed all chunks with the model's embed tower, then REPLICATE on every chip
        # (the parent's MEM_REPLICATE_BANK branch: mem_shard_axis='model', mem_v on device,
        # no CPU callback, no pre-norm — the model does mem_k norm in-JIT).
        # ----------------------------------------------------------------
        from models.qwen3_mem_embed import embed_forward
        from models.retrieval_ops import set_cpu_mem_v, set_global_mesh
        from models.utils import split_weights

        cfg = model.cfg
        _main_w, embed_w = split_weights(model.weights, ["main_model", "embed_model"])
        embed_cfg = cfg["embed_model"]
        embed_w = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in embed_w.items()}
        embed_fn = lambda docs, dmask: embed_forward(embed_cfg, docs, embed_w, dmask)

        embed_batch_size = self.cfg.get("embed_batch_size", 64)
        # jax.set_mesh is a PROCESS-WIDE, non-scoped side effect of load_qwen3 (models/qwen3.py:309)
        # — the retrieval pre-pass above loads a SEPARATE encoder at tp_devices=1 (build_encoder),
        # which overwrites JAX's ambient mesh to that encoder's OWN (data=8,model=1) ambient
        # and never restores it. embed_documents_tokenized below runs the MAIN model's embed
        # tower (on `mesh`, this eval's real tp_devices target) — its create_mask calls need the
        # ambient mesh explicitly restored to `mesh` here, or they inherit the stale tp=1 one.
        jax.set_mesh(mesh)
        set_global_mesh(mesh)
        mem_k_np, mem_v_np, mem_mask_np = embed_documents_tokenized(
            embed_fn, all_doc_ids, all_doc_masks, embed_batch_size, mesh=mesh
        )
        M = mem_k_np.shape[0]
        gather_bank = bool(self.cfg.get("gather_bank", False))
        if gather_bank and (max_chunks_per_doc or 1) != 1:
            raise NotImplementedError(
                "gather_bank assumes max_chunks_per_doc == 1 (one fixed-size block per doc)."
            )
        if not gather_bank and M > _MAX_BANK_SLOTS:
            raise ValueError(
                f"bank has {M} slots > {_MAX_BANK_SLOTS}; the replicated-bank hybrid eval is "
                "for small/medium corpora — use gather_bank: true (host-side bank, per-batch "
                "gather) or GenLargeMemEvaluator's sharded path."
            )
        eff_doc_len = M // total_corpus_chunks

        doc_id_to_flat_indices = {}
        doc_id_to_chunk = {}
        for chunk_idx in range(total_corpus_chunks):
            did = int(all_corpus_doc_ids[chunk_idx])
            base = chunk_idx * eff_doc_len
            doc_id_to_flat_indices.setdefault(did, set()).update(range(base, base + eff_doc_len))
            doc_id_to_chunk.setdefault(did, chunk_idx)

        # Free embed weights before generation (single largest HBM waste otherwise);
        # restored at the end for subsequent evals in multi-eval runs.
        embed_w_saved = embed_w
        del embed_fn, embed_w
        _main_w = None
        model.weights = {k: v for k, v in model.weights.items() if k.startswith('main_model.')}
        gc.collect()
        jax.effects_barrier()

        k_sharding = NamedSharding(mesh, P(*([None] * mem_k_np.ndim)))
        v_sharding = NamedSharding(mesh, P(*([None] * mem_v_np.ndim)))
        mask_sharding = NamedSharding(mesh, P(None))

        def _put_replicated(arr_np, sharding, dtype=None):
            return jax.make_array_from_callback(
                arr_np.shape, sharding,
                lambda idx: arr_np[idx].astype(dtype) if dtype is not None else arr_np[idx],
            )

        set_cpu_mem_v(None)
        params_with_mem = {k: v for k, v in model.weights.items() if k.startswith('main_model.')}
        if gather_bank:
            # Per-query small banks: the full bank stays in HOST numpy; every batch ships only
            # its rows' top-k docs' slots, one fixed-size block per row (fixed shapes ->
            # compile once). Retrieval-equivalent to masking the full bank (masking is applied
            # pre-top-k), but the scan covers B*k docs instead of the whole corpus, the
            # replicated-bank HBM guard no longer applies, and the aux telemetry fits again.
            bank_docs_per_row = max((len(lst) for lst in ranked_ids_per_query), default=0)
            if bank_docs_per_row == 0:
                raise ValueError("gather_bank: no retrieved docs to build per-query banks from.")
            gather_block = bank_docs_per_row * eff_doc_len
            if jax.process_index() == 0:
                print(f"  gather-bank mode: host bank {M} slots; per-row block "
                      f"{bank_docs_per_row} docs x {eff_doc_len} = {gather_block} slots")
        else:
            bank_docs_per_row = gather_block = None
            params_with_mem["main_model.mem_k"] = _put_replicated(mem_k_np, k_sharding, dtype=jnp.bfloat16)
            params_with_mem["main_model.mem_v"] = _put_replicated(mem_v_np, v_sharding, dtype=jnp.bfloat16)
            params_with_mem["main_model.mem_mask"] = _put_replicated(mem_mask_np, mask_sharding)

        model.cfg['main_model']['mem_shard_axis'] = 'model'   # mesh size 1 -> replicated lookup
        lookup_chunk_size = self.cfg.get("lookup_chunk_size", None)
        reset_chunk_size = False
        if lookup_chunk_size is not None:
            if model.cfg['main_model'].get("mem_lookup_chunk_size", None) is None:
                reset_chunk_size = True
            model.cfg['main_model']['mem_lookup_chunk_size'] = lookup_chunk_size

        if not gather_bank and jax.process_index() == 0:
            print(f"  bank replicated: mem_k {mem_k_np.shape}, mem_v {mem_v_np.shape}, "
                  f"eff_doc_len={eff_doc_len}, rows/query={n_rows}")

        # ----------------------------------------------------------------
        # Generation loop: one query per step, per-query mem_mask, fixed shapes throughout.
        # ----------------------------------------------------------------
        @partial(jax.jit, static_argnames=("forward",))
        def _aux_fn(forward, params, tokens, pmask):
            return forward(tokens, params, pad_mask=pmask, collect_aux=True).aux

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_new_tokens = int(self.cfg.get("max_new_tokens", 64))
        mask_dtype = mem_mask_np.dtype

        results = []
        hit_scores, mass_scores, active_slot_counts = [], [], []
        n_diverged = 0
        n_div_checked = 0
        aux_disabled = False
        qi = 0
        pbar = tqdm(total=num_samples, desc="Generating (rag hybrid)")

        for batch_tokens, batch_masks in dataset.generator(num_epochs=1):
            if num_samples is not None and qi >= num_samples:
                break
            if qi >= len(questions):
                break

            if not isinstance(batch_tokens, dict):
                batch_tokens = {"batch": batch_tokens}
            raw_batch = np.array(batch_tokens["batch"])
            B = raw_batch.shape[0]
            tiled_mode = (B == 1)
            if not tiled_mode and B % n_rows != 0:
                raise ValueError(
                    f"dataset.batch_size={B} must be 1 (tiled legacy mode) or a multiple of "
                    f"the mesh data-axis size ({n_rows}) so the batch shards evenly."
                )
            if multi_sample and not tiled_mode:
                raise ValueError(
                    "eval.multi_sample requires dataset.batch_size=1 (tiled mode) — the "
                    "multi-sample group is exactly the mesh's tiled replicas of one query."
                )
            batch_mask_all = np.array(batch_masks["batch_mask"])
            loss_mask_all = np.array(batch_masks["loss_mask"])

            # Per-row prompt bookkeeping. Queries were matched by content in the pre-pass;
            # the containment check only catches a non-deterministic generator restart.
            pes, gt_ids_rows, prompt_texts = [], [], []
            for r in range(B):
                ans_pos = np.where(loss_mask_all[r] > 0)[0]
                pe = int(ans_pos[0]) if len(ans_pos) > 0 else raw_batch.shape[1]
                if pe > prompt_pad_len:
                    raise ValueError(
                        f"query {qi + r}: prompt is {pe} tokens > prompt_pad_len="
                        f"{prompt_pad_len}; raise eval.prompt_pad_len (costs one extra compile).")
                prompt_text = tokenizer.decode(raw_batch[r, :pe], skip_special_tokens=False)
                q_norm = " ".join(questions[qi + r].split())
                if q_norm not in " ".join(prompt_text.split()):
                    raise ValueError(
                        f"query {qi + r}: generator restart yielded a different order — matched "
                        f"question not in this prompt.\n  question: {questions[qi + r]!r}"
                    )
                pes.append(pe)
                gt_ids_rows.append(raw_batch[r][ans_pos].tolist() if len(ans_pos) > 0 else [])
                prompt_texts.append(prompt_text)

            # Per-query bank restriction. Same shapes+dtypes every batch -> no recompile.
            # tiled_mode keeps the legacy [M] device mask; B>1 ships [B, M] sharded over data.
            all_flat = None
            if gather_bank:
                # Gather each row's docs into its own fixed-size block of a small shared bank
                # and ship it; each row's mask confines it to its own block.
                flat_rows, valid_counts = [], []
                for r in range(B):
                    flat_r, n_valid = build_gather_plan(
                        ranked_ids_per_query[qi + r], doc_id_to_flat_indices,
                        bank_docs_per_row, eff_doc_len)
                    flat_rows.append(flat_r)
                    valid_counts.append(n_valid)
                all_flat = np.concatenate(flat_rows)              # [B * gather_block]
                validity_small = mem_mask_np[all_flat]
                masks_np = np.zeros((B, all_flat.shape[0]), dtype=mem_mask_np.dtype)
                for r in range(B):
                    sl = slice(r * gather_block, r * gather_block + valid_counts[r] * eff_doc_len)
                    masks_np[r, sl] = validity_small[sl]
                params_with_mem["main_model.mem_k"] = _put_replicated(
                    mem_k_np[all_flat], k_sharding, dtype=jnp.bfloat16)
                params_with_mem["main_model.mem_v"] = _put_replicated(
                    mem_v_np[all_flat], v_sharding, dtype=jnp.bfloat16)
            else:
                masks_np = np.stack([
                    build_doc_slot_mask(ranked_ids_per_query[qi + r], doc_id_to_flat_indices, mem_mask_np)
                    for r in range(B)
                ])
            for r in range(B):
                active_slot_counts.append(int(masks_np[r].astype(bool).sum()))
            if tiled_mode:
                params_with_mem["main_model.mem_mask"] = _put_replicated(
                    masks_np[0].astype(mask_dtype), mask_sharding)
            else:
                params_with_mem["main_model.mem_mask"] = global_device_put(
                    masks_np.astype(mask_dtype), mesh, P('data', None))

            # Fixed-length left-pad; tiled to the data axis at B=1, real rows at B>1.
            prompt_rows = np.full((B, prompt_pad_len), pad_id, dtype=raw_batch.dtype)
            pmask_rows = np.zeros((B, prompt_pad_len), dtype=np.bool_)
            for r in range(B):
                pe = pes[r]
                prompt_rows[r, prompt_pad_len - pe:] = raw_batch[r, :pe]
                pmask_rows[r, prompt_pad_len - pe:] = batch_mask_all[r, :pe].astype(np.bool_)
            if tiled_mode:
                prompt_tiled = np.tile(prompt_rows, (n_rows, 1))
                pmask_tiled = np.tile(pmask_rows, (n_rows, 1))
            else:
                prompt_tiled, pmask_tiled = prompt_rows, pmask_rows
            prompt_jax = global_device_put(prompt_tiled, mesh, P('data', None))
            pmask_jax = global_device_put(pmask_tiled, mesh, P('data', None))

            gen_tokens = _generate_tokens(
                model.forward, model.init_kv, params_with_mem, prompt_jax,
                max_new_tokens, pad_mask=pmask_jax,
                temperature=self.cfg.get("temperature", 0.0),
                top_k=self.cfg.get("top_k", 20),
                top_p=self.cfg.get("top_p", 0.8),
            )
            jax.block_until_ready(gen_tokens)
            gen_np = np.array(jax.experimental.multihost_utils.process_allgather(gen_tokens, tiled=True))
            if tiled_mode:
                diverged = sum(not np.array_equal(gen_np[r], gen_np[0]) for r in range(1, gen_np.shape[0]))
                n_diverged += int(diverged > 0)
                n_div_checked += 1
            else:
                diverged = 0
            if tiled_mode:
                gen_rows = gen_np if multi_sample else gen_np[:1]
            else:
                gen_rows = gen_np                                # [B, max_new_tokens]
            n_group = gen_rows.shape[0]   # samples sharing this query: n_rows if multi_sample
                                           # else 1 (tiled) or B (data-parallel, one each)

            gls = []
            for s in range(n_group):
                eos_pos = np.where(gen_rows[s] == tokenizer.eos_token_id)[0]
                gls.append(int(eos_pos[0]) if len(eos_pos) > 0 else max_new_tokens)

            pos_sets = []
            for r in range(B):
                ps = set()
                for did in row_gold_ids[qi + r]:
                    ps.update(doc_id_to_flat_indices.get(int(did), set()))
                pos_sets.append(ps)
            # In gather mode the forward's indices are LOCAL to the gathered bank; translate
            # gold slot sets to local positions. Golds duplicated into other rows' blocks are
            # harmless — each row's mask (passed as validity below) confines hits to its own
            # block, and an uncovered gold simply has no valid local position (a true miss).
            if gather_bank:
                pos_sets_eff = []
                for ps in pos_sets:
                    if not ps:
                        pos_sets_eff.append(set())
                        continue
                    pos_arr = np.fromiter(ps, dtype=np.int64)
                    pos_sets_eff.append(set(np.where(np.isin(all_flat, pos_arr))[0].tolist()))
            else:
                pos_sets_eff = pos_sets

            # Retrieval telemetry on the answer span: one aux forward over prompt+generation
            # (fixed shape -> compiles once). Best-effort: the full-sequence aux forward
            # needs more HBM than generation (its chunk-scan buffers scale with
            # S x lookup_chunk_size); at large banks it can OOM where generation fits.
            # Judge accuracy and rag coverage do not depend on it.
            q_hits = [None] * n_group
            q_masses = [None] * n_group
            aux = None
            if not aux_disabled:
                # multi_sample: gen_rows already holds n_rows DISTINCT completions (one per
                # tiled replica) — feed them through as-is instead of tiling row 0, so the aux
                # forward's per-replica telemetry is real per-sample data, not n_rows copies
                # of the same answer.
                if tiled_mode and multi_sample:
                    gen_all = gen_rows
                elif tiled_mode:
                    gen_all = np.tile(gen_rows[0], (n_rows, 1))
                else:
                    gen_all = gen_rows
                full_seq = np.concatenate([prompt_tiled, gen_all], axis=1)
                full_mask = np.concatenate(
                    [pmask_tiled, np.ones((prompt_tiled.shape[0], max_new_tokens), dtype=np.bool_)], axis=1)
                try:
                    aux = _aux_fn(model.forward,
                                  params_with_mem,
                                  global_device_put(full_seq, mesh, P('data', None)),
                                  global_device_put(full_mask, mesh, P('data', None)))
                except jax.errors.JaxRuntimeError as e:
                    if "RESOURCE_EXHAUSTED" not in str(e):
                        raise
                    aux_disabled = True
                    if jax.process_index() == 0:
                        print("  WARNING: aux telemetry forward OOM'd — doc_hit_rate/"
                              "mem_pos_weight_mass disabled for this run "
                              "(judge accuracy and rag coverage unaffected).")
            if aux is not None and aux.get("mem_top_k_indices") and any(pos_sets):
                _ga = jax.experimental.multihost_utils.process_allgather
                idx_layers, prob_layers = [], []
                prob_list = aux.get("mem_top_k_probs") or []
                # multi_sample keeps every replica's telemetry (n_group == n_rows); plain
                # tiled mode keeps only the one representative row; data-parallel keeps all B.
                keep = n_group if (tiled_mode and multi_sample) else (1 if tiled_mode else B)
                for li in range(len(aux["mem_top_k_indices"])):
                    idx_layers.append(np.array(_ga(
                        aux["mem_top_k_indices"][li][:, :, prompt_pad_len:, :], tiled=True))[:keep])
                    if li < len(prob_list):
                        prob_layers.append(np.array(_ga(
                            prob_list[li][:, :, prompt_pad_len:, :], tiled=True))[:keep])
                # qidx indexes the REAL dataset row (pos_sets/pos_sets_eff/masks_np, size B):
                # the same query for every sample s when tiled, else one query per s.
                for s in range(n_group):
                    qidx = 0 if tiled_mode else s
                    if not pos_sets[qidx]:
                        continue
                    rr = s
                    gen_active = np.zeros((1, max_new_tokens), dtype=np.float32)
                    gen_active[0, :gls[s]] = 1.0
                    q_hits[s] = _numpy_doc_hit_rate_gen_per_example(
                        idx_layers[0][rr:rr + 1], [pos_sets_eff[qidx]], np.array([gls[s]]), masks_np[qidx])[0]
                    if prob_layers:
                        q_masses[s] = _numpy_pos_weight_mass_per_example_layers(
                            [idx[rr:rr + 1] for idx in idx_layers],
                            [pb[rr:rr + 1] for pb in prob_layers],
                            [pos_sets_eff[qidx]], gen_active, masks_np[qidx])[0]
                    # A row whose golds have no reachable local position (uncovered at this k)
                    # is a true retrieval miss, not an undefined sample.
                    if pos_sets[qidx] and not pos_sets_eff[qidx]:
                        q_hits[s] = 0.0
                        if prob_layers:
                            q_masses[s] = 0.0

            for s in range(n_group):
                qidx = 0 if tiled_mode else s
                hit_scores.append(q_hits[s])
                mass_scores.append(q_masses[s])
                gold_set_r = set(row_gold_ids[qi + qidx])
                gold_chunk_idx = [ci for ci in range(total_corpus_chunks)
                                  if int(all_corpus_doc_ids[ci]) in gold_set_r]
                generated = tokenizer.decode(gen_rows[s][:gls[s]], skip_special_tokens=True)
                thinking, generated_answer = split_thinking(generated)
                doc_ids_q = ranked_ids_per_query[qi + qidx]
                cov = coverage_per_query[qi + qidx]
                all_covered = (cov[rag_top_k][1] if cov else
                               gold_set_r <= set(int(d) for d in doc_ids_q))
                results.append({
                    "prompt": prompt_texts[qidx],
                    "generated": generated,
                    "thinking": thinking,
                    "generated_answer": generated_answer,
                    "ground_truth": tokenizer.decode(gt_ids_rows[qidx], skip_special_tokens=True),
                    "doc": " || ".join(chunk_texts[ci] for ci in gold_chunk_idx),
                    "doc_hit_rate": q_hits[s],
                    "pos_slot_weight_mass": q_masses[s],
                    "rag_doc_ids": [int(d) for d in doc_ids_q],
                    # Retrieved docs' text, rank order — what the restricted bank actually
                    # held for this query (token->text round-trip, same as "doc").
                    "rag_docs": [chunk_texts[doc_id_to_chunk[int(d)]] for d in doc_ids_q
                                 if int(d) in doc_id_to_chunk],
                    "rag_all_golds_covered": None if all_covered is None else bool(all_covered),
                    "n_gold_docs": len(row_gold_ids[qi + qidx]),
                    "rows_diverged": int(diverged),
                    # group_id identifies the query; sample_idx distinguishes independent
                    # completions of the SAME query under multi_sample (0 otherwise) — the
                    # post-hoc pass@k / intra-group-variance script groups on group_id.
                    "group_id": qi + qidx,
                    "sample_idx": s if (tiled_mode and multi_sample) else 0,
                })
            qi += B
            pbar.update(B)
        pbar.close()

        # A tail batch can overshoot num_samples (only when num_samples % batch_size != 0).
        # num_samples always counts QUERIES, not rows: under multi_sample every query
        # contributes group_size (n_rows) rows, so the trim target must scale accordingly --
        # comparing len(results) straight against num_samples would truncate to a fraction of
        # a single query's group instead of dropping only a genuine tail overshoot.
        trim_target = num_samples * n_rows if multi_sample else num_samples
        if num_samples is not None and len(results) > trim_target:
            results = results[:trim_target]
            hit_scores = hit_scores[:trim_target]
            mass_scores = mass_scores[:trim_target]
            active_slot_counts = active_slot_counts[:trim_target]

        # ----------------------------------------------------------------
        # Cleanup + restore (multi-eval runs reuse the model object).
        # ----------------------------------------------------------------
        set_global_mesh(None)
        set_cpu_mem_v(None)
        model.cfg['main_model'].pop('mem_shard_axis', None)
        if reset_chunk_size:
            model.cfg['main_model']['mem_lookup_chunk_size'] = None
        for k, v in embed_w_saved.items():
            model.weights[f'embed_model.{k}'] = v
        del embed_w_saved

        file_metrics = {
            "generated_count": len(results),
            "rag_top_k": rag_top_k,
            "rag_oracle": oracle,
            "bank_slots": int(M),
            "corpus_docs": int(total_corpus_chunks // max(max_chunks_per_doc or 1, 1)),
            "aux_telemetry_oom": aux_disabled,
            "multi_sample": multi_sample,
        }
        if multi_sample:
            file_metrics["group_size"] = n_rows
        # Only meaningful in tiled (B=1) mode, where redundant rows exist to compare.
        if n_div_checked:
            file_metrics["row_divergence_rate"] = n_diverged / n_div_checked
        if active_slot_counts:
            file_metrics["mean_active_bank_slots"] = float(np.mean(active_slot_counts))
        hit_mean = _mean_defined(hit_scores)
        if hit_mean is not None:
            file_metrics["doc_hit_rate"] = hit_mean
        mass_mean = _mean_defined(mass_scores)
        if mass_mean is not None:
            file_metrics["mem_pos_weight_mass"] = mass_mean
        for k in ks:
            anyg = _mean_defined([c[k][0] for c in coverage_per_query if c])
            allg = _mean_defined([c[k][1] for c in coverage_per_query if c])
            if anyg is not None:
                file_metrics[f"rag_any_gold@{k}"] = anyg
                file_metrics[f"rag_all_golds@{k}"] = allg
        if (file_metrics.get("row_divergence_rate", 0) > 0 and not multi_sample
                and jax.process_index() == 0):
            print(f"  WARNING: {n_diverged}/{n_div_checked} queries had divergent tiled rows — "
                  "generation is not deterministic (approx top-k on?).")

        if self.cfg.get("output_file", None) and jax.process_index() == 0:
            output_path = self._get_output_path(step, self.cfg.output_file)
            with open(output_path, "w") as f:
                json.dump({"metrics": file_metrics, "samples": results}, f, indent=2)
            print(f"Saved generations to {output_path}")
            if wandb.run is not None:
                artifact_name = (f"{wandb.run.id}-eval-{self.key}-step-{step}-results"
                                 if step is not None else f"{wandb.run.id}-eval-{self.key}-results")
                artifact = wandb.Artifact(name=artifact_name, type="evaluation_results")
                artifact.add_file(output_path)
                wandb.log_artifact(artifact)

        return dict(file_metrics)
