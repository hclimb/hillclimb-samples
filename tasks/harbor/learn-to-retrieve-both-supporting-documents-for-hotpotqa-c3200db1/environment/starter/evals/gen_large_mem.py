import gc
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import json
import wandb
from tqdm import tqdm
from functools import partial

from .base import Evaluator
from .utils import embed_documents_tokenized, split_thinking
from inference import _generate_tokens
from utils import unfreeze_dict


def _numpy_doc_access_acc(indices_np, pos_sets, loss_mask, mem_validity):
    """Numpy-based doc_access_acc for corpus-level memory.

    Args:
        indices_np:    (B, H, S, K) int array of retrieved flat memory indices.
        pos_sets:      list of B sets; each set contains flat memory indices that
                       belong to positive documents for that batch item.
        loss_mask:     (B, S) bool/int array; 1 for real prompt positions.
        mem_validity:  1-D array of length >= max(indices_np); 1 for valid tokens.
    """
    total_correct = total_valid = 0
    B, H, S, K = indices_np.shape
    for i in range(B):
        batch_idx = indices_np[i]                                  # (H, S, K)
        is_valid  = mem_validity[batch_idx]                        # (H, S, K)
        is_active = loss_mask[i][np.newaxis, :, np.newaxis]        # (1, S, 1)
        valid_active = is_valid.astype(bool) & is_active.astype(bool)
        pos_arr = (np.array(sorted(pos_sets[i]), dtype=np.int64)
                   if pos_sets[i] else np.empty(0, dtype=np.int64))
        is_pos = (np.isin(batch_idx, pos_arr)
                  if len(pos_arr) > 0
                  else np.zeros_like(batch_idx, dtype=bool))
        total_correct += int(np.sum(is_pos & valid_active))
        total_valid   += int(np.sum(valid_active))
    return total_correct / total_valid if total_valid > 0 else 0.0


def _numpy_doc_access_acc_per_example(indices_np, pos_sets, loss_mask, mem_validity):
    """Per-example version of doc_access_acc.

    Returns a list of length B. Items are floats in [0, 1] when the example has
    any valid active retrieval slots, else None.
    """
    scores = []
    B, H, S, K = indices_np.shape
    for i in range(B):
        batch_idx = indices_np[i]                                  # (H, S, K)
        is_valid  = mem_validity[batch_idx]                        # (H, S, K)
        is_active = loss_mask[i][np.newaxis, :, np.newaxis]        # (1, S, 1)
        valid_active = is_valid.astype(bool) & is_active.astype(bool)
        valid_total = int(np.sum(valid_active))
        if valid_total == 0:
            scores.append(None)
            continue
        pos_arr = (np.array(sorted(pos_sets[i]), dtype=np.int64)
                   if pos_sets[i] else np.empty(0, dtype=np.int64))
        is_pos = (np.isin(batch_idx, pos_arr)
                  if len(pos_arr) > 0
                  else np.zeros_like(batch_idx, dtype=bool))
        scores.append(float(np.sum(is_pos & valid_active)) / valid_total)
    return scores


def _numpy_pos_weight_mass(indices_np, probs_np, pos_sets, loss_mask, mem_validity):
    """Corpus-level positive-slot softmax weight mass for ONE memory layer — the corpus
    analog of losses/mem_telemetry.py::_positive_slot_stats (train/mem_pos_weight_mass).

    Same ratio as the train-time metric — Σ(w on positive-doc slots) / Σ(w on valid slots) —
    but joins positives via the corpus `pos_sets` (flat bank indices) instead of the batch
    docs_mask/pos_doc_mask layout, which doesn't exist for a shared corpus bank. This is why
    the train-time metric can't just be reused here: `_positive_slot_stats` derives the doc id
    as `index // doc_len` over a per-batch doc grid, which is meaningless for a corpus bank.

    Pooled over the batch (Σnum/Σden), matching _numpy_doc_access_acc's pooling, so examples
    contribute in proportion to their valid slot weight.

    Args:
        indices_np:   (B, H, S, K) int, retrieved flat bank indices.
        probs_np:     (B, H, S, K) float, softmax weight on each retrieved slot.
        pos_sets:     list of B sets of flat indices belonging to positive docs.
        loss_mask:    (B, S) 1 for positions to measure (the answer span).
        mem_validity: 1-D array over the bank; 1 for valid tokens.
    Returns float in [0,1], or None when no valid weight was retrieved.
    """
    num = den = 0.0
    B = indices_np.shape[0]
    for i in range(B):
        batch_idx = indices_np[i]                                  # (H, S, K)
        is_valid  = mem_validity[batch_idx].astype(bool)           # (H, S, K)
        is_active = loss_mask[i][np.newaxis, :, np.newaxis].astype(bool)
        valid_active = is_valid & is_active
        pos_arr = (np.array(sorted(pos_sets[i]), dtype=np.int64)
                   if pos_sets[i] else np.empty(0, dtype=np.int64))
        is_pos = (np.isin(batch_idx, pos_arr)
                  if len(pos_arr) > 0
                  else np.zeros_like(batch_idx, dtype=bool))
        w = probs_np[i].astype(np.float64)
        num += float(np.sum(w * (is_pos & valid_active)))
        den += float(np.sum(w * valid_active))
    return (num / den) if den > 0 else None


def _numpy_pos_weight_mass_per_example(indices_np, probs_np, pos_sets, loss_mask, mem_validity):
    """Per-example version of _numpy_pos_weight_mass. Returns a list of length B; items are
    floats in [0,1], or None where the example retrieved no valid weight. Kept per-example so
    the mass can be split by the LLM-judge verdict downstream (same use as
    generation_embed.py::_per_example_pos_slot_mass)."""
    scores = []
    B = indices_np.shape[0]
    for i in range(B):
        batch_idx = indices_np[i]
        is_valid  = mem_validity[batch_idx].astype(bool)
        is_active = loss_mask[i][np.newaxis, :, np.newaxis].astype(bool)
        valid_active = is_valid & is_active
        w = probs_np[i].astype(np.float64)
        den = float(np.sum(w * valid_active))
        if den <= 0:
            scores.append(None)
            continue
        pos_arr = (np.array(sorted(pos_sets[i]), dtype=np.int64)
                   if pos_sets[i] else np.empty(0, dtype=np.int64))
        is_pos = (np.isin(batch_idx, pos_arr)
                  if len(pos_arr) > 0
                  else np.zeros_like(batch_idx, dtype=bool))
        scores.append(float(np.sum(w * (is_pos & valid_active))) / den)
    return scores


def _mean_defined(vals):
    """Mean over non-None entries; None if all are None (mirrors run_metrics' None handling)."""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _numpy_pos_weight_mass_per_example_layers(idx_by_layer, prob_by_layer, pos_sets, loss_mask, mem_validity):
    """Per-example layer-mean mass: list of length B (None where undefined on every layer)."""
    per_layer = [
        _numpy_pos_weight_mass_per_example(idx, prob, pos_sets, loss_mask, mem_validity)
        for idx, prob in zip(idx_by_layer, prob_by_layer)
    ]
    if not per_layer:
        return []
    return [_mean_defined([lay[i] for lay in per_layer]) for i in range(len(per_layer[0]))]


def _numpy_pos_weight_mass_layers(idx_by_layer, prob_by_layer, pos_sets, loss_mask, mem_validity):
    """Layer-mean of _numpy_pos_weight_mass, matching the train-time metric's mean-over-layers
    aggregation (mem_telemetry.collect_eval_telemetry). qwen3_mem_embed has mem_layers=[14] —
    a single memory layer — so today this is that one layer; written per-layer so it stays
    correct if mem_layers grows. Returns None if no layer produced a defined value."""
    vals = [
        _numpy_pos_weight_mass(idx, prob, pos_sets, loss_mask, mem_validity)
        for idx, prob in zip(idx_by_layer, prob_by_layer)
    ]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _numpy_doc_hit_rate_gen(indices_np, pos_sets, gen_lengths, mem_validity):
    """Example-level recall over decoded tokens: fraction of examples where any
    (decode_pos, head, k) lookup hit the positive document set."""
    B, H, S_gen, K = indices_np.shape
    hits = valid_count = 0
    for i in range(B):
        if not pos_sets[i]:
            continue
        valid_count += 1
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        gl = int(gen_lengths[i])
        ex_indices = indices_np[i, :, :gl, :]                      # (H, gl, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = (np.isin(ex_indices, pos_arr) if len(pos_arr) > 0
                      else np.zeros_like(ex_indices, dtype=bool))
        if np.any(ex_valid & ex_pos):
            hits += 1
    return hits / valid_count if valid_count > 0 else 0.0


def _numpy_doc_hit_rate_gen_per_example(indices_np, pos_sets, gen_lengths, mem_validity):
    """Per-example version of decoding doc_hit_rate.

    Returns 1.0/0.0 for examples with positive docs, else None.
    """
    scores = []
    B, H, S_gen, K = indices_np.shape
    for i in range(B):
        if not pos_sets[i]:
            scores.append(None)
            continue
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        gl = int(gen_lengths[i])
        ex_indices = indices_np[i, :, :gl, :]                      # (H, gl, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = (np.isin(ex_indices, pos_arr) if len(pos_arr) > 0
                      else np.zeros_like(ex_indices, dtype=bool))
        scores.append(float(np.any(ex_valid & ex_pos)))
    return scores


def _numpy_doc_hit_rate(indices_np, pos_sets, loss_mask, mem_validity):
    """Per-example recall over prompt positions (kept for reference/testing)."""
    B, H, S, K = indices_np.shape
    hits = valid_count = 0
    for i in range(B):
        if not pos_sets[i]:
            continue
        valid_count += 1
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        batch_idx = indices_np[i]
        is_valid  = mem_validity[batch_idx].astype(bool)
        is_active = loss_mask[i][np.newaxis, :, np.newaxis].astype(bool)
        candidates = batch_idx[is_valid & is_active]
        if len(candidates) > 0 and np.any(np.isin(candidates, pos_arr)):
            hits += 1
    return hits / valid_count if valid_count > 0 else 0.0


def _numpy_doc_token_hit_rate(indices_np, pos_sets, loss_mask, mem_validity):
    """Token-level hit rate over active prompt tokens.

    A token counts as a hit if any retrieved (head, k) slot lands in the
    positive document set for that example.
    """
    B, H, S, K = indices_np.shape
    total_pos = hit_pos = 0
    for i in range(B):
        if not pos_sets[i]:
            continue
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        ex_indices = indices_np[i]                               # (H, S, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = np.isin(ex_indices, pos_arr) if len(pos_arr) > 0 else np.zeros_like(ex_indices, dtype=bool)
        any_hit    = np.any(ex_valid & ex_pos, axis=(0, 2))     # (S,)
        for s in range(S):
            if not loss_mask[i, s]:
                continue
            total_pos += 1
            if any_hit[s]:
                hit_pos += 1
    return hit_pos / total_pos if total_pos > 0 else 0.0


def _numpy_doc_token_hit_rate_per_example(indices_np, pos_sets, loss_mask, mem_validity):
    """Per-example version of prompt token-level hit rate."""
    scores = []
    B, H, S, K = indices_np.shape
    for i in range(B):
        active_total = int(np.sum(loss_mask[i]))
        if active_total == 0 or not pos_sets[i]:
            scores.append(None)
            continue
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        ex_indices = indices_np[i]                               # (H, S, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = np.isin(ex_indices, pos_arr) if len(pos_arr) > 0 else np.zeros_like(ex_indices, dtype=bool)
        any_hit    = np.any(ex_valid & ex_pos, axis=(0, 2))     # (S,)
        active_mask = loss_mask[i].astype(bool)
        scores.append(float(np.sum(any_hit & active_mask)) / active_total)
    return scores


def _numpy_doc_gen_token_hit_rate(indices_np, pos_sets, gen_lengths, mem_validity):
    """Token-level hit rate over valid generated tokens.

    A generated token counts as a hit if any retrieved (head, k) slot lands in
    the positive document set for that example.
    """
    B, H, S_gen, K = indices_np.shape
    total_tokens = hit_tokens = 0
    for i in range(B):
        if not pos_sets[i]:
            continue
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        ex_indices = indices_np[i]                                   # (H, S_gen, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = (np.isin(ex_indices, pos_arr) if len(pos_arr) > 0
                      else np.zeros_like(ex_indices, dtype=bool))
        any_hit    = np.any(ex_valid & ex_pos, axis=(0, 2))         # (S_gen,)
        gl         = int(gen_lengths[i])
        valid_mask = np.arange(S_gen) < gl
        total_tokens += int(np.sum(valid_mask))
        hit_tokens += int(np.sum(any_hit & valid_mask))
    return hit_tokens / total_tokens if total_tokens > 0 else 0.0


def _numpy_doc_gen_token_hit_rate_per_example(indices_np, pos_sets, gen_lengths, mem_validity):
    """Per-example version of decoding token-level hit rate."""
    scores = []
    B, H, S_gen, K = indices_np.shape
    for i in range(B):
        gl = int(gen_lengths[i])
        if gl <= 0 or not pos_sets[i]:
            scores.append(None)
            continue
        pos_arr = np.array(sorted(pos_sets[i]), dtype=np.int64)
        ex_indices = indices_np[i]                                   # (H, S_gen, K)
        ex_valid   = mem_validity[ex_indices].astype(bool)
        ex_pos     = (np.isin(ex_indices, pos_arr) if len(pos_arr) > 0
                      else np.zeros_like(ex_indices, dtype=bool))
        any_hit    = np.any(ex_valid & ex_pos, axis=(0, 2))         # (S_gen,)
        valid_mask = np.arange(S_gen) < gl
        scores.append(float(np.sum(any_hit & valid_mask)) / gl)
    return scores


class GenLargeMemEvaluator(Evaluator):
    """
    Two-phase large-memory generation evaluator.

    Phase 1 — Embed ALL documents using only the embed model weights.
              Documents are batched with full data-parallelism (tp=1 style:
              all devices on the 'data' axis).  Results gathered to CPU numpy
              and embed weights freed from device memory.

    Phase 2 — Shard mem_k/mem_v across ALL devices on the 'data' axis
              (N/num_devices vectors per device).  Load main model only and
              run generation with chunked sharded retrieval (shard_axis='data').
    """

    def evaluate(self, model, dataset, step=None, aux_loss_config=None, doc_dataset=None, **kwargs):
        if jax.process_index() == 0:
            print("Starting GenLargeMem Evaluation (two-phase, data-axis sharding)...")

        if doc_dataset is None:
            raise ValueError(
                "GenLargeMemEvaluator requires a doc_dataset. "
                "Configure it via eval.doc_dataset in the task config."
            )

        # ----------------------------------------------------------------
        # Resolve mesh from any existing sharded weight
        # ----------------------------------------------------------------
        some_w = next(
            v for v in model.weights.values()
            if hasattr(getattr(v, 'sharding', None), 'mesh')
        )
        mesh = some_w.sharding.mesh
        n_devices = jax.device_count()

        if jax.process_index() == 0:
            print(f"  Mesh: {mesh.shape}, total devices: {n_devices}")

        # ----------------------------------------------------------------
        # Phase 1 — build embed fn, collect all doc tokens, run embedding
        # ----------------------------------------------------------------
        from models.qwen3_mem_embed import embed_forward
        from models.utils import split_weights
        from jax.sharding import NamedSharding, PartitionSpec as P

        weights = model.weights
        cfg = model.cfg

        _main_w, embed_w = split_weights(weights, ["main_model", "embed_model"])
        embed_cfg = cfg["embed_model"]
        embed_w = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
                   for k, v in embed_w.items()}
        embed_fn = lambda docs, dmask: embed_forward(embed_cfg, docs, embed_w, dmask)

        max_docs = self.cfg.doc_dataset.get("max_docs", None)
        max_chunks_per_doc = self.cfg.doc_dataset.get("max_chunks_per_doc", None)
        max_chunks = (max_docs * max_chunks_per_doc
                      if max_docs is not None and max_chunks_per_doc is not None
                      else None)
        embed_batch_size = self.cfg.get("embed_batch_size", 64)

        if jax.process_index() == 0:
            print(f"Phase 1: collecting doc tokens (max_docs={max_docs}, "
                  f"embed_batch_size={embed_batch_size})...")

        # inject_query_gold: guarantee the eval queries' gold docs are IN the corpus, then fill
        # with distractors up to max_docs (so a small corpus is {gold} + {N - #gold distractors},
        # not a random subset that misses the gold entirely). Needs the QA dataset to carry
        # pos_doc_ids (same id space as the corpus doc_id).
        inject_gold = bool(self.cfg.get("inject_query_gold", False))
        gold_ids = set()
        if inject_gold:
            num_q = self.cfg.get("num_samples", None)
            seen_q = 0
            for _bt, _bm in dataset.generator(num_epochs=1):
                if isinstance(_bm, dict) and "pos_doc_ids" in _bm:
                    for v in np.array(_bm["pos_doc_ids"]).reshape(-1):
                        if int(v) >= 0:
                            gold_ids.add(int(v))
                _b = _bt["batch"] if isinstance(_bt, dict) else _bt
                seen_q += np.array(_b).shape[0]
                if num_q is not None and seen_q >= num_q:
                    break
            if jax.process_index() == 0:
                print(f"  inject_query_gold: {len(gold_ids)} gold doc ids from {seen_q} queries")

        if inject_gold:
            # Scan the FULL doc corpus (leave doc_dataset.max_docs=null so the generator yields
            # everything — cheap: tokenized chunks only, no embedding). target_docs sets the final
            # corpus size: corpus = all gold chunks + distractors up to target. This guarantees the
            # eval queries' gold docs are present regardless of where they sit in the corpus order.
            target_docs = self.cfg.doc_dataset.get("target_docs", max_docs)
            target_chunks = (target_docs * (max_chunks_per_doc or 1)) if target_docs is not None else None
            di, dm, dd = [], [], []
            for ids_batch, masks_batch, doc_id_batch in doc_dataset.generator():
                di.append(ids_batch); dm.append(masks_batch); dd.append(np.array(doc_id_batch))
            ids = np.concatenate(di); masks = np.concatenate(dm); dids = np.concatenate(dd)
            is_gold = np.isin(dids, np.array(sorted(gold_ids), dtype=dids.dtype)) if gold_ids else np.zeros(len(dids), bool)
            gi = np.where(is_gold)[0]; oi = np.where(~is_gold)[0]
            n_other = (target_chunks - len(gi)) if target_chunks is not None else len(oi)
            n_other = max(0, n_other)
            sel = np.concatenate([gi, oi[:n_other]])
            all_doc_ids = ids[sel]; all_doc_masks = masks[sel]; all_corpus_doc_ids = dids[sel]
            if jax.process_index() == 0:
                print(f"  corpus (scanned {len(dids)}): {len(gi)} gold + {min(n_other, len(oi))} distractor chunks = {len(sel)} total (target={target_docs})")
        else:
            all_doc_ids, all_doc_masks, all_corpus_doc_ids, doc_count = [], [], [], 0
            for ids_batch, masks_batch, doc_id_batch in doc_dataset.generator():
                all_doc_ids.append(ids_batch)
                all_doc_masks.append(masks_batch)
                all_corpus_doc_ids.append(doc_id_batch)
                doc_count += ids_batch.shape[0]
                if max_chunks is not None and doc_count >= max_chunks:
                    break

            all_doc_ids = np.concatenate(all_doc_ids)
            all_doc_masks = np.concatenate(all_doc_masks)
            all_corpus_doc_ids = np.concatenate(all_corpus_doc_ids)
            if max_chunks is not None:
                all_doc_ids = all_doc_ids[:max_chunks]
                all_doc_masks = all_doc_masks[:max_chunks]
                all_corpus_doc_ids = all_corpus_doc_ids[:max_chunks]

        aux_loss_cfg_dict = unfreeze_dict(aux_loss_config) if aux_loss_config is not None else {}
        compute_acc = (
            bool(aux_loss_cfg_dict.get('doc_access_acc', {}).get('enabled', False))
            or bool(self.cfg.get('doc_access_acc', False))
        )

        # Fallback token-byte lookup for datasets that predate the pos_doc_ids field.
        # When pos_doc_ids is present in the batch, this is not used.
        corpus_chunk_lookup = {}
        if compute_acc:
            if jax.process_index() == 0:
                print("  Building corpus chunk lookup for doc_access_acc (fallback)...")
            for i in range(all_doc_ids.shape[0]):
                corpus_chunk_lookup[all_doc_ids[i].tobytes()] = i

        if jax.process_index() == 0:
            print(f"  Collected {all_doc_ids.shape[0]} doc chunks, embedding...")

        BENCH = os.environ.get("MEMBENCH")
        # MEMBENCH times _gen() + the prompt prefill on batch 0 ONLY (it break()s after the reps),
        # and batch 0 is the SHORTEST prompt under shuffle=false — so it systematically
        # under-reports generation. MEMBENCH_PHASES=1 instead times each phase of EVERY batch at
        # real (varying) prompt lengths, attributing the loop's wall-clock instead of inferring it
        # by subtraction. Writes bench_phases.json next to MEMBENCH's output. No break, no reps.
        BENCH_PHASES = os.environ.get("MEMBENCH_PHASES") == "1"
        phase_rows = []
        doc_tokens_valid = int(all_doc_masks.astype(bool).sum())
        t_enc0 = time.perf_counter()
        # Embed — uses all devices data-parallel (fsdp = jax.device_count() in utils)
        mem_k_np, mem_v_np, mem_mask_np = embed_documents_tokenized(
            embed_fn, all_doc_ids, all_doc_masks, embed_batch_size
        )
        t_encode = time.perf_counter() - t_enc0

        # eff_doc_len: number of flat memory vectors per corpus chunk (1 when no conv
        # stride, but may differ with conv stride > 1).
        total_corpus_chunks = all_doc_ids.shape[0]
        eff_doc_len = mem_k_np.shape[0] // total_corpus_chunks

        chunk_len = int(all_doc_ids.shape[1])
        mem_top_k = int(cfg.get("main_model", {}).get("mem_top_k",
                        cfg.get("mem_top_k", 0)) or 0)
        bench = {
            "model": "qwen3_mem_embed",
            "dataset": self.cfg.get("output_file", ""),
            "num_corpus_chunks": int(total_corpus_chunks),
            "chunk_len": chunk_len,
            "bank_vectors": int(mem_k_np.shape[0]),
            "eff_doc_len": int(eff_doc_len),
            "doc_tokens_in_memory": doc_tokens_valid,
            "mem_top_k_per_position": mem_top_k,
            # doc-tokens conditioned on per query position in ONE forward.
            # Each memory vector spans chunk_len/eff_doc_len tokens (=1 here, i.e.
            # one K/V vector per doc token, unpooled), so top_k vectors = top_k tokens.
            "tokens_per_mem_vector": int(chunk_len // max(eff_doc_len, 1)),
            "attended_doc_tokens_per_position": int(mem_top_k * (chunk_len // max(eff_doc_len, 1))),
            "encode_corpus_s": round(t_encode, 3),
        }

        # Build doc_id → set of flat memory indices.  All chunks belonging to the
        # same source document are grouped under one key so that matching any chunk
        # of a document includes all of its memory positions in pos_sets.
        doc_id_to_flat_indices = {}
        if compute_acc:
            for chunk_idx in range(total_corpus_chunks):
                doc_id = int(all_corpus_doc_ids[chunk_idx])
                if doc_id not in doc_id_to_flat_indices:
                    doc_id_to_flat_indices[doc_id] = set()
                base = chunk_idx * eff_doc_len
                for off in range(eff_doc_len):
                    doc_id_to_flat_indices[doc_id].add(base + off)

        if jax.process_index() == 0:
            print(f"  mem_k: {mem_k_np.shape}, mem_v: {mem_v_np.shape}")
            print(f"  Pre-normalizing mem_k on CPU (avoids in-JIT copy)...")

        # Pre-normalize mem_k on CPU so rms_norm inside JIT is skipped.
        # This halves peak HBM: device holds only normalized mem_k, not both.
        mem_k_norm_keys = [k for k in model.weights.keys() if k.endswith('mem_k_norm')]
        # GQA per-kv-head bank (mem_k [M,Nkv,H]) is normalized per-head inside the model's
        # mem_lookup_gqa (rms over the H axis with the per-layer [H] weight), so skip the CPU
        # pre-norm optimization for it — the flat [128] weight here can't be picked/broadcast
        # unambiguously across a 3D bank. Only the classic 2D single-bank uses the CPU pre-norm.
        if len(mem_k_norm_keys) > 0 and mem_k_np.ndim == 2:
            mem_k_norm_key = mem_k_norm_keys[0]
            mem_k_norm_w = np.array(model.weights[mem_k_norm_key])
            rms_eps = cfg['main_model'].get('rms_norm_eps', 1e-6)
            sq_mean = np.mean(mem_k_np.astype(np.float32) ** 2, axis=-1, keepdims=True)
            mem_k_np = (mem_k_np.astype(np.float32) / np.sqrt(sq_mean + rms_eps) * mem_k_norm_w).astype(np.float32)
            prenormed = True
            if jax.process_index() == 0:
                print(f"  mem_k pre-normalized on CPU using {mem_k_norm_key}.")
        else:
            prenormed = False
            if jax.process_index() == 0:
                reason = "3D GQA bank -> in-JIT per-head norm" if mem_k_np.ndim == 3 else "mem_k_norm not found"
                print(f"  Skipping CPU pre-norm ({reason}).")

        if jax.process_index() == 0:
            print(f"  Freeing embed model weights from device memory...")

        # Save reference before freeing so we can restore for subsequent evals
        embed_w_saved = embed_w

        # Drop all references to embed weights so JAX can GC device buffers
        del embed_fn
        del embed_w
        _main_w = None
        # Also strip embed_model.* from model.weights — the single largest
        # source of HBM waste before phase 2
        model.weights = {k: v for k, v in model.weights.items()
                         if k.startswith('main_model.')}
        gc.collect()
        jax.effects_barrier()
        if jax.process_index() == 0:
            print("  Embed model weights freed.")

        # ----------------------------------------------------------------
        # Phase 2 — place memory bank on device, run generation
        #
        # Default: shard mem_k/mem_v across all 8 chips on the 'data' axis
        # (M/n_devices vectors per chip) + cross-device top-k merge, mem_v on CPU.
        # MEM_REPLICATE_BANK=1: replicate the FULL bank on every chip (P(None,None)),
        # mem_v on-device, no cross-device collective — i.e. how the MSA/RAG baselines
        # run (no sharding). Lets us isolate the throughput cost of bank sharding.
        # ----------------------------------------------------------------
        replicate_bank = os.environ.get("MEM_REPLICATE_BANK", "0") == "1"
        if jax.process_index() == 0:
            print(f"Phase 2: placing memory bank on device "
                  f"({'REPLICATED (no sharding)' if replicate_bank else 'sharded on data axis, mem_v on CPU'})...")

        def _make_sharded(arr_np, sharding, dtype=None):
            def cb(idx):
                shard = arr_np[idx]
                return shard.astype(dtype) if dtype is not None else shard
            return jax.make_array_from_callback(arr_np.shape, sharding, cb)

        # Pad to a multiple of n_devices for clean sharding
        M = mem_k_np.shape[0]
        remainder = M % n_devices
        if remainder != 0:
            pad = n_devices - remainder
            # ndim-agnostic pad on axis 0 (bank may be 2D [M,dim] or 3D GQA [M,Nkv,dim]).
            mem_k_np    = np.concatenate([mem_k_np,
                                          np.zeros((pad,) + mem_k_np.shape[1:], dtype=mem_k_np.dtype)])
            mem_v_np    = np.concatenate([mem_v_np,
                                          np.zeros((pad,) + mem_v_np.shape[1:], dtype=mem_v_np.dtype)])
            mem_mask_np = np.concatenate([mem_mask_np,
                                          np.zeros(pad, dtype=mem_mask_np.dtype)])

        Dv = mem_v_np.shape[-1]
        if replicate_bank:
            # Full bank replicated on every chip; lookup is a local full scan with
            # NO cross-device reduce. mem_shard_axis='model' (mesh size 1) makes
            # sharded_top_k_ip dispatch to the replicated path. mem_v lives on-device
            # (no CPU pure_callback) — matching the MSA/RAG no-sharding setup.
            # ndim-agnostic replication (bank may be 2D or 3D GQA [M,Nkv,dim]).
            k_sharding    = NamedSharding(mesh, P(*([None] * mem_k_np.ndim)))
            v_sharding    = NamedSharding(mesh, P(*([None] * mem_v_np.ndim)))
            mask_sharding = NamedSharding(mesh, P(None))

            mem_k_jax    = _make_sharded(mem_k_np,    k_sharding, dtype=jnp.bfloat16)
            mem_v_jax    = _make_sharded(mem_v_np,    v_sharding, dtype=jnp.bfloat16)
            mem_mask_jax = _make_sharded(mem_mask_np, mask_sharding)

            # Ensure no stale CPU mem_v from a prior sharded eval is used.
            from models.retrieval_ops import set_cpu_mem_v
            set_cpu_mem_v(None)

            if jax.process_index() == 0:
                print(f"  mem_k REPLICATED: {mem_k_jax.shape} (full bank on each of "
                      f"{n_devices} chips)")
                print(f"  mem_v on device: {mem_v_np.shape} (bf16, replicated)")
        else:
            # Shard mem_k across all devices (each holds M/n_devices vectors)
            k_sharding    = NamedSharding(mesh, P('data', None))
            mask_sharding = NamedSharding(mesh, P('data'))

            mem_k_jax    = _make_sharded(mem_k_np,    k_sharding, dtype=jnp.bfloat16)
            mem_mask_jax = _make_sharded(mem_mask_np, mask_sharding)

            from models.retrieval_ops import set_cpu_mem_v
            # MEM_DEVICE_V=1: keep mem_v ON-DEVICE (sharded like mem_k), skipping the
            # per-decode-step CPU pure_callback host round-trip. Default: mem_v on CPU
            # (fetched via callback) to keep HBM low for huge corpora.
            device_mem_v = os.environ.get("MEM_DEVICE_V", "0") == "1"
            if device_mem_v:
                set_cpu_mem_v(None)
                mem_v_jax = _make_sharded(mem_v_np, k_sharding, dtype=jnp.bfloat16)
                if jax.process_index() == 0:
                    print(f"  mem_k+mem_v sharded ON-DEVICE: {mem_k_jax.shape} across "
                          f"{n_devices} devices (no CPU callback)")
            else:
                # mem_v stays on CPU as float32 — fetched per-query via pure_callback
                set_cpu_mem_v(mem_v_np.astype(np.float32))
                # Tiny dummy mem_v on device — never used for values (callback handles it)
                mem_v_jax = jax.device_put(
                    jnp.zeros((1, Dv), dtype=jnp.bfloat16),
                    NamedSharding(mesh, P(None, None))
                )
                if jax.process_index() == 0:
                    print(f"  mem_k sharded: {mem_k_jax.shape} across {n_devices} devices "
                          f"({mem_k_jax.shape[0] // n_devices} vectors/device)")
                    print(f"  mem_v on CPU: {mem_v_np.shape} ({mem_v_np.nbytes / 1e9:.2f} GB)")

        # Build params dict: main model weights + memory
        params_with_mem = {k: v for k, v in model.weights.items()
                           if k.startswith('main_model.')}
        params_with_mem["main_model.mem_k"]    = mem_k_jax
        params_with_mem["main_model.mem_v"]    = mem_v_jax
        params_with_mem["main_model.mem_mask"] = mem_mask_jax

        # Shard axis: 'data' (8-way) by default, 'model' (size 1 = replicated) when
        # the full bank is replicated on every chip.
        model.cfg['main_model']['mem_shard_axis'] = 'model' if replicate_bank else 'data'
        if prenormed:
            model.cfg['main_model']['mem_k_prenormed'] = True

        # Chunked retrieval (avoids materialising the full score matrix)
        lookup_chunk_size = self.cfg.get("lookup_chunk_size", None)
        reset_chunk_size = False
        if lookup_chunk_size is not None:
            if jax.process_index() == 0:
                print(f"  Enabling chunked retrieval (lookup_chunk_size={lookup_chunk_size})")
            if model.cfg['main_model'].get("mem_lookup_chunk_size", None) is None:
                reset_chunk_size = True
            model.cfg['main_model']['mem_lookup_chunk_size'] = lookup_chunk_size

        # Register concrete mesh so shard_map uses it inside JIT
        from models.retrieval_ops import set_global_mesh
        set_global_mesh(mesh)
        if jax.process_index() == 0:
            print(f"  Registered mesh for sharded retrieval: {mesh}")

        # ----------------------------------------------------------------
        # Generation loop
        # ----------------------------------------------------------------
        tokenizer = model.tokenizer
        num_samples = self.cfg.get("num_samples", None)

        if jax.process_index() == 0:
            try:
                dev = jax.local_devices()[0]
                ms = dev.memory_stats()
                bytes_in_use = ms.get('bytes_in_use', ms.get('peak_bytes_in_use', None))
                limit = ms.get('bytes_limit', None)
                if bytes_in_use is not None:
                    used_gb = bytes_in_use / 1e9
                    limit_gb = limit / 1e9 if limit else '?'
                    free_gb = (limit - bytes_in_use) / 1e9 if limit else '?'
                    print(f"  [MEM] Before generation: {used_gb:.2f} GB used / {limit_gb:.2f} GB total ({free_gb:.2f} GB free)")
            except Exception as e:
                print(f"  [MEM] Could not read memory stats: {e}")

        if compute_acc:
            @partial(jax.jit, static_argnames=("forward",))
            def _prefill_aux_fn(forward, params, prompt_tokens, pad_mask):
                return forward(prompt_tokens, params, pad_mask=pad_mask, collect_aux=True).aux

        results = []
        acc_scores = []
        telemetry_batches = []
        pos_mass_scores = []            # mem_pos_weight_mass on the answer span, per batch
        gen_hit_scores = []             # decoding example-level recall, per batch
        gen_token_hit_scores = []       # decoding token-level hit rate, per batch
        count = 0
        pbar = tqdm(total=num_samples, desc="Generating (large mem)")
        _mem_printed = False

        for batch_tokens, batch_masks in dataset.generator(num_epochs=1):
            if num_samples is not None and count >= num_samples:
                break
            _t_batch0 = time.perf_counter()

            if not isinstance(batch_tokens, dict):
                batch_tokens = {"batch": batch_tokens}
            raw_batch   = np.array(batch_tokens["batch"])
            batch_mask  = np.array(batch_masks["batch_mask"])
            loss_mask   = np.array(batch_masks["loss_mask"])
            raw_docs    = np.array(batch_tokens["docs"])      if "docs" in batch_tokens else None
            docs_mask   = np.array(batch_masks["docs_mask"])  if "docs_mask" in batch_masks else None
            pos_doc_mask_np = np.array(batch_masks["pos_doc_mask"]) if "pos_doc_mask" in batch_masks else None
            # pos_doc_ids: (B, P) int64 array of corpus doc IDs; -1 = no ID / padding.
            # Present when the QA dataset was augmented with prepare_*_with_ids.py.
            pos_doc_ids_np = np.array(batch_masks["pos_doc_ids"]) if "pos_doc_ids" in batch_masks else None

            B, T = raw_batch.shape

            prompt_ends, gt_answer_ids = [], []
            for i in range(B):
                ans_pos = np.where(loss_mask[i] > 0)[0]
                pe = int(ans_pos[0]) if len(ans_pos) > 0 else T
                prompt_ends.append(pe)
                gt_answer_ids.append(raw_batch[i, ans_pos].tolist() if len(ans_pos) > 0 else [])

            max_prompt_len = max(prompt_ends)
            pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id)

            prompt_tokens   = np.full((B, max_prompt_len), pad_id, dtype=raw_batch.dtype)
            prompt_pad_mask = np.zeros((B, max_prompt_len), dtype=np.bool_)
            for i in range(B):
                pe = prompt_ends[i]
                offset = max_prompt_len - pe
                prompt_tokens[i, offset:]   = raw_batch[i, :pe]
                prompt_pad_mask[i, offset:] = batch_mask[i, :pe].astype(np.bool_)

            prompt_jax = jax.device_put(jnp.array(prompt_tokens),   P('data', None))
            pmask_jax  = jax.device_put(jnp.array(prompt_pad_mask), P('data', None))

            def _gen():
                return _generate_tokens(
                    model.forward,
                    model.init_kv,
                    params_with_mem,
                    prompt_jax,
                    self.cfg.get("max_new_tokens", 64),
                    pad_mask=pmask_jax,
                    temperature=self.cfg.get("temperature", 0.0),
                    top_k=self.cfg.get("top_k", 20),
                    top_p=self.cfg.get("top_p", 0.8),
                )

            t_gen0 = time.perf_counter()
            gen_tokens = _gen()
            jax.block_until_ready(gen_tokens)
            t_gen = time.perf_counter() - t_gen0

            # `and not BENCH_PHASES`: this block break()s after batch 0, which is exactly what
            # PHASES exists to avoid (batch 0 is the shortest prompt under shuffle=false).
            if BENCH and not BENCH_PHASES:
                # Repeat on THIS (natural-shape) batch: iter 0 above was JIT warmup;
                # the reps below reuse the compiled program -> clean steady-state
                # timing (avoids per-batch recompiles AND the OOM from forcing a
                # larger fixed prompt length). One batch is enough; then stop.
                B_b = int(prompt_tokens.shape[0])
                reps = int(os.environ.get("BENCH_REPS", "4"))
                for _ in range(reps):
                    t_g = time.perf_counter()
                    g = _gen(); jax.block_until_ready(g)
                    t_g = time.perf_counter() - t_g
                    t_p = time.perf_counter()
                    _pf = _prefill_aux_fn(model.forward, params_with_mem, prompt_jax, pmask_jax)
                    jax.block_until_ready(_pf if _pf is not None else g)
                    t_p = time.perf_counter() - t_p
                    bench.setdefault("batches", []).append({
                        "B": B_b, "gen_e2e_s": round(t_g, 4),
                        "prefill_s": round(t_p, 4),
                        "prompt_len": int(max_prompt_len),
                        "max_new_tokens": int(self.cfg.get("max_new_tokens", 64)),
                    })
                os.makedirs(BENCH, exist_ok=True)
                with open(os.path.join(BENCH, "bench_membed.json"), "w") as f:
                    json.dump(bench, f, indent=2)
                print(f"[MEMBED][BENCH] wrote bench (reps={reps}, prompt_len={max_prompt_len})", flush=True)
                break

            gen_tokens_np = np.array(
                jax.experimental.multihost_utils.process_allgather(gen_tokens, tiled=True)
            )

            # -- doc_access_acc for large-mem (corpus-level lookup) ---------
            # Prefer ID-based matching (no text matching required) when the QA
            # dataset carries pos_doc_ids.  Fall back to token-byte matching for
            # datasets that predate the augmentation scripts.
            has_pos_ids = (
                pos_doc_ids_np is not None
                and np.any(pos_doc_ids_np >= 0)
            )
            can_use_token_fallback = (raw_docs is not None and pos_doc_mask_np is not None)

            pos_sets = None
            if compute_acc and (has_pos_ids or can_use_token_fallback):
                pos_sets = []
                for i in range(B):
                    pos_flat = set()
                    if has_pos_ids:
                        for doc_id in pos_doc_ids_np[i]:
                            if doc_id >= 0:
                                pos_flat.update(doc_id_to_flat_indices.get(int(doc_id), set()))
                    elif can_use_token_fallback:
                        M_docs = pos_doc_mask_np.shape[1]
                        raw_docs_3d = raw_docs.reshape(B, M_docs, -1)
                        for m in range(M_docs):
                            if pos_doc_mask_np[i, m]:
                                key = raw_docs_3d[i, m].tobytes()
                                if key in corpus_chunk_lookup:
                                    chunk_idx = corpus_chunk_lookup[key]
                                    doc_id = int(all_corpus_doc_ids[chunk_idx])
                                    pos_flat.update(doc_id_to_flat_indices.get(doc_id, set()))
                    pos_sets.append(pos_flat)

            batch_doc_access_acc = [None] * B
            batch_doc_hit_rate = [None] * B
            batch_doc_token_hit_rate = [None] * B
            batch_pos_weight_mass = [None] * B

            _t_auxp0 = time.perf_counter()
            aux_data = _prefill_aux_fn(model.forward, params_with_mem, prompt_jax, pmask_jax)
            if BENCH_PHASES:
                jax.block_until_ready(aux_data) if aux_data is not None else None
            _t_auxp = time.perf_counter() - _t_auxp0
            _t_auxgen = 0.0
            if aux_data is not None and aux_data.get("mem_top_k_indices"):
                # NOTE: doc_access_acc + mem telemetry are computed below on the
                # generated-answer forward (aux_gen), masked to the answer span, to
                # match training's answer-position measurement (mode B). This prompt
                # prefill only feeds the per-example fallback / prefill timing.
                indices_np = np.array(
                    jax.experimental.multihost_utils.process_allgather(
                        aux_data["mem_top_k_indices"][0], tiled=True
                    )
                )[:B]
                
                # (prompt-prefill doc_access removed — computed on answer span below)

                # -- generation-time retrieval curve -------------------------
                # Run one extra forward pass on (prompt + generated tokens) to
                # capture what the model looked up at each decode step.
                gen_np   = gen_tokens_np[:B]                         # (B, max_new_tokens)
                S_prompt = prompt_tokens.shape[1]
                full_seq = np.concatenate([prompt_tokens, gen_np], axis=1)
                full_mask = np.concatenate(
                    [prompt_pad_mask,
                     np.ones((B, gen_np.shape[1]), dtype=prompt_pad_mask.dtype)],
                    axis=1,
                )
                full_seq_jax  = jax.device_put(jnp.array(full_seq),  P('data', None))
                full_mask_jax = jax.device_put(jnp.array(full_mask), P('data', None))
                _t_ag0 = time.perf_counter()
                aux_gen = _prefill_aux_fn(
                    model.forward, params_with_mem, full_seq_jax, full_mask_jax
                )
                if BENCH_PHASES:
                    jax.block_until_ready(aux_gen) if aux_gen is not None else None
                _t_auxgen = time.perf_counter() - _t_ag0
                if aux_gen is not None and aux_gen.get("mem_top_k_indices") and pos_sets is not None:
                    full_indices_np = np.array(
                        jax.experimental.multihost_utils.process_allgather(
                            aux_gen["mem_top_k_indices"][0], tiled=True
                        )
                    )[:B]                                            # (B, H, S_full, K)
                    gen_indices_np = full_indices_np[:, :, S_prompt:, :]  # (B, H, max_new_tokens, K)

                    # Per-example generation lengths (tokens before first EOS)
                    max_gen = gen_np.shape[1]
                    gl = np.full(B, max_gen, dtype=np.int32)
                    for i in range(B):
                        eos_pos = np.where(gen_np[i] == tokenizer.eos_token_id)[0]
                        if len(eos_pos) > 0:
                            gl[i] = int(eos_pos[0])

                    # -- doc_access_acc + telemetry on GENERATED-ANSWER positions (mode B) --
                    # answer-span mask over generated tokens (before EOS): measure retrieval
                    # where the model is producing the answer, matching training's positions.
                    gen_active = np.zeros((B, max_gen), dtype=np.float32)
                    for i in range(B):
                        gen_active[i, :gl[i]] = 1.0
                    acc_scores.append(_numpy_doc_access_acc(
                        gen_indices_np, pos_sets, gen_active, mem_mask_np
                    ))
                    batch_doc_access_acc = _numpy_doc_access_acc_per_example(
                        gen_indices_np, pos_sets, gen_active, mem_mask_np
                    )

                    # -- mem_pos_weight_mass on the answer span --------------------------
                    # collect_eval_telemetry below returns entropy/slots/top1 only: it needs a
                    # batch pos_doc_mask for the positive-slot join, which a shared corpus bank
                    # has no analog of. Compute it here from pos_sets instead (see
                    # _numpy_pos_weight_mass). Slice to the generated span ON-DEVICE before the
                    # allgather so only the answer window crosses the host boundary.
                    prob_list = aux_gen.get("mem_top_k_probs")
                    if prob_list:
                        try:
                            _ga = jax.experimental.multihost_utils.process_allgather
                            idx_layers, prob_layers = [], []
                            for li in range(len(prob_list)):
                                idx_layers.append(np.array(_ga(
                                    aux_gen["mem_top_k_indices"][li][:, :, S_prompt:, :], tiled=True))[:B])
                                prob_layers.append(np.array(_ga(
                                    prob_list[li][:, :, S_prompt:, :], tiled=True))[:B])
                            _m = _numpy_pos_weight_mass_layers(
                                idx_layers, prob_layers, pos_sets, gen_active, mem_mask_np)
                            if _m is not None:
                                pos_mass_scores.append(_m)
                            batch_pos_weight_mass = _numpy_pos_weight_mass_per_example_layers(
                                idx_layers, prob_layers, pos_sets, gen_active, mem_mask_np)
                        except Exception as e:
                            print(f"  mem_pos_weight_mass warn: {e}", flush=True)

                    try:
                        from losses.mem_telemetry import collect_eval_telemetry
                        answer_mask_full = np.concatenate(
                            [np.zeros((B, S_prompt), dtype=np.float32), gen_active], axis=1)
                        tel = collect_eval_telemetry(aux_gen, jnp.array(answer_mask_full), None)
                        if tel:
                            telemetry_batches.append(tel)
                    except Exception:
                        pass

                    gen_hit_scores.append(_numpy_doc_hit_rate_gen(
                        gen_indices_np, pos_sets, gl, mem_mask_np
                    ))
                    gen_token_hit_scores.append(_numpy_doc_gen_token_hit_rate(
                        gen_indices_np, pos_sets, gl, mem_mask_np
                    ))
                    batch_doc_hit_rate = _numpy_doc_hit_rate_gen_per_example(
                        gen_indices_np, pos_sets, gl, mem_mask_np
                    )
                    batch_doc_token_hit_rate = _numpy_doc_gen_token_hit_rate_per_example(
                        gen_indices_np, pos_sets, gl, mem_mask_np
                    )

            if not _mem_printed and jax.process_index() == 0:
                _mem_printed = True
                try:
                    dev = jax.local_devices()[0]
                    ms = dev.memory_stats()
                    bytes_in_use = ms.get('bytes_in_use', ms.get('peak_bytes_in_use', None))
                    peak = ms.get('peak_bytes_in_use', None)
                    limit = ms.get('bytes_limit', None)
                    if bytes_in_use is not None:
                        print(f"  [MEM] After first gen batch: {bytes_in_use/1e9:.2f} GB used, peak={peak/1e9:.2f} GB, limit={limit/1e9:.2f} GB, free={(limit-bytes_in_use)/1e9:.2f} GB")
                        print(f"  [MEM] mem_k on device: {mem_k_jax.shape[0]//n_devices} vecs/device = {mem_k_jax.shape[0]//n_devices * mem_k_jax.shape[1] * 2 / 1e9:.3f} GB/device")
                except Exception as e:
                    print(f"  [MEM] Could not read memory stats: {e}")

            actual = min(B, num_samples - count) if num_samples is not None else B
            for i in range(actual):
                eos_positions = np.where(gen_tokens_np[i] == tokenizer.eos_token_id)[0]
                gen_i = (gen_tokens_np[i, :eos_positions[0]]
                         if len(eos_positions) > 0 else gen_tokens_np[i])

                if raw_docs is not None and docs_mask is not None:
                    if pos_doc_mask_np is not None:
                        M_docs = pos_doc_mask_np.shape[1]
                        raw_docs_3d = raw_docs.reshape(B, M_docs, -1)
                        dm_3d       = docs_mask.reshape(B, M_docs, -1)
                        doc_texts = []
                        for d in range(M_docs):
                            if pos_doc_mask_np[i, d]:
                                idx = np.where(dm_3d[i, d] > 0)[0]
                                doc_texts.append(tokenizer.decode(raw_docs_3d[i, d, idx], skip_special_tokens=True))
                        doc_str = " || ".join(doc_texts) if doc_texts else ""
                    else:
                        doc_indices = np.where(docs_mask[i] > 0)[0]
                        doc_str = tokenizer.decode(raw_docs[i, doc_indices], skip_special_tokens=True)
                else:
                    doc_str = ""

                generated = tokenizer.decode(gen_i, skip_special_tokens=True)
                thinking, generated_answer = split_thinking(generated)
                results.append({
                    "prompt":           tokenizer.decode(raw_batch[i, :prompt_ends[i]], skip_special_tokens=False),
                    "generated":        generated,
                    "thinking":         thinking,
                    "generated_answer": generated_answer,
                    "ground_truth":     tokenizer.decode(gt_answer_ids[i], skip_special_tokens=True),
                    "doc":              doc_str,
                    "doc_access_acc":   batch_doc_access_acc[i],
                    "doc_hit_rate":     batch_doc_hit_rate[i],
                    "doc_token_hit_rate": batch_doc_token_hit_rate[i],
                    "pos_slot_weight_mass": batch_pos_weight_mass[i],
                })

            if BENCH_PHASES:
                _t_batch = time.perf_counter() - _t_batch0
                phase_rows.append({
                    "batch": len(phase_rows), "B": int(B), "prompt_len": int(max_prompt_len),
                    "gen_s": round(float(t_gen), 3),
                    "aux_prefill_s": round(_t_auxp, 3),
                    "aux_gen_s": round(_t_auxgen, 3),
                    "other_s": round(_t_batch - float(t_gen) - _t_auxp - _t_auxgen, 3),
                    "batch_total_s": round(_t_batch, 3),
                })
                r = phase_rows[-1]
                print(f"[PHASES] batch {r['batch']} B={r['B']} prompt_len={r['prompt_len']:4d} "
                      f"total={r['batch_total_s']:7.2f}s | gen={r['gen_s']:6.2f} "
                      f"aux_prefill={r['aux_prefill_s']:5.2f} aux_gen={r['aux_gen_s']:6.2f} "
                      f"other={r['other_s']:6.2f}", flush=True)

            count += actual
            pbar.update(actual)

        pbar.close()
        if BENCH_PHASES and jax.process_index() == 0 and phase_rows:
            steady = phase_rows[1:] or phase_rows      # batch 0 carries JIT
            def _med(k):
                v = sorted(r[k] for r in steady)
                return v[len(v) // 2]
            print("\n[PHASES] median over steady-state batches (batch 0 excluded — JIT):",
                  flush=True)
            for k in ("gen_s", "aux_prefill_s", "aux_gen_s", "other_s", "batch_total_s"):
                tot = _med("batch_total_s") or 1e-9
                print(f"    {k:16s} {_med(k):7.2f}s  ({100*_med(k)/tot:5.1f}%)", flush=True)
            if BENCH:
                os.makedirs(BENCH, exist_ok=True)
                with open(os.path.join(BENCH, "bench_phases.json"), "w") as f:
                    json.dump(phase_rows, f, indent=2)

        # Cleanup
        set_global_mesh(None)
        from models.retrieval_ops import set_cpu_mem_v
        set_cpu_mem_v(None)
        model.cfg['main_model'].pop('mem_shard_axis', None)
        model.cfg['main_model'].pop('mem_k_prenormed', None)
        if reset_chunk_size:
            model.cfg['main_model']['mem_lookup_chunk_size'] = None

        # Restore embed model weights for subsequent evals in multi-eval runs
        for k, v in embed_w_saved.items():
            model.weights[f'embed_model.{k}'] = v
        del embed_w_saved

        if self.cfg.get("output_file", None):
            if jax.process_index() == 0:
                output_path = self._get_output_path(step, self.cfg.output_file)
                file_metrics = {"generated_count": count}
                if acc_scores:
                    file_metrics["doc_access_acc"] = float(np.mean(acc_scores))
                if telemetry_batches:
                    for tk in set().union(*telemetry_batches):
                        vals = [b[tk] for b in telemetry_batches if tk in b]
                        if vals:
                            file_metrics[tk] = float(np.mean(vals))
                if pos_mass_scores:
                    file_metrics["mem_pos_weight_mass"] = float(np.mean(pos_mass_scores))
                if gen_hit_scores:
                    file_metrics["doc_hit_rate"] = float(np.mean(gen_hit_scores))
                if gen_token_hit_scores:
                    file_metrics["doc_token_hit_rate"] = float(np.mean(gen_token_hit_scores))
                with open(output_path, "w") as f:
                    json.dump({"metrics": file_metrics, "samples": results}, f, indent=2)
                print(f"Saved generations to {output_path}")
                if wandb.run is not None:
                    artifact_name = (
                        f"{wandb.run.id}-eval-{self.key}-step-{step}-results"
                        if step is not None
                        else f"{wandb.run.id}-eval-{self.key}-results"
                    )
                    artifact = wandb.Artifact(name=artifact_name, type="evaluation_results")
                    artifact.add_file(output_path)
                    wandb.log_artifact(artifact)

        if BENCH and jax.process_index() == 0:
            os.makedirs(BENCH, exist_ok=True)
            bpath = os.path.join(BENCH, "bench_membed.json")
            with open(bpath, "w") as f:
                json.dump(bench, f, indent=2)
            print(f"[MEMBED][BENCH] wrote {bpath}", flush=True)

        inference_metrics = {"generated_count": count}
        if acc_scores:
            inference_metrics["doc_access_acc"] = float(np.mean(acc_scores))
        if pos_mass_scores:
            inference_metrics["mem_pos_weight_mass"] = float(np.mean(pos_mass_scores))
        if gen_hit_scores:
            inference_metrics["doc_hit_rate"] = float(np.mean(gen_hit_scores))
        if gen_token_hit_scores:
            inference_metrics["doc_token_hit_rate"] = float(np.mean(gen_token_hit_scores))
        return inference_metrics
