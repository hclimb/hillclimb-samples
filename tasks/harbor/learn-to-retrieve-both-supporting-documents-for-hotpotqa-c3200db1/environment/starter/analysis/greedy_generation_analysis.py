"""
Greedy generation + retrieval-analysis pipeline for qwen3_mem_embed.

Unlike retrieval_analysis.py (which uses teacher forcing), this script runs
true autoregressive greedy decoding.  At each generation step we collect the
memory-layer retrieval indices so we can see what the model actually attends
to when it has to produce each token from scratch.

This script processes true batches, leveraging left-padding so that all
prompts within the batch finish at the same step.
"""

import os
import sys
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()


# =========================================================================
# CLI  (identical flags to retrieval_analysis.py)
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Memory-layer greedy generation analysis")
    p.add_argument("--checkpoint_dir",  default="/home/suhas/memory-layers/outputs/2026-03-22/19-44-33")
    p.add_argument("--model_name",      default="qwen3_mem_embed")
    p.add_argument("--hf_name",         default="vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-evensplit-RQ-8B",
                   help="HuggingFace dataset name")
    p.add_argument("--batch_size",      type=int, default=32,
                   help="Inference batch size")
    p.add_argument("--num_examples",    type=int, default=32,
                   help="Number of examples to inspect in detail")
    p.add_argument("--num_positions",   type=int, default=32,
                   help="Max tokens to generate / trace per example")
    p.add_argument("--context_window",  type=int, default=6,
                   help="Tokens of context around each retrieved position")
    p.add_argument("--skip_first_n",    type=int, default=0,
                   help="Number of examples to skip before taking a batch "
                        "(use multiples of batch_size to skip whole batches)")
    p.add_argument("--step",            type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--split",           default="validation")
    p.add_argument("--output",          default="analysis/greedy_generation_analysis_100k_no_pos.json",
                   help="Path for JSON output (default: misc/greedy_generation_analysis_10k.json)")
    return p.parse_args()


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    max_new_tokens = args.num_positions

    # ------------------------------------------------------------------
    # 1. Load the saved training config
    # ------------------------------------------------------------------
    hydra_cfg_path = os.path.join(args.checkpoint_dir, ".hydra", "config.yaml")
    if not os.path.exists(hydra_cfg_path):
        raise FileNotFoundError(f"No saved training config at {hydra_cfg_path}")
    train_cfg = OmegaConf.load(hydra_cfg_path)
    print("=" * 60)
    print("Training config")
    print("=" * 60)
    print(OmegaConf.to_yaml(train_cfg))

    # ------------------------------------------------------------------
    # 2. JAX init
    # ------------------------------------------------------------------
    from utils import init_jax_distributed; init_jax_distributed()
    print(f"JAX devices: {jax.device_count()}")

    # ------------------------------------------------------------------
    # 3. Build model (arch + HF base weights)
    # ------------------------------------------------------------------
    from models import get_model
    print("Initializing model …")
    model = get_model(train_cfg.model, tp_devices=1)

    # ------------------------------------------------------------------
    # 4. Restore trained checkpoint
    # ------------------------------------------------------------------
    import orbax.checkpoint as ocp
    from utils import load_inference_checkpoint

    ckpt_model_dir = os.path.join(args.checkpoint_dir, args.model_name)
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_model_dir,
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=2),
    )
    step = load_inference_checkpoint(ckpt_mgr, model, step=args.step)
    print(f">> Restored step {step} from {ckpt_model_dir}")

    model.weights = jax.device_put(model.weights)

    # ------------------------------------------------------------------
    # 5. Prepare one batch from the dataset
    # ------------------------------------------------------------------
    from data import get_dataset

    # Extract field_map from training config (check sources first, then top-level)
    field_map = None
    if train_cfg.dataset.get("sources"):
        # Grab field_map from the first source that has one
        for src in train_cfg.dataset.sources.values():
            if src.get("field_map"):
                field_map = dict(src.field_map)
                break
    if field_map is None:
        field_map = dict(train_cfg.dataset.field_map) if train_cfg.dataset.get("field_map") else None

    ds_cfg = OmegaConf.create({
        "name":            "qa",
        "hf_name":         args.hf_name,
        "split":           args.split,
        "batch_size":      args.batch_size,
        "num_workers":     0,
        "shuffle":         True,
        "seq_len":         int(train_cfg.dataset.seq_len),
        "doc_chunk_seq_len": int(train_cfg.dataset.get("doc_chunk_seq_len", train_cfg.dataset.seq_len)),
        "num_chunks_per_doc": int(train_cfg.dataset.get("num_chunks_per_doc", 16)),
        "provide_docs":    True,
        "chat_template":   bool(train_cfg.dataset.get("chat_template", True)),
        "mask_prefix":     bool(train_cfg.dataset.get("mask_prefix", True)),
        "field_map":       field_map,
    })
    print(f"Loading dataset (hf_name={args.hf_name}, split={args.split}, bs={args.batch_size}) …")
    dataset = get_dataset(ds_cfg, model)
    gen = dataset.generator()

    # Skip batches if requested
    batches_to_skip = args.skip_first_n // args.batch_size
    if batches_to_skip > 0:
        print(f"Skipping {batches_to_skip} batches ({batches_to_skip * args.batch_size} examples) …")
        for _ in range(batches_to_skip):
            next(gen)
    print(f"Taking batch at position {batches_to_skip} (examples {batches_to_skip * args.batch_size}–{(batches_to_skip + 1) * args.batch_size - 1})")

    batch_tokens, batch_masks = next(gen)

    # ------------------------------------------------------------------
    # 6. Unpack raw tokens
    # ------------------------------------------------------------------
    raw_batch   = np.array(batch_tokens["batch"])       # (B, T)
    raw_docs    = np.array(batch_tokens["docs"])        # (B*M, doc_len)
    batch_mask  = np.array(batch_masks["batch_mask"])   # (B, T)
    loss_mask   = np.array(batch_masks["loss_mask"])    # (B, T)
    docs_mask   = np.array(batch_masks["docs_mask"])    # (B*M, doc_len)
    pos_doc_mask_np = np.array(batch_masks["pos_doc_mask"]) if "pos_doc_mask" in batch_masks else None

    B        = raw_batch.shape[0]
    T        = raw_batch.shape[1]
    num_docs = raw_docs.shape[0]
    doc_len  = raw_docs.shape[1]
    
    # Calculate M (docs per item)
    M = num_docs // B if num_docs % B == 0 else 1

    print(f"B={B}  T={T}  num_docs={num_docs}  doc_token_len={doc_len}  M={M}")

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 7. For each example: extract the prompt prefix and gt answers
    # ------------------------------------------------------------------
    gt_answers    = []
    prompt_ends   = []   # index of first answer token in raw_batch

    for i in range(B):
        ans_pos = np.where(loss_mask[i] > 0)[0]
        prompt_ends.append(int(ans_pos[0]) if len(ans_pos) > 0 else T)
        if len(ans_pos) > 0:
            gt_answers.append(raw_batch[i, ans_pos].tolist())
        else:
            gt_answers.append([])

    max_prompt_len = max(prompt_ends)

    # Left-pad (right-align) prompts so that position -1 is always the
    # last real token.
    prompt_tokens = np.full((B, max_prompt_len), pad_id, dtype=raw_batch.dtype)
    prompt_pad_mask = np.zeros((B, max_prompt_len), dtype=np.bool_)

    for i in range(B):
        pe = prompt_ends[i]
        offset = max_prompt_len - pe
        prompt_tokens[i, offset:]  = raw_batch[i, :pe]
        prompt_pad_mask[i, offset:] = batch_mask[i, :pe].astype(np.bool_)

    prompt_tokens_jax   = jnp.array(prompt_tokens)
    prompt_pad_mask_jax = jnp.array(prompt_pad_mask)
    docs_jax            = jnp.array(raw_docs)
    docs_mask_jax       = jnp.array(docs_mask.astype(np.bool_))

    # ------------------------------------------------------------------
    # 8. Compute effective doc length (post-conv)
    # ------------------------------------------------------------------
    kern    = int(train_cfg.model.embed_model.get("embed_conv_kernel_size", 1))
    strd    = int(train_cfg.model.embed_model.get("embed_conv_stride", 1))
    use_conv = bool(train_cfg.model.embed_model.get("embed_conv", False))

    if use_conv:
        eff_doc_len = (doc_len - kern) // strd + 1
    else:
        eff_doc_len = doc_len

    total_mem = num_docs * eff_doc_len
    print(f"effective_doc_len={eff_doc_len}  total_memory_vectors={total_mem}")

    # ------------------------------------------------------------------
    # 9. Pre-decode doc tokens for human-readable output
    # ------------------------------------------------------------------
    doc_tok_strs: list[list[str]] = []
    for d in range(num_docs):
        doc_tok_strs.append([tokenizer.decode([int(t)]) for t in raw_docs[d]])

    doc_texts = [
        tokenizer.decode(raw_docs[d].tolist(), skip_special_tokens=True)
        for d in range(num_docs)
    ]

    from models.qwen3_mem_embed import embed_forward, main_forward
    from models.utils import split_weights

    main_cfg  = model.cfg["main_model"]
    embed_cfg = model.cfg["embed_model"]

    # Number of data-parallel devices. We only pad B to be a multiple of N_data.
    N_data = jax.device_count()
    if B % N_data != 0:
         print(f"Warning: Batch size {B} is not divisible by devices {N_data}. Pad if needed.")
         # Here we assume B is a multiple of N_data (e.g. B=8).


    def slice_pos(aux, t_offsets):
        """Slice time position t from prefill aux (B, H, T, K) → (B, H, 1, K).
           t_offsets: list of time offsets (len B) since right-aligned prompts have valid data at their specific ends.
           For left-padded sequences, the last prompt token is ALWAYS at index `max_prompt_len - 1` for all sequences.
           Thus, we just slice at the last position of the auxiliary outputs.
        """
        if not aux:
            return {}
        out = {}
        for key, val_list in aux.items():
            if key == "mem_top_k_indices" or key == "mem_top_k_logits":
                # shape: (B, N, T, K) → slice T dim at -1
                out[key] = [v[:, :, -1:, :] for v in val_list]
            elif key == "mem_scores":
                sliced = []
                for item in val_list:
                    if isinstance(item, tuple):
                        sliced.append(tuple(s[:, -1:, :, :] for s in item))
                    else:
                        sliced.append(item[:, -1:, :, :])
                out[key] = sliced
            else:
                out[key] = val_list
        return out

    # ------------------------------------------------------------------
    # 10. Embed ALL docs once — same for every example
    # ------------------------------------------------------------------
    main_w, embed_w = split_weights(model.weights, ["main_model", "embed_model"])

    docs_for_embed = jax.device_put(
        jnp.array(raw_docs),
        jax.sharding.PartitionSpec('data', None)
    )
    dmask_for_embed = jax.device_put(
        jnp.array(docs_mask.astype(np.bool_)),
        jax.sharding.PartitionSpec('data', None)
    )
    mem_k_all, mem_v_all, mem_mask_all, edl = embed_forward(embed_cfg, docs_for_embed, embed_w, dmask_for_embed)
    eff_doc_len = int(edl)
    print(f"effective_doc_len (post-embed)={eff_doc_len}")

    # Fully replicate the memory bank
    mem_k_rep    = jax.device_put(np.array(mem_k_all),    jax.sharding.PartitionSpec())
    mem_v_rep    = jax.device_put(np.array(mem_v_all),    jax.sharding.PartitionSpec())
    mem_mask_rep = jax.device_put(np.array(mem_mask_all), jax.sharding.PartitionSpec())

    main_w["mem_k"]    = mem_k_rep
    main_w["mem_v"]    = mem_v_rep
    main_w["mem_mask"] = mem_mask_rep


    # ------------------------------------------------------------------
    # 11. Batched Prefill & Decode
    # ------------------------------------------------------------------
    print(f"\nProcessing {B} examples in a single batch …")

    prompt_jax = jax.device_put(prompt_tokens_jax, jax.sharding.PartitionSpec('data', None))
    pmask_jax  = jax.device_put(prompt_pad_mask_jax, jax.sharding.PartitionSpec('data', None))

    # Prefill with KV slots for prompt + generation
    max_kv_len = max_prompt_len + max_new_tokens
    kv_batch = model.init_kv(B, max_kv_len)
    
    # Run prefill (up to max_prompt_len)
    # The pad_mask must match the KV cache second dimension `S`, which is initialized to `max_kv_len`.
    extended_pad_mask_np = np.zeros((B, max_kv_len), dtype=np.bool_)
    extended_pad_mask_np[:, :max_prompt_len] = prompt_pad_mask
    extended_pad_mask_jax = jax.device_put(jnp.array(extended_pad_mask_np), jax.sharding.PartitionSpec('data', None))

    pf_logits, ret_kv, pf_aux = main_forward(
        main_cfg, prompt_jax, main_w,
        pad_mask=extended_pad_mask_jax, 
        kv=kv_batch, pos=0, collect_aux=True
    )

    pf_np     = np.array(pf_logits)              # (B, max_prompt_len, V)
    first_toks = np.argmax(pf_np[:, -1, :], axis=-1)  # (B,)  predictions for the first generated token
    
    # Slice retrieval info at the last token position over the entire batch
    step0_aux = slice_pos(pf_aux, None)

    # Pad ret_kv back up to max_kv_len since main_forward returns KV up to input len
    padded_kv = []
    for lkv in ret_kv:
        pad_len = max_kv_len - lkv.shape[2]
        zeros = jax.device_put(
            jnp.zeros((2, B, pad_len, lkv.shape[3], lkv.shape[4]), dtype=lkv.dtype),
            lkv.sharding
        )
        padded_kv.append(jnp.concatenate([lkv, zeros], axis=2))
    kv_cur = padded_kv

    # Decode loop
    gen_toks_batch = [[first_toks[b]] for b in range(B)]
    # We keep step_auxes per batch
    step_auxes = [step0_aux]

    # Current mask tracks attention
    cur_mask_jax = extended_pad_mask_jax

    next_jax = jax.device_put(
        jnp.array(first_toks[:, None], dtype=raw_batch.dtype),  # (B, 1)
        jax.sharding.PartitionSpec('data', None)
    )

    for step_i in range(1, max_new_tokens):
        # We append to max_prompt_len + step_i - 1 natively in right-aligned sequences
        pos = max_prompt_len + step_i - 1 
        
        # Enable attention for the new position
        cur_mask_jax = cur_mask_jax.at[:, pos].set(True)

        lg, kv_cur, aux = main_forward(
            main_cfg, next_jax, main_w,
            pad_mask=cur_mask_jax, kv=kv_cur, pos=pos, collect_aux=True
        )

        next_ids = np.argmax(np.array(lg)[:, 0, :], axis=-1) # (B,)
        
        for b in range(B):
            gen_toks_batch[b].append(int(next_ids[b]))
            
        step_auxes.append(aux)

        next_jax = jax.device_put(
            jnp.array(next_ids[:, None], dtype=raw_batch.dtype),  # (B, 1)
            jax.sharding.PartitionSpec('data', None)
        )

    # ------------------------------------------------------------------
    # 12. Build results
    # ------------------------------------------------------------------
    all_results = []
    cw = args.context_window

    for ex_i in range(min(args.num_examples, B)):
        pe = prompt_ends[ex_i]
        
        # Grab original (unpadded) prompt
        prompt_ids   = raw_batch[ex_i, :pe].tolist()
        input_text   = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # Ground-truth answer
        tgt_answer = tokenizer.decode(gt_answers[ex_i], skip_special_tokens=True)

        # Predicted answer
        gen_ids = []
        n_steps = len(gen_toks_batch[ex_i])
        for s in range(n_steps):
            tok = int(gen_toks_batch[ex_i][s])
            if tok == eos_token_id:
                break
            gen_ids.append(tok)
        pred_answer = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Formulate doc snippet
        if pos_doc_mask_np is not None:
            # Multi-doc logic
            doc_texts_ex = []
            for d in range(M):
                if pos_doc_mask_np[ex_i, d]:
                    flat_d_i = (ex_i * M) + d
                    doc_texts_ex.append(doc_texts[flat_d_i])
            own_doc_text = " || ".join(doc_texts_ex) if doc_texts_ex else ""
        else:
            own_doc_text = doc_texts[ex_i]

        ans_start = pe
        
        example_result = {
            "example_idx":      ex_i,
            "input_text":       input_text,
            "answer_span":      [ans_start, ans_start + len(gt_answers[ex_i])],
            "target_answer":    tgt_answer,
            "pred_answer":      pred_answer,
            "own_doc_snippet":  own_doc_text,
            "positions":        [],
        }

        print(f"\n--- Example {ex_i} ---")
        print(f"  Prompt  : {input_text[:200]}")
        print(f"  Own doc : {own_doc_text[:120]}…")
        print(f"  Target  : {tgt_answer}")
        print(f"  Predicted: {pred_answer}")
        print()

        # Iterate over generation steps
        # n_steps from loop logic, but slice on generated IDs count
        for step_i in range(len(gen_ids)):
            tok_id     = gen_ids[step_i]
            pred_tok   = tokenizer.decode([tok_id])

            gt_pos = step_i
            if gt_pos < len(gt_answers[ex_i]):
                tgt_tok = tokenizer.decode([gt_answers[ex_i][gt_pos]])
            else:
                tgt_tok = "<pad>"

            aux = step_auxes[step_i]

            top_k_list = (aux.get("mem_top_k_indices", []) if aux else [])
            if not isinstance(top_k_list, list):
                top_k_list = [top_k_list]

            pos_info = {
                "position":        step_i,
                "predicted_token": pred_tok,
                "target_token":    tgt_tok,
                "heads":           {},
            }

            print(f"  step {step_i:3d}  pred={repr(pred_tok):>14s}  tgt={repr(tgt_tok):>14s}")

            if top_k_list:
                top_k_np = np.array(top_k_list[0])  # (B, H, 1, K)
                _, H, T_aux, K = top_k_np.shape

                # Get logits for this batch
                top_k_logits_np = None
                raw_top_k = aux.get("mem_top_k_logits") if aux else None
                if raw_top_k is not None:
                    top_k_logits_np = np.array(raw_top_k[0])  # (B, 1, K) or (B, N, 1, K)? Wait, main_forward returns (B, N, T, K) where B=B_jax. 
                    # If we don't have N rep, we have N_data. 
                    # Actually we passed a proper B through the sharded model, so (B, H, 1, K) is the shape retrieved back from jnp array?
                    # the first dim is batch, so ex_i gives the batch index.

                for h in range(H):
                    flat_idxs = top_k_np[ex_i, h, 0, :]   # (K,)

                    head_weights = [None] * K
                    pre_softmax_weights = [None] * K
                    if top_k_logits_np is not None:
                        # Depends on shape. Typically (B, H, 1, K)
                        raw_lvals = top_k_logits_np[ex_i, h, 0, :].astype(np.float32)
                        pre_softmax_weights = raw_lvals.tolist()
                        lvals = raw_lvals - raw_lvals.max()
                        ew = np.exp(lvals)
                        hw = ew / ew.sum()
                        head_weights = hw.tolist()

                    retrievals = []
                    for ki in range(K):
                        fi  = int(flat_idxs[ki])
                        d_i = fi // eff_doc_len
                        tp  = fi %  eff_doc_len

                        ex_of_doc = d_i // M if pos_doc_mask_np is not None else d_i
                        real_doc  = pos_doc_mask_np[ex_of_doc, d_i % M] if pos_doc_mask_np is not None else True
                        is_own_doc = (ex_of_doc == ex_i) and bool(real_doc)

                        if d_i < num_docs and tp < len(doc_tok_strs[d_i]):
                            tok_str = doc_tok_strs[d_i][tp]
                            c_start = max(0, tp - cw)
                            c_end   = min(len(doc_tok_strs[d_i]), tp + cw + 1)
                            ctx     = "".join(doc_tok_strs[d_i][c_start:c_end])
                            full_doc = doc_texts[d_i]
                        else:
                            tok_str, ctx, full_doc = "<OOB>", "<OOB>", "<OOB>"

                        w_val  = head_weights[ki]
                        pw_val = pre_softmax_weights[ki]
                        retrievals.append({
                            "rank":                ki,
                            "flat_idx":            fi,
                            "doc_idx":             d_i,
                            "tok_pos":             tp,
                            "token":               tok_str,
                            "context":             ctx,
                            "doc_full_text":       full_doc,
                            "weight":              round(w_val,  4) if w_val  is not None else None,
                            "pre_softmax_weight":  round(pw_val, 4) if pw_val is not None else None,
                            "is_own_doc":          is_own_doc,
                        })

                    pos_info["heads"][f"head_{h}"] = {
                        "true_doc_idx": ex_i,
                        "retrievals":   retrievals,
                    }

                    print(f"    head {h}  (true_doc={ex_i}):")
                    for r in retrievals:
                        own = " *OWN*" if r["is_own_doc"] else ""
                        w_s = f"  w={r['weight']:.3f}" if r["weight"] is not None else ""
                        print(f"      [{r['rank']}] doc={r['doc_idx']:3d} pos={r['tok_pos']:3d}"
                              f"  tok={repr(r['token']):>12s}{w_s}"
                              f"  ctx=\"{r['context']}\"{own}")

            example_result["positions"].append(pos_info)

        all_results.append(example_result)

    # ------------------------------------------------------------------
    # 13. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("RETRIEVAL SUMMARY  (greedy generation)")
    print(f"{'='*80}")
    for res in all_results:
        ex  = res["example_idx"]
        own = 0
        tot = 0
        for p in res["positions"]:
            for hk, hdata in p["heads"].items():
                for r in hdata["retrievals"]:
                    tot += 1
                    if r["is_own_doc"]:
                        own += 1
        pct = 100.0 * own / tot if tot > 0 else 0.0
        print(f"  Example {ex}: {own}/{tot} retrievals from own doc ({pct:.1f}%)"
              f"  | target: \"{res['target_answer'][:60]}\""
              f"  | pred: \"{res['pred_answer'][:60]}\"")

    # ------------------------------------------------------------------
    # 14. Save JSON
    # ------------------------------------------------------------------
    out_path = args.output or os.path.join(
        Path(__file__).resolve().parent, "greedy_generation_analysis.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print("\nDone \u2713")


if __name__ == "__main__":
    main()