"""
Standalone inference + retrieval-analysis pipeline for qwen3_mem_embed.

Loads the trained checkpoint, runs 1 batch through the model, and for the first
few examples dumps exactly which documents and token strings were retrieved by
the memory layer at each *answer* token position.

The memory layer works as follows:
  1. The embed model encodes every doc in the batch into per-token key/value
     vectors (optionally compressed by a 1-D conv).  These are flattened into a
     single memory bank: mem_k / mem_v of shape (num_docs * effective_doc_len, dim).
  2. At layer 14 of the main model, a learned query projection produces queries
     from the hidden state.  A top-k lookup into the memory bank retrieves the
     k nearest key vectors.  The corresponding values are mixed back into the
     hidden stream.
  3. Each flat index in that top-k can be decomposed back into (doc_idx, tok_pos)
     via divmod by effective_doc_len.


"""

import os
import sys
import json
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()                             # pick up HF_TOKEN from .env / env

# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Memory-layer retrieval analysis")
    p.add_argument("--checkpoint_dir",  default="/home/suhas/memory-layers/outputs/2026-02-21/05-06-01")
    p.add_argument("--model_name",      default="qwen3_mem_embed")
    p.add_argument("--batch_size",      type=int, default=8,
                   help="Inference batch size (keep small to avoid OOM from aux logits)")
    p.add_argument("--num_examples",    type=int, default=4,
                   help="Number of examples to inspect in detail")
    p.add_argument("--num_positions",   type=int, default=12,
                   help="Max answer-token positions to trace per example")
    p.add_argument("--context_window",  type=int, default=6,
                   help="Tokens of context around each retrieved position")
    p.add_argument("--batch_num",       type=int, default=1,
                   help="Which batch to analyse (1-indexed, default: 1)")
    p.add_argument("--split",           default="validation")
    p.add_argument("--output",          default=None,
                   help="Path for JSON output (default: inf/retrieval_analysis.json)")
    return p.parse_args()


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()

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
    step = load_inference_checkpoint(ckpt_mgr, model)
    print(f">> Restored step {step} from {ckpt_model_dir}")

    # ------------------------------------------------------------------
    # 5. Prepare one batch from the dataset
    #    * Use the SAME data pipeline as training but with small batch &
    #      deterministic ordering so we can map indices back to docs.
    # ------------------------------------------------------------------
    from data import get_dataset

    ds_cfg = OmegaConf.create({
        "name":            "bior",
        "hf_name":         str(train_cfg.dataset.hf_name),
        "split":           args.split,
        "batch_size":      args.batch_size,
        "num_workers":     0,
        "shuffle":         True,
        "seq_len":         int(train_cfg.dataset.seq_len),
        "doc_type":        str(train_cfg.dataset.doc_type),
        "provide_docs":    True,
        "chat_template":   bool(train_cfg.dataset.chat_template),
        "mask_prefix":     bool(train_cfg.dataset.mask_prefix),
        "bio_interval":    int(train_cfg.dataset.bio_interval),
        "num_qa_per_bio":  int(train_cfg.dataset.num_qa_per_bio),
        "bio_limit":       0,
        "qa_limit":        1870,  # just enough for 1 full batch
    })
    print(f"Loading dataset (split={args.split}, bs={args.batch_size}) …")
    dataset = get_dataset(ds_cfg, model)
    gen = dataset.generator()

    # Skip to the requested batch
    for batch_i in range(1, args.batch_num + 1):
        batch_tokens, batch_masks = next(gen)
        if batch_i < args.batch_num:
            continue  # skip
    print(f"Using batch {args.batch_num}")

    # ------------------------------------------------------------------
    # 6. Teacher-forced unpack (same as trainer)
    # ------------------------------------------------------------------
    from utils import process_train_pairs
    inputs, targets, input_masks, loss_masks = process_train_pairs(batch_tokens, batch_masks)

    # Cast masks to bool — create_mask does bitwise AND which requires bool, not float32
    input_masks = {
        "batch_mask": input_masks["batch_mask"].astype(jnp.bool_),
        "docs_mask":  input_masks["docs_mask"].astype(jnp.bool_),
    }

    B         = inputs["batch"].shape[0]
    T         = inputs["batch"].shape[1]         # seq positions (seq_len)
    num_docs  = inputs["docs"].shape[0]          # == B (one doc per example)
    doc_len   = inputs["docs"].shape[1]          # raw doc token length

    print(f"B={B}  T={T}  num_docs={num_docs}  doc_token_len={doc_len}")

    # ------------------------------------------------------------------
    # 7. Forward pass  (collect_aux=True → retrieval indices)
    # ------------------------------------------------------------------
    print("\nForward pass with collect_aux …")
    # Don't wrap in jax.jit — ModelOutput is not a JAX pytree type.
    # The forward pass already uses jax.remat per-layer internally.
    output = model.forward(inputs, model.weights, pad_mask=input_masks, collect_aux=True)

    logits = output.logits                         # (B, T, V)
    aux    = output.aux or {}
    print(f"Logits shape : {logits.shape}")
    print(f"Aux keys     : {list(aux.keys())}")

    # ------------------------------------------------------------------
    # 8. Compute effective doc length (post-conv)
    # ------------------------------------------------------------------
    kern = int(train_cfg.model.embed_model.get("embed_conv_kernel_size", 1))
    strd = int(train_cfg.model.embed_model.get("embed_conv_stride", 1))
    use_conv = bool(train_cfg.model.embed_model.get("embed_conv", False))

    if use_conv:
        eff_doc_len = (doc_len - kern) // strd + 1
    else:
        eff_doc_len = doc_len

    # If effective_doc_len is explicitly in aux, prefer that (it's ground-truth)
    if "effective_doc_len" in aux:
        eff_doc_len = int(aux["effective_doc_len"])

    total_mem = num_docs * eff_doc_len
    print(f"effective_doc_len={eff_doc_len}  total_memory_vectors={total_mem}")

    # ------------------------------------------------------------------
    # 9. Extract retrieval indices & (optional) logits from aux
    # ------------------------------------------------------------------
    # mem_top_k_indices : list[array (B, N_heads, T, K)]  — one per mem layer
    # mem_scores        : list[tuple(array)]               — raw logits
    top_k_list = aux.get("mem_top_k_indices", [])
    if not isinstance(top_k_list, list):
        top_k_list = [top_k_list]

    scores_list = aux.get("mem_scores", [])
    if not isinstance(scores_list, list):
        scores_list = [scores_list]

    if not top_k_list:
        print("ERROR: No mem_top_k_indices found in aux — cannot analyse.")
        return

    # ------------------------------------------------------------------
    # 10. Pre-decode doc tokens for human-readable output
    # ------------------------------------------------------------------
    tokenizer = model.tokenizer
    raw_docs   = np.array(inputs["docs"])       # (B, doc_len)
    raw_batch  = np.array(inputs["batch"])       # (B, T)

    # Per-token string for every doc
    doc_tok_strs: list[list[str]] = []
    for d in range(num_docs):
        doc_tok_strs.append([tokenizer.decode([int(t)]) for t in raw_docs[d]])

    # Full doc text (for summary)
    doc_texts = [tokenizer.decode(raw_docs[d].tolist(), skip_special_tokens=True) for d in range(num_docs)]

    # ------------------------------------------------------------------
    # 11. Analyse each memory layer
    # ------------------------------------------------------------------
    pred_ids_all = np.array(jnp.argmax(logits, axis=-1))   # (B, T)
    target_ids   = np.array(targets)                         # (B, T)
    loss_mask_np = np.array(loss_masks)                      # (B, T)

    all_results = []

    for layer_i, top_k_idx in enumerate(top_k_list):
        top_k_np = np.array(top_k_idx)                      # (B, N, T, K)
        B_, N, T_, K = top_k_np.shape

        # Grab the raw logits for this layer (to compute attention weights)
        raw_logits = None
        if layer_i < len(scores_list):
            # scores_list[layer_i] is a tuple of 1 tensor (standard path)
            # or a tuple of 2 tensors (product-key path); we handle both
            s = scores_list[layer_i]
            if isinstance(s, tuple) and len(s) == 1 and s[0].ndim == 4:
                raw_logits = np.array(s[0])  # (B, T, N, M)  standard path

        print(f"\n{'='*80}")
        print(f"MEMORY LAYER {layer_i}  —  B={B_}  heads={N}  T={T_}  top_k={K}")
        print(f"{'='*80}")

        for ex_i in range(min(args.num_examples, B_)):
            input_ids  = raw_batch[ex_i].tolist()
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            # Find the answer span (loss_mask == 1)
            ans_pos = np.where(loss_mask_np[ex_i] > 0)[0]
            if len(ans_pos) == 0:
                print(f"\n--- Example {ex_i}: no answer positions (loss_mask=0 everywhere) ---")
                continue

            ans_start = int(ans_pos[0])
            ans_end   = int(ans_pos[-1]) + 1

            pred_answer = tokenizer.decode(pred_ids_all[ex_i, ans_start:ans_end].tolist(),
                                           skip_special_tokens=True)
            tgt_answer  = tokenizer.decode(target_ids[ex_i, ans_start:ans_end].tolist(),
                                           skip_special_tokens=True)

            # Which doc is "my own doc" for this example
            own_doc_text = doc_texts[ex_i]

            example_result = {
                "example_idx":  ex_i,
                "input_text":   input_text,
                "answer_span":  [ans_start, ans_end],
                "target_answer": tgt_answer,
                "pred_answer":   pred_answer,
                "own_doc_snippet": own_doc_text,
                "positions":    [],
            }

            print(f"\n--- Example {ex_i} ---")
            print(f"  Input   : {input_text[:200]}")
            print(f"  Own doc : {own_doc_text}…")
            print(f"  Target  : {tgt_answer}")
            print(f"  Predicted: {pred_answer}")
            print()

            # Iterate over answer positions only
            positions_to_show = ans_pos[:args.num_positions]
            for t in positions_to_show:
                t = int(t)
                pred_tok = tokenizer.decode([int(pred_ids_all[ex_i, t])])
                tgt_tok  = tokenizer.decode([int(target_ids[ex_i, t])])

                # Compute per-head attention weights over top-k from raw logits
                head_weights = {}
                if raw_logits is not None:
                    for h in range(N):
                        flat_idxs = top_k_np[ex_i, h, t, :]          # (K,)
                        lvals = np.array([raw_logits[ex_i, t, h, fi] for fi in flat_idxs])
                        # softmax over top-k
                        lvals = lvals - lvals.max()
                        ew = np.exp(lvals.astype(np.float32))
                        head_weights[h] = (ew / ew.sum()).tolist()

                pos_info = {
                    "position":         t,
                    "predicted_token":  pred_tok,
                    "target_token":     tgt_tok,
                    "heads":            {},
                }

                print(f"  pos {t:3d}  pred={repr(pred_tok):>14s}  tgt={repr(tgt_tok):>14s}")

                cw = args.context_window
                for h in range(N):
                    retrievals = []
                    for ki in range(K):
                        fi  = int(top_k_np[ex_i, h, t, ki])
                        d_i = fi // eff_doc_len
                        tp  = fi %  eff_doc_len

                        if d_i < num_docs and tp < len(doc_tok_strs[d_i]):
                            tok_str  = doc_tok_strs[d_i][tp]
                            c_start  = max(0, tp - cw)
                            c_end    = min(len(doc_tok_strs[d_i]), tp + cw + 1)
                            ctx      = "".join(doc_tok_strs[d_i][c_start:c_end])
                        else:
                            tok_str, ctx = "<OOB>", "<OOB>"

                        w_val = head_weights.get(h, [None]*K)[ki]
                        retrievals.append({
                            "rank":      ki,
                            "flat_idx":  fi,
                            "doc_idx":   d_i,
                            "tok_pos":   tp,
                            "token":     tok_str,
                            "context":   ctx,
                            "weight":    round(w_val, 4) if w_val is not None else None,
                            "is_own_doc": d_i == ex_i,
                        })

                    pos_info["heads"][f"head_{h}"] = {
                        "true_doc_idx": ex_i,
                        "retrievals":   retrievals,
                    }

                    # Pretty-print
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
    # 12. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("RETRIEVAL SUMMARY")
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
              f"  | target: \"{res['target_answer'][:60]}\"")

    # ------------------------------------------------------------------
    # 13. Save JSON
    # ------------------------------------------------------------------
    out_path = args.output or os.path.join(Path(__file__).resolve().parent, "retrieval_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # ------------------------------------------------------------------
    # 14. Quick argmax-decode vs ground truth for answer span
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("ARGMAX DECODE vs GROUND TRUTH  (answer span)")
    print(f"{'='*80}")
    for ex_i in range(min(args.num_examples, B)):
        ans_pos = np.where(loss_mask_np[ex_i] > 0)[0]
        if len(ans_pos) == 0:
            continue
        s, e = int(ans_pos[0]), int(ans_pos[-1]) + 1
        pred_txt = tokenizer.decode(pred_ids_all[ex_i, s:e].tolist(), skip_special_tokens=True)
        tgt_txt  = tokenizer.decode(target_ids[ex_i, s:e].tolist(),   skip_special_tokens=True)
        q_txt    = tokenizer.decode(raw_batch[ex_i].tolist(),          skip_special_tokens=True)
        print(f"\n  Ex {ex_i}:")
        print(f"    Q   : {q_txt[:150]}")
        print(f"    Tgt : {tgt_txt}")
        print(f"    Pred: {pred_txt}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
