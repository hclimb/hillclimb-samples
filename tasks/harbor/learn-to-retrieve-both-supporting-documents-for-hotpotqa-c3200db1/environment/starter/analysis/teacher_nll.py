"""
Evaluate NLL of the teacher model (base Qwen3-1.7B with context) on science QA.

The teacher sees [doc | question | answer] as a flat sequence.
We compute NLL only on the answer tokens (teacher_distill_mask).
"""

import os
import sys
import json
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Teacher NLL evaluation on science QA")
    p.add_argument("--hf_ckpt_dir", default="~/weights/huggingface")
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--hf_dataset", default="ragrawal36/nemotron-cc-v21-Parsed-QA4-filtered-1.7B")
    p.add_argument("--split", default="validation")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--doc_chunk_seq_len", type=int, default=256)
    p.add_argument("--num_chunks_per_doc", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--output", default=None, help="Path to save JSON results")
    return p.parse_args()


def main():
    args = parse_args()

    from utils import init_jax_distributed; init_jax_distributed()
    print(f"JAX devices: {jax.device_count()}")

    # ------------------------------------------------------------------
    # 1. Load teacher model (base qwen3)
    # ------------------------------------------------------------------
    from models.qwen3 import forward as qwen3_forward, load as load_qwen3

    print(f"Loading teacher model: {args.model_id} ...")
    teacher = load_qwen3(
        args.model_id,
        tp_devices=1,
        load_weights=True,
        hf_ckpt_dir=args.hf_ckpt_dir,
        mask_type="causal",
    )
    teacher_cfg = teacher.cfg
    teacher_weights = jax.device_put(teacher.weights)
    tokenizer = teacher.tokenizer
    print("Teacher model loaded.")

    # ------------------------------------------------------------------
    # 2. Build dataset (science QA only, distill format for teacher inputs)
    # ------------------------------------------------------------------
    from data.qa import QADataset

    sources = {
        "science": {
            "hf_name": args.hf_dataset,
            "field_map": {"answer": "synthetic_answer"},
            "think_field": "think",
            "teacher_prompt_path": "data/prompts/teacher_summarization_qa.txt",
            "student_prompt_path": "data/prompts/student_summarization_qa.txt",
        }
    }

    dataset = QADataset(
        tokenizer=tokenizer,
        split=args.split,
        seq_len=args.seq_len,
        doc_chunk_seq_len=args.doc_chunk_seq_len,
        num_chunks_per_doc=args.num_chunks_per_doc,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        mask_prefix=True,
        chat_template=True,
        provide_docs=True,
        sources=sources,
        distill=True,
    )

    # ------------------------------------------------------------------
    # 3. JIT-compiled loss function on teacher
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnames=("forward",))
    def teacher_loss_fn(forward, weights, inputs, targets, pad_mask, loss_mask):
        pad_mask_bool = pad_mask.astype(jnp.bool_)
        output = forward(teacher_cfg, inputs, weights, pad_mask=pad_mask_bool)
        logits = output.logits  # [B, T, V]

        preds = jnp.argmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot)  # [B, T]

        batch_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
        per_sample_loss = (loss * loss_mask).sum(axis=-1) / (loss_mask.sum(axis=-1) + 1e-9)

        return batch_loss, preds, per_sample_loss

    # ------------------------------------------------------------------
    # 4. Evaluation loop
    # ------------------------------------------------------------------
    nll_scores = []
    samples = []
    step_count = 0

    pbar = tqdm(total=args.eval_steps, desc="Teacher NLL")

    for tokens, masks in dataset.generator(num_epochs=1):
        if step_count >= args.eval_steps:
            break

        # Teacher inputs: teacher_batch[:, :-1] -> teacher_batch[:, 1:]
        teacher_batch = tokens["teacher_batch"]          # [B, T_teacher]
        teacher_attn  = masks["teacher_mask"]            # [B, T_teacher]
        teacher_distill_mask = masks["teacher_distill_mask"]  # [B, T_teacher]

        inputs    = teacher_batch[:, :-1]
        targets   = teacher_batch[:, 1:]
        pad_mask  = teacher_attn[:, :-1]
        loss_mask = teacher_distill_mask[:, 1:]

        batch_loss, preds, per_sample_loss = teacher_loss_fn(
            qwen3_forward,
            teacher_weights,
            inputs,
            targets,
            pad_mask,
            loss_mask,
        )

        nll_scores.append(float(batch_loss))

        # Gather for per-sample output
        batch_preds   = np.array(jax.experimental.multihost_utils.process_allgather(preds, tiled=True))
        batch_targets = np.array(jax.experimental.multihost_utils.process_allgather(targets, tiled=True))
        batch_lmask   = np.array(jax.experimental.multihost_utils.process_allgather(loss_mask, tiled=True))
        per_sample_nlls = np.array(jax.experimental.multihost_utils.process_allgather(per_sample_loss, tiled=True))
        batch_inputs  = np.array(jax.experimental.multihost_utils.process_allgather(inputs, tiled=True))

        for b in range(batch_preds.shape[0]):
            valid = np.where(batch_lmask[b] > 0)[0]
            if len(valid) == 0:
                continue
            samples.append({
                "nll": float(per_sample_nlls[b]),
                "prompt": tokenizer.decode(batch_inputs[b]),
                "generated": tokenizer.decode(batch_preds[b, valid]),
                "ground_truth": tokenizer.decode(batch_targets[b, valid]),
            })

        step_count += 1
        pbar.update(1)
        pbar.set_postfix(nll=f"{nll_scores[-1]:.4f}")

    pbar.close()

    avg_nll = float(np.mean(nll_scores)) if nll_scores else 0.0
    print(f"\nTeacher NLL on science QA ({args.split}): {avg_nll:.4f}")
    print(f"Steps evaluated: {step_count}, Samples: {len(samples)}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    out_path = args.output or os.path.join(
        Path(__file__).resolve().parent, "teacher_nll_results.json"
    )
    result = {
        "stats": {
            "avg_nll": avg_nll,
            "num_steps": step_count,
            "num_samples": len(samples),
            "model_id": args.model_id,
            "dataset": args.hf_dataset,
            "split": args.split,
            "seq_len": args.seq_len,
            "doc_chunk_seq_len": args.doc_chunk_seq_len,
        },
        "samples": samples,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
