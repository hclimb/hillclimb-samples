"""
Exercises lines 102-130 of evals/generation_embed.py with a real dataset
generator and prints intermediate variables so you can visually verify that
the prompt/answer split and left-padding are correct.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer
from data.bior import BioR


def decode_nonpad(tokenizer, ids):
    """Decode, skipping pad tokens."""
    pad = tokenizer.pad_token_id
    return tokenizer.decode([t for t in ids if t != pad])


def print_mask_grid(arr, label, width=80):
    """Print a 1D int/bool array as a compact grid of 0/1."""
    row = "".join(str(int(v)) for v in arr)
    # chunk into groups of 10 for readability
    chunks = [row[i:i+10] for i in range(0, len(row), 10)]
    print(f"  {label}: {' '.join(chunks)}")


def inspect_batch(tokenizer, batch_tokens, batch_masks, example_idx=0):
    raw_batch  = np.array(batch_tokens["batch"])
    batch_mask = np.array(batch_masks["batch_mask"])
    loss_mask  = np.array(batch_masks["loss_mask"])
    B, T = raw_batch.shape

    print(f"\n{'='*70}")
    print(f"Batch shape: {raw_batch.shape}  (B={B}, seq_len={T})")
    print(f"{'='*70}")

    # ------------------------------------------------------------------ #
    # Step 1: replicate lines 102-114 — find prompt_ends and answer ids   #
    # ------------------------------------------------------------------ #
    prompt_ends   = []
    gt_answer_ids = []
    for i in range(B):
        ans_pos = np.where(loss_mask[i] > 0)[0]
        pe = int(ans_pos[0]) if len(ans_pos) > 0 else T
        prompt_ends.append(pe)
        gt_answer_ids.append(raw_batch[i, ans_pos].tolist() if len(ans_pos) > 0 else [])

    max_prompt_len = max(prompt_ends)

    # ------------------------------------------------------------------ #
    # Step 2: replicate lines 122-129 — left-pad the prompt arrays        #
    # ------------------------------------------------------------------ #
    prompt_tokens   = np.zeros((B, max_prompt_len), dtype=raw_batch.dtype)
    prompt_pad_mask = np.zeros((B, max_prompt_len), dtype=np.bool_)
    for i in range(B):
        pe     = prompt_ends[i]
        offset = max_prompt_len - pe
        prompt_tokens[i,   offset:] = raw_batch[i, :pe]
        prompt_pad_mask[i, offset:] = batch_mask[i, :pe].astype(np.bool_)

    # ------------------------------------------------------------------ #
    # Print for every example in the batch                                #
    # ------------------------------------------------------------------ #
    for i in range(B):
        pe     = prompt_ends[i]
        offset = max_prompt_len - pe

        print(f"\n--- Example {i} ---")
        print(f"  seq_len T={T}, prompt_end={pe}, max_prompt_len={max_prompt_len}, left-pad offset={offset}")

        # Raw input
        print(f"\n  [Raw input_ids (first {min(T,40)} positions)]")
        print(f"  {raw_batch[i, :min(T,40)].tolist()}")
        print_mask_grid(batch_mask[i, :min(T,40)], "batch_mask")
        print_mask_grid(loss_mask[i,  :min(T,40)], "loss_mask ")

        # Decoded full sequence
        real_ids = raw_batch[i][batch_mask[i] > 0]
        print(f"\n  [Full decoded sequence (batch_mask==1 tokens)]")
        print(f"  {repr(decode_nonpad(tokenizer, real_ids)[:200])}")

        # Prompt split
        prompt_ids_raw = raw_batch[i, :pe]
        print(f"\n  [Prompt token IDs  (raw_batch[{i}, :{pe}])]")
        print(f"  {prompt_ids_raw[:40].tolist()}")
        print(f"  Decoded: {repr(tokenizer.decode(prompt_ids_raw)[:200])}")

        # Answer split
        print(f"\n  [Answer token IDs  ({len(gt_answer_ids[i])} tokens)]")
        print(f"  {gt_answer_ids[i][:40]}")
        print(f"  Decoded: {repr(tokenizer.decode(gt_answer_ids[i])[:200])}")

        # After left-padding
        print(f"\n  [prompt_tokens[{i}] after left-padding  (shape={prompt_tokens.shape[1]})]")
        print(f"  {prompt_tokens[i, :min(max_prompt_len,40)].tolist()}")
        print_mask_grid(prompt_pad_mask[i, :min(max_prompt_len,40)], "prompt_pad_mask")
        print(f"  Last token ID     : {prompt_tokens[i, -1]}  "
              f"({repr(tokenizer.decode([int(prompt_tokens[i, -1])]))})")
        print(f"  prompt_pad_mask[-1]: {bool(prompt_pad_mask[i, -1])}  "
              f"(must be True — last position is last real token)")

        if not prompt_pad_mask[i, -1]:
            print(f"  *** FAIL: last position is padding, not a real token! ***")
        if offset > 0 and np.any(prompt_pad_mask[i, :offset]):
            print(f"  *** FAIL: left-padding positions have mask=True ***")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset with provide_docs=True...")
    dataset = BioR(
        tokenizer=tokenizer,
        hf_name="vm2825/squad-formatted",
        split="validation",
        batch_size=4,
        num_workers=0,
        shuffle=False,
        seq_len=256,
        doc_type="bio",
        provide_docs=True,
        chat_template=True,
        mask_prefix=True,
        bio_interval=1,
        num_qa_per_bio=5,
        bio_limit=0,
        qa_limit=4,
    )

    gen = dataset.generator()
    batch_tokens, batch_masks = next(gen)

    inspect_batch(tokenizer, batch_tokens, batch_masks)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
