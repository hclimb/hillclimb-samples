"""
Tests that GenerationEmbedEvaluator left-pads (right-aligns) prompts correctly.

_generate_tokens takes logits[:, -1, :] after prefill to seed the first
generated token, so position -1 must always be the last *real* token, not a
padding zero.  These tests validate the extraction + padding logic in
evals/generation_embed.py without requiring a model or dataset.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_prompt_arrays(raw_batch, batch_mask, loss_mask):
    """
    Replicates the prompt extraction + left-padding logic from
    GenerationEmbedEvaluator.evaluate(), extracted here for unit testing.
    """
    B, T = raw_batch.shape

    prompt_ends = []
    gt_answer_ids = []
    for i in range(B):
        ans_pos = np.where(loss_mask[i] > 0)[0]
        pe = int(ans_pos[0]) if len(ans_pos) > 0 else T
        prompt_ends.append(pe)
        gt_answer_ids.append(raw_batch[i, ans_pos].tolist() if len(ans_pos) > 0 else [])

    max_prompt_len = max(prompt_ends)

    prompt_tokens   = np.zeros((B, max_prompt_len), dtype=raw_batch.dtype)
    prompt_pad_mask = np.zeros((B, max_prompt_len), dtype=np.bool_)

    for i in range(B):
        pe = prompt_ends[i]
        offset = max_prompt_len - pe
        prompt_tokens[i,   offset:] = raw_batch[i, :pe]
        prompt_pad_mask[i, offset:] = batch_mask[i, :pe].astype(np.bool_)

    return prompt_tokens, prompt_pad_mask, prompt_ends


def test_last_position_always_real():
    """
    For every example, prompt_pad_mask[:, -1] must be True.
    Failing this means _generate_tokens seeds generation from a padding logit.
    """
    T = 10
    B = 4
    raw_batch = np.arange(1, B * T + 1, dtype=np.int32).reshape(B, T)
    batch_mask = np.ones((B, T), dtype=np.int32)

    # Prompt lengths: 8, 5, 7, 3  (answers fill the rest)
    loss_mask = np.zeros((B, T), dtype=np.int32)
    loss_mask[0, 8:] = 1
    loss_mask[1, 5:] = 1
    loss_mask[2, 7:] = 1
    loss_mask[3, 3:] = 1

    _, prompt_pad_mask, _ = build_prompt_arrays(raw_batch, batch_mask, loss_mask)

    assert np.all(prompt_pad_mask[:, -1]), (
        f"prompt_pad_mask[:, -1] = {prompt_pad_mask[:, -1]} — "
        "not all True; some examples have padding at the last position"
    )
    print("PASS: last position is always a real token")


def test_padding_is_at_front():
    """
    Shorter prompts must have their padding zeros at the front (left-aligned
    zeros would be at the back, which is wrong).
    """
    T = 8
    raw_batch = np.array([
        [10, 20, 30, 40, 50, 60, 99, 99],   # prompt_end=6, will be max
        [1,  2,  3,  99, 99, 99, 99, 99],   # prompt_end=3, needs 3 left-pad zeros
    ], dtype=np.int32)
    batch_mask = np.ones((2, T), dtype=np.int32)
    loss_mask  = np.zeros((2, T), dtype=np.int32)
    loss_mask[0, 6:] = 1
    loss_mask[1, 3:] = 1

    prompt_tokens, prompt_pad_mask, _ = build_prompt_arrays(raw_batch, batch_mask, loss_mask)

    # max_prompt_len=6, example 1 has offset=3
    # Positions 0,1,2 should be padding; positions 3,4,5 should be real
    assert not np.any(prompt_pad_mask[1, :3]), (
        "expected first 3 positions of example 1 to be padding"
    )
    assert np.all(prompt_pad_mask[1, 3:]), (
        "expected last 3 positions of example 1 to be real"
    )
    assert np.all(prompt_tokens[1, :3] == 0), (
        "expected padding positions to contain zeros"
    )
    print("PASS: padding is at the front for shorter prompts")


def test_token_values_preserved():
    """
    The actual token IDs must be correctly right-aligned — not shifted or scrambled.
    """
    T = 8
    raw_batch = np.array([
        [10, 20, 30, 40, 50, 60, 99, 99],  # prompt tokens: [10,20,30,40,50,60]
        [11, 22, 33, 44, 99, 99, 99, 99],  # prompt tokens: [11,22,33,44]
    ], dtype=np.int32)
    batch_mask = np.ones((2, T), dtype=np.int32)
    loss_mask  = np.zeros((2, T), dtype=np.int32)
    loss_mask[0, 6:] = 1
    loss_mask[1, 4:] = 1

    prompt_tokens, _, prompt_ends = build_prompt_arrays(raw_batch, batch_mask, loss_mask)

    # max_prompt_len=6
    # Example 0 (offset=0): positions 0..5 = [10,20,30,40,50,60]
    np.testing.assert_array_equal(
        prompt_tokens[0, :], [10, 20, 30, 40, 50, 60],
        err_msg="example 0 token values wrong"
    )
    # Example 1 (offset=2): positions 0,1 = 0 (pad), positions 2..5 = [11,22,33,44]
    np.testing.assert_array_equal(
        prompt_tokens[1, :], [0, 0, 11, 22, 33, 44],
        err_msg="example 1 token values wrong"
    )
    print("PASS: token values correctly right-aligned")


def test_uniform_lengths_no_padding():
    """
    When all prompts are the same length, no padding should be added
    (offset=0 for every example).
    """
    T = 8
    B = 3
    raw_batch = np.arange(1, B * T + 1, dtype=np.int32).reshape(B, T)
    batch_mask = np.ones((B, T), dtype=np.int32)
    loss_mask  = np.zeros((B, T), dtype=np.int32)
    loss_mask[:, 5:] = 1  # all prompt_ends = 5

    _, prompt_pad_mask, _ = build_prompt_arrays(raw_batch, batch_mask, loss_mask)

    assert np.all(prompt_pad_mask), (
        "expected no padding when all prompts have the same length"
    )
    print("PASS: uniform prompt lengths produce no padding")


def test_batch_mask_respected():
    """
    If the dataset already left-pads a sequence (batch_mask=0 at the start),
    those positions must remain masked in prompt_pad_mask even after our
    own left-padding is applied.
    """
    T = 8
    # Example 0: dataset has 2 left-padding tokens, real prompt is tokens 2..5
    # Example 1: no dataset padding, real prompt is tokens 0..5
    raw_batch = np.array([
        [0,  0,  10, 20, 30, 40, 99, 99],
        [1,  2,  3,  4,  5,  6,  99, 99],
    ], dtype=np.int32)
    batch_mask = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.int32)
    loss_mask = np.zeros((2, T), dtype=np.int32)
    loss_mask[:, 6:] = 1  # both prompt_ends = 6, max_prompt_len = 6, offset = 0

    _, prompt_pad_mask, _ = build_prompt_arrays(raw_batch, batch_mask, loss_mask)

    # Example 0: positions 0,1 are dataset-padding → should be False in mask
    assert not prompt_pad_mask[0, 0] and not prompt_pad_mask[0, 1], (
        "dataset-padding positions should remain masked"
    )
    # Example 0: positions 2..5 are real → should be True
    assert np.all(prompt_pad_mask[0, 2:]), (
        "real token positions should be True in mask"
    )
    # Last position of every example is real
    assert np.all(prompt_pad_mask[:, -1]), (
        "last position must always be real"
    )
    print("PASS: existing batch_mask padding is respected")


def main():
    test_last_position_always_real()
    test_padding_is_at_front()
    test_token_values_preserved()
    test_uniform_lengths_no_padding()
    test_batch_mask_respected()
    print("\nAll padding tests passed.")


if __name__ == "__main__":
    main()
