"""
Profile data loading vs TPU compute to identify the bottleneck.

Measures:
  1. Time spent in the Python data pipeline (CPU)
  2. Time spent in the JIT-compiled forward pass (TPU)
  3. Ratio and idle analysis

Usage:
  python scripts/profile_data_vs_compute.py \
    --hf_name "vm2825/nemotron-cc-v21-Parsed-QA4" \
    --batch_size 64 \
    --seq_len 64 \
    --doc_chunk_seq_len 256 \
    --num_chunks_per_doc 2 \
    --num_steps 30
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def profile_data_only(dataset, num_steps):
    """Measure pure data loading time (no TPU involved)."""
    print("\n" + "=" * 70)
    print("PHASE 1: Profiling DATA PIPELINE (CPU only)")
    print("=" * 70)

    times = []
    gen = dataset.generator(num_epochs=1)

    for i in range(num_steps):
        t0 = time.perf_counter()
        try:
            tokens, masks = next(gen)
        except StopIteration:
            print(f"  Dataset exhausted after {i} steps")
            break
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)

        if i < 5 or i % 10 == 0:
            # Print shape info on first few
            if isinstance(tokens, dict):
                batch_shape = np.array(tokens["batch"]).shape
                docs_shape = np.array(tokens["docs"]).shape
                print(f"  Step {i:3d}: {elapsed:.4f}s  |  batch={batch_shape}  docs={docs_shape}")
            else:
                print(f"  Step {i:3d}: {elapsed:.4f}s  |  shape={np.array(tokens).shape}")

    return times


def profile_compute_only(model, dataset, num_steps):
    """Measure forward pass time with pre-loaded data."""
    import jax
    import jax.numpy as jnp
    import optax
    from functools import partial
    from utils import process_train_pairs

    print("\n" + "=" * 70)
    print("PHASE 2: Profiling TPU COMPUTE (pre-loaded data)")
    print("=" * 70)

    # Pre-load all batches first (remove data loading from measurement)
    print("  Pre-loading batches...")
    batches = []
    gen = dataset.generator(num_epochs=1)
    for i in range(num_steps):
        try:
            tokens, masks = next(gen)
            batches.append((tokens, masks))
        except StopIteration:
            break
    print(f"  Pre-loaded {len(batches)} batches")

    @partial(jax.jit, static_argnames=("forward",))
    def forward_only(forward, weights, inputs, input_masks):
        pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        output = forward(inputs, weights, pad_mask=pad_mask)
        return output.logits

    # Warmup JIT
    print("  Warming up JIT (first forward pass)...")
    tokens, masks = batches[0]
    inputs, targets, input_masks, loss_masks = process_train_pairs(tokens, masks)
    _ = forward_only(model.forward, model.weights, inputs, input_masks)
    # Block until done
    jax.block_until_ready(_)
    print("  JIT warmup complete")

    # Now measure
    times = []
    for i, (tokens, masks) in enumerate(batches):
        inputs, targets, input_masks, loss_masks = process_train_pairs(tokens, masks)

        t0 = time.perf_counter()
        logits = forward_only(model.forward, model.weights, inputs, input_masks)
        jax.block_until_ready(logits)  # Force sync — measure actual TPU time
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times.append(elapsed)

        if i < 5 or i % 10 == 0:
            print(f"  Step {i:3d}: {elapsed:.4f}s")

    return times


def profile_end_to_end(model, dataset, num_steps):
    """Measure interleaved data + compute (realistic training loop)."""
    import jax
    import jax.numpy as jnp
    from functools import partial
    from utils import process_train_pairs

    print("\n" + "=" * 70)
    print("PHASE 3: Profiling END-TO-END (data + compute interleaved)")
    print("=" * 70)

    @partial(jax.jit, static_argnames=("forward",))
    def forward_only(forward, weights, inputs, input_masks):
        pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        output = forward(inputs, weights, pad_mask=pad_mask)
        return output.logits

    gen = dataset.generator(num_epochs=1)

    data_times = []
    compute_times = []

    for i in range(num_steps):
        # Data loading
        t_data_start = time.perf_counter()
        try:
            tokens, masks = next(gen)
        except StopIteration:
            print(f"  Dataset exhausted after {i} steps")
            break
        inputs, targets, input_masks, loss_masks = process_train_pairs(tokens, masks)
        t_data_end = time.perf_counter()

        # Compute
        t_compute_start = time.perf_counter()
        logits = forward_only(model.forward, model.weights, inputs, input_masks)
        jax.block_until_ready(logits)
        t_compute_end = time.perf_counter()

        d_time = t_data_end - t_data_start
        c_time = t_compute_end - t_compute_start
        data_times.append(d_time)
        compute_times.append(c_time)

        if i < 5 or i % 10 == 0:
            print(f"  Step {i:3d}: data={d_time:.4f}s  compute={c_time:.4f}s  "
                  f"ratio={'DATA>>TPU' if d_time > c_time * 1.5 else 'TPU>>DATA' if c_time > d_time * 1.5 else 'balanced'}")

    return data_times, compute_times


def print_summary(data_times, compute_times, e2e_data_times, e2e_compute_times):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if data_times:
        # Skip first step (includes dataset init overhead)
        dt = np.array(data_times[1:]) if len(data_times) > 1 else np.array(data_times)
        print(f"\n  Data pipeline (CPU):")
        print(f"    Mean:   {dt.mean():.4f}s")
        print(f"    Median: {np.median(dt):.4f}s")
        print(f"    Std:    {dt.std():.4f}s")
        print(f"    Min:    {dt.min():.4f}s")
        print(f"    Max:    {dt.max():.4f}s")

    if compute_times:
        ct = np.array(compute_times[1:]) if len(compute_times) > 1 else np.array(compute_times)
        print(f"\n  TPU compute (forward only):")
        print(f"    Mean:   {ct.mean():.4f}s")
        print(f"    Median: {np.median(ct):.4f}s")
        print(f"    Std:    {ct.std():.4f}s")
        print(f"    Min:    {ct.min():.4f}s")
        print(f"    Max:    {ct.max():.4f}s")

    if data_times and compute_times:
        dt_mean = np.array(data_times[1:]).mean() if len(data_times) > 1 else np.array(data_times).mean()
        ct_mean = np.array(compute_times[1:]).mean() if len(compute_times) > 1 else np.array(compute_times).mean()
        ratio = dt_mean / (ct_mean + 1e-9)

        print(f"\n  Data / Compute ratio: {ratio:.2f}x")
        print()
        if ratio > 2.0:
            print("  *** VERDICT: CPU DATA PIPELINE IS THE BOTTLENECK ***")
            print(f"  TPU is idle ~{(1 - 1/ratio) * 100:.0f}% of the time waiting for data.")
            print("  Adding more TPU chips won't help — you need faster data loading.")
            print("  Recommendations:")
            print("    - Add prefetching / multiprocess data loading")
            print("    - Reduce redundant tokenization in streaming_qa.py")
            print("    - Pre-tokenize and cache the dataset")
        elif ratio < 0.5:
            print("  *** VERDICT: TPU COMPUTE IS THE BOTTLENECK ***")
            print("  Data loading is fast enough. Scaling chips (with larger batch) should help.")
            print("  Recommendations:")
            print("    - Scale batch size proportionally with chip count")
            print("    - Profile model forward pass for optimization opportunities")
        else:
            print("  *** VERDICT: ROUGHLY BALANCED ***")
            print("  Both data and compute take similar time.")
            print("  Scaling chips should help IF you also scale batch size,")
            print("  but data loading may become the bottleneck at higher chip counts.")

    if e2e_data_times and e2e_compute_times:
        ed = np.array(e2e_data_times[1:]) if len(e2e_data_times) > 1 else np.array(e2e_data_times)
        ec = np.array(e2e_compute_times[1:]) if len(e2e_compute_times) > 1 else np.array(e2e_compute_times)
        print(f"\n  End-to-end per step: {(ed + ec).mean():.4f}s  "
              f"(data: {ed.mean():.4f}s + compute: {ec.mean():.4f}s)")
        print(f"  Throughput: {1.0 / (ed + ec).mean():.1f} steps/sec")

    print()


def main():
    parser = argparse.ArgumentParser(description="Profile data loading vs TPU compute")
    parser.add_argument("--hf_name", type=str, default="vm2825/nemotron-cc-v21-Parsed-QA4")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--doc_chunk_seq_len", type=int, default=256)
    parser.add_argument("--num_chunks_per_doc", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=30, help="Number of steps to profile")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_only", action="store_true", help="Only profile data pipeline (no model needed)")
    args = parser.parse_args()

    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"doc_chunk_seq_len={args.doc_chunk_seq_len}, num_chunks_per_doc={args.num_chunks_per_doc}")

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    from data.qa import QADataset
    print("Loading QADataset...")
    dataset = QADataset(
        tokenizer=tokenizer,
        hf_name=args.hf_name,
        split=args.split,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        doc_chunk_seq_len=args.doc_chunk_seq_len,
        num_chunks_per_doc=args.num_chunks_per_doc,
        provide_docs=True,
        chat_template=True,
        shuffle=False,
    )

    # Phase 1: Data only
    data_times = profile_data_only(dataset, args.num_steps)

    if args.data_only:
        print_summary(data_times, [], [], [])
        return

    # Load model for compute profiling
    import jax
    print(f"\nJAX devices: {jax.device_count()} ({jax.devices()[0].platform})")

    from models import get_model
    from omegaconf import OmegaConf

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "model", "qwen3_mem_embed.yaml")
    model_cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(model_cfg, False)  # Allow dynamic access
    print("Loading model...")
    model = get_model(model_cfg, tp_devices=1)

    # Phase 2: Compute only (pre-loaded data)
    compute_times = profile_compute_only(model, dataset, args.num_steps)

    # Phase 3: End-to-end
    e2e_data, e2e_compute = profile_end_to_end(model, dataset, args.num_steps)

    print_summary(data_times, compute_times, e2e_data, e2e_compute)


if __name__ == "__main__":
    main()
