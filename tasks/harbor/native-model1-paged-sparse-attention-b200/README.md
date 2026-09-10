# MODEL1 Attention: Fixed-Workload Roofline Efficiency on B200

Hardware: one B200. Agent budget: six hours (21,600 seconds). Verifier budget: one hour, run afterwards in a separate offline sandbox.

## Task

Speed up a FlashMLA implementation of MODEL1 FP8 paged sparse attention on one scored workload while keeping the kernel correct on every supported setting. The agent works in `/environment/starter`, builds with `solve.sh`, and may redesign CUDA kernels, dispatch, scheduling, memory movement, dequantization, and reductions as long as the native attention API contract holds.

The scored setting is batch 128, query length 3, 128 heads, head dimension 512, with a 128-entry recent window plus 1,024 additional sparse entries per query. Primary and extra page sizes are 256 and 64, and each backing cache holds 32,768 positions per request. MODEL1 stores 448 FP8 E4M3 non-positional values, 64 BF16 positional values, and seven E8M0 scales per cached token; the native implementation dequantizes to BF16 for tensor-core attention.

The starting point is the captured optimized native FlashMLA source, not a naive PyTorch kernel.

## Rules

Specialization for the scored setting is allowed, but all 22 original workloads remain correctness gates: padding, invalid selections, empty rows, sinks, both head families, and changed-value outputs on the same storage. Outputs and input values may not be cached in place of attention computation; the timed block uses previously unseen queries and includes three sampled output snapshots that are checked afterwards. Input-dependent per-call work must stay inside the API.

## Public testing

```
python /environment/starter/optifine_public_tests/verifiers/benchmark.py --checkout /environment/starter
```

Public and private tools use the same configuration and logic with different seeds. `/environment/starter/optifine_public_tests/README.md` gives the complete public contract.

## Scoring

For nominal attention work `F = 4 * B * Q * H * K * D` on the scored setting:

```
ideal_seconds = max(F / 2.25e15, modeled_bytes_per_call / 8e12)
efficiency    = ideal_seconds / median(candidate_block_seconds / 128)
reward        = (efficiency - 0.2938167337239794) / (1 - 0.2938167337239794)
```

The median is taken over nine independently seeded workers on one B200, each running 32 warmups and 128 timed queries. The untouched starter is calibrated to reward 0 and the estimated dense-BF16 roofline to reward 1; results above 1 are reported without clipping, and a valid but slower submission scores negative. Compilation, input generation, references, and warmup are unscored. Any failed correctness case yields reward 0.
