# Fixed-workload MODEL1 attention roofline efficiency

Use one exclusive NVIDIA B200 and the supplied CUDA development image. No model
weights, text dataset, network connection or server is needed.

From `/environment/starter`:

```sh
bash /environment/starter/solve.sh
python /environment/starter/optifine_public_tests/verifiers/benchmark.py --checkout /environment/starter
```

Build finishes before measurement. The public and private evaluators run the same
candidate-only algorithm, workload and numerical tolerances; only seeds differ.
There is no baseline timing in the reward.

## Fixed scored workload

- Batch 128, three query tokens per request, 128 query heads, dimension 512.
- Primary recent window: exactly 128 selected positions, page size 256.
- Additional cache: exactly 1,024 selected positions, page size 64.
- Each cache region has exactly 32,768 positions per request; all selections are valid.
- Finite attention sinks; BF16 queries and the unchanged mixed FP8/BF16 MODEL1 cache.
- Values and equivalent physical page/index realizations vary, not work counts.

The original 22 cases remain mandatory correctness checks. Optimize the scored
setting, but maintain correct behavior across every supported setting, including
64 heads, small batches, long reductions, multiple cache scopes, masked lengths,
sinks, empty rows and unusual page sizes. Their cold times are diagnostics, not reward.
The exact profiles are published in `utils/manifest.json`.

## Reward: starter at zero, estimated roofline at one

For the fixed workload, B=128, Q=3, H=128, K=1152 and D=512.
The two attention matrix products require nominal work F=4*B*Q*H*K*D.
The declared dense-BF16 B200 peak is 2.25e15 FLOP/s and HBM bandwidth is 8e12 byte/s.

`ideal_seconds = max(F / peak_flops, modeled_bytes_per_call / hbm_bytes_per_second)`

`theoretical_query_tokens_per_second = (B * Q) / ideal_seconds`

`efficiency = candidate_query_tokens_per_second / theoretical_query_tokens_per_second`

`anchor = 0.2938167337239794`

`reward = (efficiency - anchor) / (1 - anchor)`

Efficiency is ideal_seconds divided by the median of nine measured amortized call
times. The fixed anchor is the untouched starter's nine-seed public B200 measurement
from September 10, 2026. That measurement maps to zero; fresh starter runs fluctuate
around zero. The estimated roofline maps to one. Valid slower submissions receive
negative rewards; invalid submissions receive zero with valid=0. No baseline is
remeasured for normalization, and neither panel recalibrates the anchor.

This is a fixed, estimated dense-BF16 roofline, not a proven attainable maximum.
The memory model assumes optimistic reuse across heads and calls; detailed counts
and hardware-source links are in `utils/roofline.py`. Dequantization, softmax,
launch overhead and imperfect memory access make the model optimistic. Faster
internal arithmetic remains permitted if correctness passes. Rewards are not
clipped at one; exceeding the estimate is reported for investigation, not treated
as a correctness failure. Neither the ceiling nor its traffic counts are fitted
to candidate or baseline timings.

## Warmup, distinct inputs and timing

Every case gets a cold call and a same-storage query/cache/sink replacement check.
For the scored case, pre-generate 32 warmup queries and 128 separate measured
queries on the GPU before timing. No measured query is used during warmup.
KV data remains resident within a block and changes between seeded workers.
Scheduling metadata may be reused only under the API's shape/length invariants.

The scored block uses a monotonic wall clock and device-wide synchronization at
its boundaries. All 128 eager calls receive distinct queries. Host dispatch,
recurring allocation, all GPU streams and three preallocated output-snapshot
copies are included. Snapshots preserve sampled results even with reused output
buffers. Input generation, compilation, independent references and 32 warmups
are outside the measured block. No full-model serving or concurrent-caller claim
is made. Caching outputs or input values in place of attention computation is invalid.

## Correctness, limits and feedback

All 23 cases must pass cold and same-storage reference comparisons on nine seeds.
Sampled outputs from the measured block are also checked. Preserve output/LSE
shapes, dtypes, native tolerances, nonfinite masks, padding and empty-row behavior.
The independent oracle decodes preserved packed bytes and computes float32 attention.

Worker limits remain 30 seconds for initial warmup, 120 seconds for workload and
180 seconds for the whole process. Candidate build timeout is 1,800 seconds.
Missing cases, samples or numerical checks cannot be silently dropped.
Candidate build/runtime/numerical failures receive zero; recognized infrastructure
failures retain diagnostics without being scored as candidate invalidity.

Default output is `attention-results`; use `--output PATH` to change it.
`--case NAME` is a subset diagnostic, not a full qualifying score.
Reports retain raw times, validity checks, throughput, the roofline calculation
and the anchored reward. A frozen original wheel remains available in
`utils/incumbent` for optional experiments; it is not required by the scorer.

The auxiliary-stream timing diagnostic remains:
`python optifine_public_tests/verifiers/check_timing.py --output timing.json`.

## Automatic public checkpoints

Each public benchmark, including `--case`, snapshots the source before evaluation
and retains the exact compiled package used by every worker. Build first with
`solve.sh`; the benchmark evaluates that completed build. Reports are written to a
unique `/logs/artifacts/public-verifier/<run-id>/results` directory and copied to
your requested output when the run finishes. `checkpoint.json` appears immediately
and links the source checkpoint, retained `kernel/` package and SHA-256 hashes.
Source checkpoints use the normal `/logs/artifacts/progress` restore command.
Reusing an output directory preserves earlier archives, including failed runs.
Archiving is outside scored kernel time.
