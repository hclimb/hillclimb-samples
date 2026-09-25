# Fixed-workload MODEL1 attention speedup

Use one exclusive NVIDIA B200 and the supplied CUDA development image. No model
weights, text dataset, network connection or server is needed.

From `/environment/starter`:

```sh
bash /environment/starter/solve.sh
python /environment/starter/optifine_public_tests/verifiers/benchmark.py --checkout /environment/starter
```

Build finishes before measurement. The public and private evaluators run the same
paired algorithm, workload and numerical tolerances; only seeds differ. Each
evaluation remeasures the frozen baseline on its allocated GPU.

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

## Reward: linear progress from starter to best observed throughput

`baseline_rate = 384 / median(baseline_seconds_per_call)`

`candidate_rate = 384 / median(candidate_seconds_per_call)`

`ratio = candidate_rate / baseline_rate`

`progress = (ratio - 1) / (1.3515515495176977 - 1)`

`reward = clip((progress - 0.01) / 0.98, 0, 1)`

The baseline is the packaged frozen native FlashMLA implementation in
`utils/incumbent/site`. Each evaluation runs nine baseline/candidate pairs with
identical seeds on the same GPU, alternating which implementation runs first.
Each implementation's rate uses its median amortized call time over those nine
workers. The baseline is remeasured; the fixed upper ratio is the best observed
submission. The roofline does not determine reward.

Progress = (candidate_rate / baseline_rate - 1) / (1.3515515495176977 - 1). Reward = clip((progress - 0.01) / 0.98, 0, 1). The 1% margin is a fraction of the starter-to-best gap at each end: progress up to 0.01 scores 0 and progress from 0.99 scores 1. The frozen starter is remeasured on the same GPU with matching workloads and seeds. Invalid submissions score 0; baseline or infrastructure failures remain evaluator errors.
Raw throughput ratios remain available. Fresh runs can vary slightly.
The estimated roofline is retained only as a diagnostic, not an attainable target
or a reward endpoint. Its calculation remains in `utils/roofline.py`.

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
The private verifier allows 5,400 seconds overall for the build and 18 workers.
Missing cases, samples or numerical checks cannot be silently dropped.
Candidate build/runtime/numerical failures receive zero; recognized infrastructure
failures retain diagnostics without being scored as candidate invalidity.

Default output is `attention-results`; use `--output PATH` to change it.
`--case NAME` is a subset diagnostic, not a full qualifying score.
Reports retain both sets of raw timings and correctness results, measured baseline
and candidate rates, the throughput ratio, and per-implementation roofline diagnostics.

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
