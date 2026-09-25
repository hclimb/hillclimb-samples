# MODEL1 FP8 paged sparse attention on B200

## Task

Optimize FlashMLA's `flash_mla_with_kvcache` attention operator on one NVIDIA B200. The evaluator compares the candidate with a frozen starter wheel remeasured on the same GPU. Reward measures progress from starter performance to the recorded best throughput ratio, with a 1% margin at each end.

The scored workload has batch 128, three query tokens per request, 128 heads and head dimension 512. Each query selects 128 entries from a primary cache with page size 256 and 1,024 from an extra cache with page size 64. Each region has 32,768 positions per request. All selections are valid, queries are BF16, and attention sinks—per-head logits that absorb attention mass through the softmax normalization—are finite. The other supported shapes remain correctness gates.

Work in `/environment/starter` in a no-network container with one B200, 16 CPUs and 131072 MB of memory. The solver window is 3600 seconds; builds have 1800 seconds. Workers have a 30-second initial warmup alarm, a 120-second workload alarm and a 180-second process limit. Private verification allows 5400 seconds overall. The image pins PyTorch 2.11.0 and CUDA 13.0.

This is an attention-operator workload, not complete 32k-context language-model serving. No model checkpoint or text dataset is involved. Concurrent callers are outside the evaluation scope.

## Background

Multi-head latent attention shares a compressed key/value representation across heads. A paged cache stores tokens in fixed-size pages; sparse attention gathers only the positions listed for each query. The kernel converts packed FP8 values into BF16, computes attention with tensor cores, and returns an output and its log-sum-exp (LSE), the normalization value for each query and head.

Each MODEL1 cached token occupies 584 bytes: 448 FP8 E4M3 values without positional encoding, 64 BF16 rotary-position values, seven E8M0 power-of-two scales and one pad byte. The kernel must gather these mixed-format rows, apply their scales and keep the attention arithmetic numerically correct.

The starter is an optimized native FlashMLA implementation captured from SparsePrefillKernel revision `d302d1e622c27b5202efa20ed5d94fc77e862e9f`. Its head-128 decode path uses two cooperating CUDA thread blocks. Separate groups of threads fetch queries, prepare indices and scales, gather cached rows, dequantize them, perform tensor-core multiplies, and update softmax while tracking a running maximum for numerical stability. Shared-memory buffers and barriers coordinate those stages.

Split-KV divides a request's selected tokens among blocks. Each produces a partial output and LSE; a separate combine kernel merges them. Tensor Memory Accelerator (TMA) gathers move rows asynchronously, while programmatic dependent launch (PDL) lets combine work start before the attention kernel has entirely finished. These mechanisms already overlap work, so an optimization must improve their interaction as well as an individual operation.

## Difficulty

The baseline is already tuned. Large recorded improvements came from combined changes to conversion, gathering, buffering and scheduling; individual edits also produced small gains or regressions. Paired measurement reduces sensitivity to GPU-speed differences shared by baseline and candidate, but it is not deterministic. The control measurements in Solution do not establish a general noise bound.

A fast path for the scored shape must preserve all 23 correctness cases, including 64-head variants, unusual page sizes, long reductions, masked lengths, empty rows, padding and sinks. One all-valid specialization passed the full public manifest but exposed a rounding difference in an extra masked, variable-length check. A measured gain does not establish broader numerical correctness.

Earlier private controls without endpoint margins used about 25 GB per worker, completed workload phases in 24–26 seconds against the 120-second alarm, and built within their limit. These observations concern evaluator limits, not whether the solver's working window binds. Full one-hour solver runs and automated release qualification under the current scoring remain pending.

Historical results used superseded scoring rules. The best earlier 15-minute solver was about 1.24% faster than the starter, which did not demonstrate large headroom; those final sources did not complete full public reruns within their work windows. An older direct-scheduling reference regressed and remains a diagnostic. Other early runs were retained provisionally after a provider-budget interruption and were not a completed automated release review. Repeated experiments within one run are not independent replications, and transfer to other versions, devices or workloads is not established.

## Optimization directions

The solver may change kernels, dispatch, scheduling, memory movement, dequantization and reductions, while preserving `flash_mla_with_kvcache(...)`, `get_mla_metadata()`, output/LSE shapes and dtypes, and the numerical contract. The evaluator, workload manifest, oracle, frozen baseline and scoring constants stay fixed. Scheduling metadata may be reused only while the relevant shapes and lengths are unchanged; caching input values or answers in place of attention computation is invalid.

These directions come from recorded attempts on the earlier ratio-scoring version of this workload. Local timings and public/private ratios have different protocols. Combined edits do not isolate component effects, and absence from these reviews does not mean an approach was untried.

- Coordinate conversion, gathers and buffer release. The reference changes all three together. Wider conversion groups were not automatically faster: a 32-thread grouping measured 232.8 µs against 173.5 µs. Arithmetic, thread assignments, row layouts and barriers must work as a complete candidate.
- Evaluate alternative storage and routing as complete candidates. A separate-query-storage lineage using cluster launch control, which assigns work to running clusters, reached 1.332 publicly after multiple edits. A 64-head routing prototype measured 0.734. One prototype does not settle an approach family.
- Reduce unnecessary combine and launch work while retaining useful split-KV parallelism. A compact combine grid with PDL and first-split prefetch reached a public ratio of 1.239; whole-request scheduling measured 0.948 in another lineage. Shared-memory limits also constrain launch choices.
- Prepare indices and addresses ahead of consumption. Asynchronous scale copies with moved consumer waits reached 1.235 publicly. FastDivmod arithmetic and bounded look-ahead helped an intermediate reference artifact, while other look-ahead variants regressed. Confirm the scored workload dispatches to the edited kernel: one subset gain disappeared in the full run.
- Guard mask and reduction specializations. The all-valid fast path with scheduling and split-rescale changes reached 1.249 publicly but exposed a rounding difference in the extra masked, variable-length case noted above. Retain the general path. Packed BF16 scaling experiments had unpaired timings without a full public benchmark.
- Tune buffers, cache hints and waits with resource usage in mind. Four-to-six index buffers measured public ratios of 1.019 and 1.020 on a repeat. An explicit prefetch pipeline measured 280.0 µs against a later 188.0 µs no-prefetch version. More buffers or prefetching do not reliably help.

Ablations omitting KV loads, tensor-core multiplies, dequantization or positional data skip required work. Their lower times are neither valid speedups nor isolated component costs. Missing checks and invalid runs are not throughput measurements.

## Solution

The packaged [solution/solve.sh](solution/solve.sh) installs the exact source from the winning six-hour Astra/Codex submission `KeRjELa`, then invokes the starter build script. Its archive replaces three files beneath `native_paged_sparse_prefill/upstreams/FlashMLA/csrc`: `smxx/decode/combine/combine.cu`, `sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh`, and that directory's `config.h`. [Source hashes](solution/source-hashes.json) pin the files; the archive contains no compiled binaries or model weights.

The head-128 decode kernel adds a fifth 128-thread warpgroup, allowing two groups to share dequantization. Its TMA gathers include the positional bytes in the raw tiles, removing a separate positional-data gather path. Conversion threads stage scales and raw values in registers, then release shared slots earlier so the next fetch can overlap conversion. Coordinate preparation gets separate barriers, FastDivmod page/slot arithmetic and a two-block look-ahead (`AHEAD = 2`); decode index buffers change from four to three. Tensor-core scheduling brings the previous block's value multiply forward when the next key tile is not ready. Prefill branches retain their thread counts.

The combine launch shrinks from `b * s_q` to `min(b, num_sm_parts) * s_q` blocks. For the large-batch path, scheduler metadata identifies partition boundaries and unnecessary blocks return early. These edits were evaluated together; their individual contributions are not isolated.

Fresh private-verifier controls with the endpoint margins:

| Submission | Throughput ratio | Reward | Baseline tokens/s | Candidate tokens/s |
|---|---:|---:|---:|---:|
| Unchanged starter | 1.0007342240224681 | 0 | 2213892.8986052233 | 2215518.391954551 |
| Reference `solve.sh` | 1.3498032134514668 | 1 | 2185802.069999421 | 2950402.6580540864 |

Both controls passed, including the 23 cases and sampled timed outputs across nine seeds. The reference scores one despite falling slightly below the anchor because of the endpoint margin. Its lineage's public ratio was 1.360; the fixed upper anchor comes from its separate private replay. Earlier no-margin controls measured starter/reference ratios of 0.999 and 1.359. These are distinct measurements, not interchangeable scores or a general noise bound. Exact values, trial paths and tested task hashes are in the [calibration record](tests/calibration/timing.json).

## Verification

The public comparison runs from `/environment/starter` after building the edited candidate:

```sh
bash solve.sh
python optifine_public_tests/verifiers/benchmark.py --checkout /environment/starter
```

The starter's `solve.sh` invokes `optifine_public_tests/verifiers/build.py` and writes `.flashmla-build/{site,build.json,build.log}`. The public benchmark uses the existing compiled package at `.flashmla-build/site`; it does not rebuild it. The private `verifiers/verify.py` calls the evaluator with `build_candidate=True`, rebuilding submitted source before measurement.

Each worker checks all 23 cases against an independent float32 oracle: a cold call, then another call after query, cache and sink values are overwritten in the same storage while indices remain unchanged. For the scored case, it prepares 32 warmup queries and 128 distinct measured queries. Three sampled measured outputs, including the last call, must also match the oracle. Both attention output and LSE are checked, including shapes, dtypes, nonfinite values and empty-row behavior. Exact tolerances and case definitions are in the [workload manifest](environment/starter/optifine_public_tests/utils/manifest.json); the [public contract](environment/starter/optifine_public_tests/README.md) describes the protocol.

Device-wide synchronization brackets the timed block. Its wall time divided by 128 is that worker's amortized call time. Timing includes eager dispatch, recurring allocations, all GPU streams and three preallocated output/LSE snapshot copies. Compilation, input generation, oracle computation and warmup occur outside the interval. For each role, aggregation takes the median across nine workers; it is not a median of individually timed calls.

The public tool snapshots the source and retains the tested compiled package. Reports are archived under `/logs/artifacts/public-verifier/<run-id>/results` and copied to `attention-results`, or to `--output PATH` outside that managed archive. `checkpoint.json` links the source snapshot, retained package and hashes. Private reports go to `/logs/verifier`. `--case NAME` is an unscored subset diagnostic.

Failed candidate builds, worker failures, timeouts, numerical mismatches and incomplete or inconsistent reports invalidate a submission and score zero. Baseline or infrastructure failures retain diagnostics and raise an evaluator error instead of producing a usable reward. The candidate executes inside the harness process: fresh queries address ordinary repeated-answer caching, not arbitrary candidate interference with the evaluator. That trust boundary is unchanged.

## Scoring

Every evaluation runs nine baseline/candidate pairs with matching seeds on the same GPU, alternating which implementation runs first. The frozen baseline is remeasured each time. Public seeds are fixed; private evaluation draws nine distinct random seeds. Seeds change input values and equivalent physical page/index realizations, not the amount of work.

```text
baseline_rate = 384 / median(baseline amortized seconds per call)
candidate_rate = 384 / median(candidate amortized seconds per call)
ratio = candidate_rate / baseline_rate
progress = (ratio - 1) / (1.3515515495176977 - 1)
reward = clip((progress - 0.01) / 0.98, 0, 1)
```

Here 384 is batch 128 times three query tokens. The baseline is the frozen wheel under `optifine_public_tests/utils/incumbent/site`; the candidate is the submitted build. The upper anchor is the recorded private replay ratio of the winning submission, not an estimated hardware limit. It stays fixed when controls are rerun, and the replay score was not established as feedback available to the solver.

The margin is 1% of the starter-to-best gap at each end: progress at or below 0.01 scores zero, progress at or above 0.99 scores one, and progress 0.5 still scores 0.5. `reward.json` retains `valid`, `reward`, the unclipped `throughput_ratio`, `baseline_rate` and `candidate_rate`.

The dense-BF16 roofline in `utils/roofline.py` is diagnostic only. It estimates time from declared peak compute and memory bandwidth and excludes softmax, dequantization, scheduling, padding and snapshot overhead. It is neither an attainable target nor a universal ceiling; valid lower-precision implementations may exceed it. A gap to that estimate is not demonstrated optimization headroom. The roofline does not enter the reward formula.
