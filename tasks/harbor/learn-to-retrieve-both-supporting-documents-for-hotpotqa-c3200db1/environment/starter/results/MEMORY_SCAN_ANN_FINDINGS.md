# Memory-layer score-scan @ BS=1: profiling + can ANN beat the brute scan on TPU?

Investigation into the memory layer's per-decode-step bank lookup at **batch size 1, single v6e
chip, no tensor-parallelism**. Question: the O(M) full-bank score-scan dominates decode at large
banks — can a sublinear ANN index (IVF, PQ) cut it on TPU, or is quantizing the brute scan the play?

Bench scripts: `scripts/embed/profile_bs1.py`, `ivf_bench_bs1.py`, `ivf_contig_bench_bs1.py`,
`quant_bench_bs1.py`. All micro-benches: real shapes (Qwen3-4B, mem@L14, k/v_dim 1024, 4 mem heads,
top_k 128), dummy bf16 data, `jax.devices()[0]` (true 1-chip, no TP), median of 30 reps.

## 1. Component profile vs bank size (profile_bs1.py)

Per-decode-step cost, everything constant except bank size M (KV=2560, top_k=128, full un-sharded bank):

| M (vectors) | ~docs | bank (k+v bf16) | score_scan µs | mem TOTAL µs | mem % of step | est tok/s |
|---|---|---|---|---|---|---|
| 128k | ~500  | 0.26 GB | 299  | 1098 | 6.4%  | 58.4 |
| 512k | ~2000 | 1.0 GB  | 831  | 1703 | 9.3%  | 54.4 |
| 1M   | ~3900 | 2.0 GB  | 1516 | 2448 | 12.4% | 50.5 |
| 2M   | ~7800 | 4.0 GB  | 2872 | 3798 | 18.3% | 48.2 |
| 4M   | ~15600| 8.0 GB  | 8053 | 9006 | 34.4% | 38.1 |

- **score_scan grows ~linearly with M** (super-linear past ~8GB) — it's HBM-bandwidth-bound: every
  decode step streams the WHOLE `mem_k` bank. It dominates the memory cost and rises to ~90% of it at 4M.
- **Transformer is flat & M-independent** (~16,000 µs: layers-before ~6k, after+lm_head ~10k). This is
  the 4B-weight-load floor.
- Everything else in the memory layer (q/o_proj, gather, softmax, approx_topk) is ~constant in M —
  they touch top_k=128 items, not M. Lowering top_k does nothing for large banks.

→ The only lever for large banks is **shrinking bytes-scanned-per-step**.

## 2. IVF (partitioning) — FAILS on TPU (ivf_bench_bs1.py, ivf_contig_bench_bs1.py)

k-means partition → score C=√M centroids → probe nprobe clusters → scan only that subset. Tried both
a flat member-id gather into the original bank AND a reordered contiguous `[C,CAP,KD]` block layout.

**Result: IVF is SLOWER than brute at every scale tested** (speedup 0.16–0.76×), despite scanning
11–32× fewer bytes. IVF time is ~flat ~10ms regardless of nprobe → it scans 128–500MB but runs at
**~1% of HBM bandwidth**, while brute scans 4GB at **~1.4 TB/s (near-peak)**.

**Root cause:** the `bank[cluster_ids]` step is a **data-dependent (dynamic) gather**. XLA cannot
lower a runtime-indexed gather to contiguous DMA — even when the blocks are physically contiguous, it
doesn't know the indices select whole rows, so it emits a general gather of tiny non-contiguous loads.
TensorCore HBM transactions are 512-byte granular; scattered row reads waste each transaction and can't
coalesce. Contiguous layout did NOT help (plain `kb[cl]` still compiles to a general gather).

## 3. Quantized contiguous scan — WINS (quant_bench_bs1.py)

Keep the brute contiguous scan (near-peak BW); just scan fewer bytes. Recall = top-128 overlap vs exact
bf16 (quant error is meaningful even on synthetic data).

**int8** (per-row scale, native TPU int8 MXU matmul):

| M | brute µs | int8 µs | speedup | recall@128 |
|---|---|---|---|---|
| 128k | 383 | 294 | 1.30× | 0.976 |
| 1M | 1567 | 884 | 1.77× | 0.972 |
| 4M | 7408 | 3233 | **2.29×** | 0.971 |

**int4** (native int4 matmul):

| M | brute µs | int4 µs | speedup | recall@128 |
|---|---|---|---|---|
| 1M | 1563 | 598 | 2.61× | 0.647 |
| 4M | 7602 | 1732 | **4.39×** | 0.640 |

- **int8 = free ~2× at large banks, recall 0.97, ~0 accuracy cost.** Speedup GROWS with M (pure
  bandwidth: bigger scan → bigger win). This is the recommended memory-specific BS=1 lever. No gather.
- **int4 = up to 4.4×** but recall craters to ~0.64 on synthetic isotropic-gaussian (worst case for
  quantization — no structure for per-row scale). Needs real-embedding recall validation before use.
- **PQ / ADC = DEAD in plain JAX**: 50–190 ms (~100× SLOWER), OOM past 1M. Same dynamic-gather tax —
  the per-code LUT lookup is a runtime gather. ScaNN wins only via a hand-written kernel (in-VMEM LUT +
  SIMD shuffle) that plain XLA can't express.

## 4. Why TPUs fail at dynamic gathers (research)

- The MXU is a **systolic array** — it needs fixed, compile-time-known dataflow. XLA schedules all
  HBM→VMEM DMA ahead of time in large contiguous tiles (~512-byte granularity, dims padded to 8/128).
  Regular contiguous access hits near-peak HBM; runtime-indexed gather can't be prefetched/tiled → falls
  back to a general gather doing tiny non-contiguous loads → effective BW collapses ~100×.
- Google's hardware answer is **SparseCore** (fine-grained 4B/32B DMA + HW gather/scatter, built for
  RecSys embeddings), but it's not exposed from plain JAX/XLA for arbitrary gathers and Pallas SC support
  is nascent/buggy (jax#34640).
- **Escape routes:** (a) *don't gather — compute densely*: TPU-KNN (arXiv 2206.14286) achieves peak-FLOP
  KNN on TPU via dense-matmul brute-force + approx top-k, exactly our brute+approx_max_k primitive;
  (b) one-hot@table matmul (only for tiny tables — why PQ's M×256 one-hot is infeasible);
  (c) **Pallas scalar-prefetch / block-sparse** — prefetch runtime block indices into SMEM, DMA whole
  contiguous blocks (~6× on block-sparse). This is the ONLY way to make IVF pay off on TPU: contiguous
  cluster blocks + prefetched probe list in a custom kernel.

Sources: henryhmko TPU deep dive; TPU-KNN (arXiv 2206.14286); TPU v4 SparseCore (arXiv 2304.01433);
JAX Pallas scalar-prefetch docs; jax#34640; Cloud TPU performance guide.

## Verdict

- **On-TPU, in-HBM: the brute contiguous scan runs at near-peak BW; no gather-based ANN (IVF/PQ) beats
  it in plain JAX/XLA.** Partitioning wins on CPU/GPU or when the corpus lives on disk (that's RAG).
- **Memory-specific BS=1 lever = quantize the contiguous scan.** int8 mem_k = ~2–2.3× at large banks,
  recall 0.97, trivial code. Recommended next step: wire int8 into `models/memory.py` + validate
  recall→LLM-judge accuracy on the 9 MSA evals. int4 is a further ~2× pending real-data recall.
- **Sublinear on TPU is possible but needs a Pallas block-DMA kernel** (scalar-prefetch probe list +
  contiguous cluster blocks) — justified only at 10M+ vector banks. Parked.
