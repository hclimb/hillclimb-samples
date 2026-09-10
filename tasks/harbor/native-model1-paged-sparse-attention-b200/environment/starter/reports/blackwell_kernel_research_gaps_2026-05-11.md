# Blackwell Kernel Research Gaps - 2026-05-11

## Scope

Focus: practical 2026 NVIDIA Blackwell-class systems for LLM pretraining, prefill, decode, MoE serving, and RL/post-training. I deliberately deprioritized A100 and Hopper-only tuning except when the code exposed a migration gap to Blackwell.

Local repos cloned under `/Users/aumdesai/kernel+infra/repos`:

| Repo | Commit |
|---|---:|
| vLLM | 581b5e9 |
| SGLang | faad02b |
| FlashInfer | 0a128d1 |
| CUTLASS | ae6bccf |
| TensorRT-LLM | 8efeddc |
| TransformerEngine | 0e28953 |
| Megatron-LM | 5e31514 |
| FlashAttention | ab66326 |
| Triton | feb6c04 |
| DeepGEMM | 891d57b |
| DeepEP | b306af0 |
| FlashMLA | 9241ae3 |

## Quick Read

The biggest gaps are no longer "fuse one more epilogue" style wins. The practical opportunities are where kernels, runtime scheduling, precision format, and communication policy interact:

1. Distributed FP4/MXFP4 MoE as a full layer schedule, not isolated grouped GEMMs.
2. Blackwell-native sparse MLA / DeepSeek Sparse Attention across prefill and decode.
3. GPU-resident serving orchestration to remove CPU dispatch/sampling bubbles.
4. Unified FP8/FP4 RL rollout and training precision with fast weight update and KV staleness control.
5. KV cache hierarchy and quantization that does not steal compute from the model.
6. End-to-end NVFP4 pretraining support beyond linear layers.

## Evidence From Current Repos

### 1. MoE backends are fragmented and shape-dependent

vLLM has a long backend ladder for NVFP4 MoE: FlashInfer TRT-LLM, FlashInfer CUTLASS, FlashInfer CuTeDSL, vLLM CUTLASS, Marlin, and emulation. The selector notes that shape-specific runtime fallback can still occur. See:

- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:44`
- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:153`
- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:260`

MXFP4 selection is similarly fragmented across DeepGEMM, FlashInfer TRT-LLM, FlashInfer CUTLASS, Marlin, Triton, AITER, XPU, and emulation:

- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:53`
- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:253`

SGLang's FP4 MoE path routes through FlashInfer TRT-LLM kernels, quantizes activations at runtime, and only supports `silu` and `relu2` activation types:

- `/Users/aumdesai/kernel+infra/repos/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:811`
- `/Users/aumdesai/kernel+infra/repos/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:862`

TensorRT-LLM's public DeepSeek R1 guide shows backend support varies by hardware, precision, and EP degree. For B200/GB200 EP<=8, NVFP4 supports CUTLASS/TRTLLM; for GB200 NVL72 EP>8, NVFP4 uses WIDEEP. Source: https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.html

### 2. Communication-compute overlap is now a first-order MoE problem

DeepEP V2 is explicitly an EP communication library with high-throughput and low-latency all-to-all dispatch/combine, targeting minimal SM usage. It reports SM100 NVLink logical bandwidth around 726/740 GB/s dispatch/combine at 64 SMs and 643/675 GB/s at 24 SMs:

- `/Users/aumdesai/kernel+infra/repos/DeepEP/README.md:3`
- `/Users/aumdesai/kernel+infra/repos/DeepEP/README.md:17`
- `/Users/aumdesai/kernel+infra/repos/DeepEP/README.md:45`

DeepGEMM's new Mega MoE fuses and overlaps EP dispatch, FP8xFP4 linear 1, SwiGLU, FP8xFP4 linear 2, and EP combine in one mega-kernel. It requires multiprocess launch with symmetric memory:

- `/Users/aumdesai/kernel+infra/repos/DeepGEMM/README.md:11`
- `/Users/aumdesai/kernel+infra/repos/DeepGEMM/README.md:114`

This is already beyond normal fusion, but it is still model/layout/runtime specific. That leaves room for a general Blackwell MoE layer scheduler.

### 3. Sparse MLA is promising but still narrow

FlashMLA supports sparse prefill and sparse decoding, with FP8 KV cache for decoding. The README says sparse MLA decoding reaches up to 350 TFLOPS on B200 but is "not really optimized yet"; sparse prefill reaches up to 1450 TFLOPS on B200:

- `/Users/aumdesai/kernel+infra/repos/FlashMLA/README.md:21`
- `/Users/aumdesai/kernel+infra/repos/FlashMLA/README.md:35`
- `/Users/aumdesai/kernel+infra/repos/FlashMLA/README.md:51`

The support matrix remains narrow: dense decode is SM90-only, sparse decode is SM90/SM100 MQA with FP8 KV, dense prefill is SM100 MHA, sparse prefill is SM90/SM100 MQA:

- `/Users/aumdesai/kernel+infra/repos/FlashMLA/README.md:59`
- `/Users/aumdesai/kernel+infra/repos/FlashMLA/tests/test_flash_mla_dense_decoding.py:201`

The interface also has strict layout and shape assumptions:

- Sparse SM100 KV cache must be contiguously valid, not arbitrary disjoint pages: `/Users/aumdesai/kernel+infra/repos/FlashMLA/flash_mla/flash_mla_interface.py:73`
- Sparse decode requires FP8 KV cache: `/Users/aumdesai/kernel+infra/repos/FlashMLA/flash_mla/flash_mla_interface.py:151`
- Sparse prefill has no batch dimension in the public API: `/Users/aumdesai/kernel+infra/repos/FlashMLA/README.md:141`
- SM100 dense prefill backward does not support GQA: `/Users/aumdesai/kernel+infra/repos/FlashMLA/flash_mla/flash_mla_interface.py:282`

vLLM's FlashInfer sparse MLA backend only supports Blackwell SM10.x and requires `qk_nope_head_dim` in `[128, 192]` plus an `index_topk` model config:

- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:100`
- `/Users/aumdesai/kernel+infra/repos/vllm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:116`

### 4. Blackwell attention is bottlenecked by asymmetric hardware, not just memory

FlashAttention-4 frames the Blackwell attention problem as asymmetric hardware scaling: tensor core throughput doubled while shared memory bandwidth and exponential units did not. It reports up to 1.3x over cuDNN 9.13 and 2.7x over Triton on B200 BF16, reaching 1613 TFLOPS/s. Source: https://arxiv.org/abs/2603.05451

PyTorch's FlexAttention FA4 integration states that Blackwell high-performance attention needs deeply pipelined, warp-specialized kernels, TCGEN05/TMEM, and async data movement/matmul; Triton-based implementations cannot express all of this well. Source: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

This means "generic attention compiler" work is only interesting if it can expose Blackwell-specific pipeline structure, not if it just emits a more flexible Triton kernel.

### 5. Host overhead is visible on Blackwell serving

vLLM's Blackwell GPT-OSS optimization note says the CPU host often becomes the bottleneck because it cannot dispatch kernels quickly enough; prepare_batch, scheduling, and sampling create gaps between kernels. vLLM reports async scheduling around 10% gain and stream interval up to 57% end-to-end gain in a high-concurrency benchmark. Source: https://vllm.ai/blog/gpt-oss-optimizations

This is a systems/kernel co-design opportunity: on-device or GPU-resident orchestration may matter more than another fused pointwise op.

### 6. KV offloading/transfer is not a solved "copy faster" problem

vLLM's KV offloading connector post found a custom GPU kernel can be 6% worse at 0% hit rate because it interferes with model computation. Source: https://vllm.ai/blog/kv-offloading-connector

TensorRT-LLM config examples already expose disaggregated context/generation roles, FP8 KV cache, NIXL transfer, MNNVL allreduce, and stream intervals:

- `/Users/aumdesai/kernel+infra/repos/TensorRT-LLM/tests/integration/defs/perf/disagg/test_configs/disagg/perf/deepseek-r1-fp4_8k1k_ctx8_gen1_dep32_bs16_eplb0_mtp3_ccb-NIXL.yaml:45`
- `/Users/aumdesai/kernel+infra/repos/TensorRT-LLM/tests/integration/defs/perf/disagg/test_configs/disagg/perf/deepseek-r1-fp4_128k8k_ctx1_pp8_gen8_tep4_bs4_eplb0_mtp0-Default.yaml:45`

The gap is a policy and scheduling problem: when to transfer, where to place KV, how to overlap without stealing tensor-core or SM resources, and how to account for cache hit probability.

### 7. RL adds weight-update, rollout precision, and staleness constraints

LMSYS' FP8 RL report argues that BF16 training plus FP8 rollout creates train-inference inconsistency, and unified FP8 training plus FP8 rollout reduces TIS clip fraction and train-rollout logprob gap at 30B and 235B MoE scale. It lists future work around quantization error and hiding kernel-launch/quantization latency. Source: https://www.lmsys.org/blog/2025-11-25-fp8-rl/

Megatron-LM's RL utilities show asynchronous rollout collection and explicit logging of policy/KV staleness:

- `/Users/aumdesai/kernel+infra/repos/Megatron-LM/megatron/rl/rl_utils.py:643`
- `/Users/aumdesai/kernel+infra/repos/Megatron-LM/megatron/rl/rl_utils.py:992`

SGLang slime is built around SGLang rollout plus Megatron training, with RL-specific weight update support and partial rollout abort endpoints. Source: https://www.lmsys.org/blog/2025-07-09-slime/

The gap is not just faster inference. RL needs precision consistency, fast recurring weight synchronization, low-latency rollout generation, stale-KV management, and efficient logprob/reward plumbing.

## Ranked Research Opportunities

### 1. Blackwell distributed MoE layer scheduler

Build a full MoE layer runtime that jointly schedules:

- routing/top-k and load-balance metadata;
- EP dispatch/combine;
- FP4/FP8 activation quantization and scale layout conversion;
- two grouped GEMMs;
- activation;
- shared experts;
- output allreduce/reduce-scatter;
- CUDA graph and dynamic batch constraints.

Why it is high value: MoE dominates current frontier serving. Current stacks expose many backends and special cases instead of one robust scheduler. DeepGEMM Mega MoE proves the direction, but it is not yet a general runtime abstraction across vLLM/SGLang/TensorRT-LLM, EP degree, NVLink vs RDMA, prefill vs decode, and model variants.

Research angle: dynamic persistent CLC-based grouped schedules, communication-aware token bin packing, expert hotness prediction, and precision-aware scale layout planning. This is well beyond ordinary fusion.

### 2. Blackwell-native sparse MLA / DSA engine

Build a sparse MLA path that covers:

- sparse prefill and decode;
- MQA/GQA/MHA variants;
- batched sparse prefill;
- arbitrary paged KV layouts, not only contiguously valid sparse SM100 cache;
- FP8 and possibly FP4/MXFP4 KV formats;
- indexer computation plus attention in a coordinated schedule;
- variable top-k and extra KV regions without high CPU rebuild cost.

Why it is high value: long-context DeepSeek-style models can make attention the bottleneck again, especially in prefill and sparse retrieval/indexer paths. FlashMLA's B200 sparse decode is explicitly not fully optimized, and public APIs are narrow.

Research angle: co-design the sparse indexer, KV cache layout, page table conversion, and attention kernel so sparse attention is not just "dense attention over a gathered top-k tensor."

### 3. GPU-resident serving orchestration

Move more decode-loop control to the GPU:

- persistent decode step loop or device-side work queue;
- on-device sampling, logits filtering, and stop checks;
- GPU-side page table/KV metadata updates;
- asynchronous CPU preparation only for coarse events;
- safe integration with CUDA graphs and dynamic batching.

Why it is high value: vLLM reports Blackwell host overhead directly. Async scheduling and stream interval already produce meaningful gains; the next jump is reducing the CPU from per-token coordinator to coarse-grained controller.

Research angle: a device-side scheduler that handles dynamic request arrivals, page allocation, sampling, and kernel selection without giving up the economics of CUDA graphs.

### 4. RL-specific low-precision rollout/training runtime

Build a unified low-precision RL runtime for MoE models:

- FP8/FP4 training and rollout consistency;
- fast quantized weight update into serving workers;
- per-token policy/KV staleness tracking;
- low-latency partial rollout cancellation;
- fused generation-logprob bookkeeping;
- checkpoint/scale synchronization across Megatron, SGLang, and vLLM-like serving engines.

Why it is high value: RL's bottleneck is online generation plus frequent weight updates. Mixed precision can destabilize policy updates. The kernel opportunity is tied to the RL algorithm, not just the transformer layer.

Research angle: precision schemes that trade small rollout bias for large generation throughput, plus kernel/runtime support that makes those schemes measurable and controllable.

### 5. KV cache transfer/quantization policy with minimal compute interference

Build a KV hierarchy runtime:

- FP8/NVFP4 cache formats;
- NVLink-C2C, NIXL/RDMA, CPU/GPU tiers;
- prefetch and eviction policy tied to prefix hit probability;
- copy engines where possible, SM-light kernels where necessary;
- overlap plans that avoid interfering with model kernels.

Why it is high value: long-context serving is memory-placement dominated, but naive GPU-copy kernels can reduce throughput. The open problem is deciding when a transfer is worth it, not just making a faster memcpy.

Research angle: scheduling KV migration as part of the serving policy, with performance models for SM occupancy, memory bandwidth, interconnect, and hit-rate distributions.

### 6. End-to-end NVFP4 pretraining kernels beyond linear layers

TransformerEngine supports NVFP4 on Blackwell and handles scale metadata, stochastic rounding, Hadamard transforms, and quantized all-gather requirements. But full pretraining still needs strong support around:

- backward kernels for attention variants and GQA;
- scale synchronization and all-gather behavior;
- optimizer/update precision choices;
- activation checkpointing with NVFP4 tensors;
- MoE backward and grouped weight-gradient paths;
- validation of stability across model families.

Why it is high value: if stable NVFP4 pretraining works broadly, it changes training economics. But the risk is higher than serving because numerical stability is the hard part.

Research angle: numerically aware kernels that expose quantization error controls, not opaque low-precision GEMMs.

## Lower Priority

- Hopper/A100-only attention or GEMM tuning.
- One-off RoPE + quant + cache-store fusions unless they are part of a larger orchestration path.
- Another generic Triton attention kernel for Blackwell without TCGEN05/TMEM/warp-specialized pipeline control.
- Single-GPU grouped GEMM microbenchmarks disconnected from EP, routing, CUDA graphs, and batch dynamics.
- Pure SM120 workstation compatibility work, unless the goal is desktop Blackwell inference. It is practical, but less impactful for frontier datacenter workloads than SM100/SM103/GB200/GB300 paths.

## Best Bet

If choosing one research direction, pick the Blackwell distributed MoE layer scheduler.

Reason: it sits on the hottest workload class, touches every major runtime, has obvious fragmentation today, benefits from new Blackwell mechanisms, and can produce large gains by changing scheduling and communication structure rather than just shaving one memory round trip. A good prototype would target Qwen/DeepSeek/Nemotron-style MoE on B200/GB200, compare against vLLM/SGLang/TensorRT-LLM backends, and report throughput, latency, SM use, communication overlap, and correctness across prefill/decode/RL rollout shapes.
