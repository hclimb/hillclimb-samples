# Serious Gaps in Post-Training RL Training/Inference Infrastructure

Date: 2026-05-11  
Local clones: `/Users/aumdesai/kernel+infra/sota-rl-infra-repos`

Scope: post-training/RL run infrastructure for LLMs and agentic/VLM variants. I focused on the training-inference boundary in `verl`, `OpenRLHF`, `NVIDIA-NeMo/RL`, `AReaL`, `ROLL`, `slime`, `SkyRL`, `TorchRL`, `rLLM`, `SGLang`, and `vLLM`. I excluded generic RL libraries except where they now touch LLM rollout/weight-sync infrastructure.

Core finding: the serious frontier is not another PPO/GRPO implementation. It is the mutable boundary between trainer and rollout serving: live weight updates, KV/cache validity, logprob provenance, partial rollouts, replay freshness, and schedulers that keep trainers fed without silently breaking RL semantics.

## 1. Token-Level Behavior-Policy Ledger for Async RL

This is the highest-value research gap.

Current systems already expose the problem:
- `verl` fully async has `staleness_threshold`, `partial_rollout`, rollout logprobs, and old-logprob correctness warnings.
- NeMo-RL async GRPO requires importance-sampling correction and tracks trajectory age.
- AReaL has per-token `versions` and proximal-logprob approximations.
- SkyRL explicitly says its staleness manager gives an aggregate capacity bound, not a hard per-group guarantee.
- slime stores rollout logprobs and partial rollout metadata, but partial/mixed-policy samples are still framework-local.
- OpenRLHF uses queue sizing and vLLM locks; partial rollout intentionally relaxes the no-overlap guarantee.

The gap: no common token-level ledger records the exact behavior policy behind every token across partial rollouts, in-flight weight updates, tool calls, VLM payloads, and retries. Most stacks reason at batch, sample, rollout, or coarse step granularity. That is not enough when one trajectory can span multiple policy versions or when old logprobs are produced by an inference engine with different tokenization/logprob semantics than training.

Research direction: build a `PolicyVersionedTrace` runtime:
- token table: token id, position, policy version, engine id, sampling params, raw and processed logprob, route/expert metadata, reward provenance, and target training version
- partial-rollout spans: first policy version, continuation version, masked vs trainable tokens
- loss adapters: PPO/GRPO/DAPO/decoupled PPO/MIS corrections selected from trace metadata
- scheduler hooks: admit/continue/drop samples based on measured policy drift, not just age

Why it is serious: full async can produce 2x+ speedups, but large labs still have to tune staleness conservatively because correctness is underspecified. A general provenance layer would let systems push async harder while keeping the loss mathematically defensible.

Evaluation: fixed-quality time-to-target on math/code/tool tasks, stale-token ratio, accepted-token-per-GPU-hour, KL drift, collapse rate, and recovery from interrupted partial rollouts.

## 2. Transactional Weight-Sync and Refit Plane

Every serious framework is rebuilding this:
- `verl` fully async uses NCCL parameter synchronization.
- NeMo-RL supports vLLM/SGLang/Megatron generation and async weight refits.
- AReaL has an experimental `weight_update` service with FSDP/Megatron/SGLang adapters.
- SkyRL has dedicated weight sync abstractions, transfer strategies, CUDA IPC/broadcast, and RDT tracking issues.
- ROLL has Megatron/vLLM/SGLang patches and an open Ray Direct Transport weight-sync RFC.
- TorchRL added thousands of lines of vLLM/SGLang weight-update infrastructure.
- slime updates SGLang rollout workers from distributed or disk paths.

The gap: weight sync is not a reusable, transactional system component. It is backend-specific, fragile under LoRA/MoE/VLM variants, hard to verify, and frequently entangled with sleep/wake/offload behavior. Most implementations still lack crisp semantics for: when a new version is visible, whether a request can observe mixed weights, how KV cache should be invalidated, and how to roll back after a failed update.

Research direction: a backend-neutral transactional weight plane:
- versioned weight chunks with checksums and shape/dtype schema
- two-phase apply: prepare, verify, commit visibility
- pluggable transports: NCCL, CUDA IPC, Ray RDT, RDMA/Mooncake, disk fallback
- model-layout adapters: FSDP2, DTensor, Megatron TP/PP/EP/CP, vLLM, SGLang, Megatron inference
- delta/adapter paths: LoRA-only updates, sparse updates, expert/router-only updates
- built-in correctness probes: sentinel prompt logprobs before/after commit

Why it is serious: for large runs, trainer idle time, rollout idle time, and OOM/deadlock risk often come from this boundary. A reliable weight plane is leverage across every post-training stack.

Evaluation: refit wall time, peak memory, trainer/rollout idle ratio, failed-update recovery, 70B/235B/MoE scaling, and logprob equality after sync.

## 3. RL-Aware Inference Scheduling

vLLM/SGLang are excellent serving engines, but RL rollout serving has different objectives than online inference.

Serving optimizes throughput/latency/prefix cache. RL rollout needs to optimize:
- group completion for `n` samples per prompt
- freshness relative to trainer progress
- verifier/reward-model latency
- long-tail output prediction
- partial rollout continuation vs abort
- prompt budget allocation under dynamic filtering
- cache locality without overproducing stale samples

Current frameworks bolt this on externally: `verl` fully async modes, SkyRL capacity control, slime dynamic sampling and partial rollout, ROLL async parallel rollout, and StreamRL length-aware scheduling. There is no standard scheduler contract in the inference engine/router that understands RL semantics.

Research direction: an RL-native rollout scheduler inside or beside vLLM/SGLang:
- schedule prompt groups, not just individual requests
- expose generation checkpoint/pause/resume with policy-version tags
- predict output length and verifier cost; prioritize samples that unblock trainer batches
- staleness-aware admission control and early-stop/continue policies
- return structured partial states rather than opaque text

Why it is serious: long-tail rollout is the dominant bubble in agentic/code/tool RL. This is likely one of the largest GPU-dollar opportunities.

Evaluation: trainer idle ratio, rollouter idle ratio, P95/P99 rollout delay, stale discard rate, wall-clock to target benchmark score, and cache hit rate under bounded staleness.

## 4. KV-Cache Semantics Under Live Weight Updates

In-flight weight update is now a major performance knob, but the semantics are underdeveloped.

Examples:
- NeMo-RL exposes `recompute_kv_cache_after_weight_updates`.
- OpenRLHF partial rollout uses pause/resume around vLLM generation.
- TorchRL puts vLLM engines to sleep before updates and wakes scheduling after.
- slime stores partial requests and can continue later.

The unresolved question: if a request begins under policy version N, receives a weight update to N+1 mid-generation, and reuses old KV cache, what policy generated the next token? It is not simply N or N+1. The hidden state was computed under old weights, while later projections may use new weights. That breaks clean behavior-policy assumptions unless traced, recomputed, masked, or isolated.

Research direction: `PolicyKVEpoch`:
- tag KV pages with model/weight version and sampling span
- support continuation modes: finish-on-old, recompute-on-new, mixed-with-ledger, or discard
- quantify bias from mixed KV continuation against recompute ground truth
- add engine-level API for safe pause/resume/refit of active requests

Why it is serious: safe in-flight updates can remove the long-tail wait before every refit. Unsafe in-flight updates can silently poison RL.

Evaluation: speedup vs recompute, logprob error, reward/quality regression, and stability under long-context tool/code rollouts.

## 5. Portable RL Trace/Data Plane

The data-plane story is fragmented:
- `verl` has `DataProto` and is integrating TransferQueue to remove single-controller bottlenecks.
- NeMo-RL uses its own `BatchedDataDict` and replay buffer.
- OpenRLHF uses a `NaiveReplayBuffer` with dynamic batching.
- slime has a `Data Buffer` and `Sample` objects.
- SkyRL has async dataloader and generator-output groups.
- rLLM has a model gateway/session layer.

The gap: there is no portable trace/data format for post-training rollouts. A serious trace needs tokens, masks, logprobs, policy versions, chat template metadata, tool spans, verifier results, sandbox hashes, VLM references, routing/expert metadata, reward provenance, and retry/failure state.

Research direction: `OpenRLTrace`:
- Arrow/Parquet-style columnar schema plus tensor references
- streaming append/read API for trainers, reward workers, and inference workers
- zero-copy backends where possible: Ray RDT, Mooncake, shared memory, object store
- compatibility adapters for `DataProto`, TensorDict, OpenRLHF experiences, slime samples

Why it is serious: without a trace standard, debugging async collapse and reproducing runs remains bespoke. It also blocks common benchmarks for infra improvements.

Evaluation: overhead vs native pipelines, replay fidelity, cross-framework import/export, and end-to-end debugging time for synthetic corruption cases.

## 6. Cross-Backend Logprob and Tokenization Conformance

This is less glamorous but very high leverage.

Signals:
- NeMo-RL explicitly designs generation to exchange tokens rather than text to avoid tokenizer drift.
- OpenRLHF emphasizes token-in/token-out agent execution.
- TorchRL has wrapper code handling vLLM/SGLang logprob shape/version differences and TODOs around prompt logprobs.
- `verl` has open issues on raw vs processed logprobs in fully async rollouter/trainer paths.

The gap: no shared conformance suite proves that train-side and inference-side logprobs match across vLLM, SGLang, Megatron inference, HF, FP8, packed sequences, VLM placeholders, chat templates, MoE routing, LoRA adapters, truncation, and EOS/BOS edge cases.

Research direction: RL-infra conformance suite:
- golden prompts covering chat/template/tool/VLM/MoE/long-context cases
- compare rollout logprobs, recomputed train logprobs, masks, and truncation
- run nightly against multiple engine versions
- produce minimal repro traces when drift exceeds tolerance

Why it is serious: logprob mismatch turns PPO/GRPO into the wrong algorithm. It can waste enormous compute before surfacing as "instability".

Evaluation: caught regressions, false positive rate, runtime overhead, backend coverage.

## 7. Exactly-Once Fault Tolerance for Async RL

Async systems make failure semantics harder:
- generation can be in flight
- samples can be partial
- weight sync can fail mid-update
- queue state may not match dataloader state
- a retry can duplicate a prompt or train a stale partial sample

Open issues across stacks include deadlocks, slow/failed weight updates, async rollout timeouts, checkpoint conversion/saving problems, and queue/race issues. Current systems handle pieces of this, but not the whole run as a recoverable transaction log.

Research direction: exactly-once async rollout ledger:
- every prompt/group/partial token has idempotency keys
- generator, reward, trainer, and weight-update states are checkpointed together
- recovery can classify pending work as replay, discard, continue, or train
- no duplicate training after resume

Why it is serious: this is mandatory for long 1000+ GPU runs where worker death is normal, not exceptional.

Evaluation: failure-injection suite: kill trainer, rollout worker, router, reward worker, and weight-sync receiver; measure recovery correctness and wasted GPU time.

## 8. Closed-Loop Resource and Staleness Controller

Most frameworks expose knobs: rollout GPUs, trainer GPUs, queue size, staleness threshold, sync frequency, microbatch sizes, offload mode, vLLM memory fraction, generation concurrency. Users manually tune them.

The gap: no system closes the loop using runtime metrics to rebalance resources and staleness while preserving convergence. `verl` fully async docs recommend manual adjustment from trainer/rollouter idle ratios. NeMo-RL has GPU time-sharing discussions. SkyRL/ROLL track RDT and weight-sync improvements. This is still mostly operator craft.

Research direction: post-training control plane:
- estimate trainer and rollout service curves online
- choose sync interval/staleness budget/resource split to minimize time-to-quality
- account for reward/verifier latency and long-tail generators
- emit guardrails when measured policy drift exceeds the algorithm’s correction budget

Why it is serious: the same recipe behaves differently across H100/H200/B200/MI300/Ascend clusters and model sizes. A closed-loop planner can turn fragile recipes into portable runs.

Evaluation: auto-tuned vs expert-tuned throughput/quality across cluster shapes, model sizes, and tasks.

## What I Would Build First

Build a policy-versioned rollout runtime that combines gaps 1, 2, 3, and 4:

1. A token-level rollout ledger with policy/KV/version/logprob provenance.
2. A transactional weight-update plane for one trainer backend and one inference backend first, e.g. FSDP2 or Megatron to vLLM/SGLang.
3. A freshness-aware scheduler that uses the ledger to decide continue/abort/recompute/train.
4. A conformance harness that verifies logprob/mask equivalence before large runs.

The first practical target should be `verl` or NeMo-RL with vLLM/SGLang because they already expose fully async, partial rollout, and multiple training backends. The research claim should not be "we are faster"; it should be "we can safely run more asynchronous, in-flight, long-tail rollout without quality loss, with verifiable provenance." That is a bigger claim and more defensible.

## Sources

- `verl`: https://github.com/verl-project/verl and local docs `docs/advance/fully_async.md`, `docs/data/transfer_queue.md`
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF and local `openrlhf/trainer/ppo_trainer_async.py`
- NeMo-RL: https://github.com/NVIDIA-NeMo/RL and local `docs/guides/async-grpo.md`, `docs/design-docs/generation.md`
- AReaL: https://github.com/inclusionAI/AReaL and https://arxiv.org/abs/2505.24298
- ROLL: https://github.com/alibaba/ROLL
- slime: https://github.com/THUDM/slime
- SkyRL: https://github.com/NovaSky-AI/SkyRL
- TorchRL: https://github.com/pytorch/rl
- SGLang: https://github.com/sgl-project/sglang
- vLLM: https://github.com/vllm-project/vllm
- AsyncFlow: https://arxiv.org/abs/2507.01663
- StreamRL: https://arxiv.org/abs/2504.15930
- Magistral: https://arxiv.org/abs/2506.10910

Relevant open issue signals:
- `verl` fully async/logprob/partial rollout issues: https://github.com/verl-project/verl/issues/6240, https://github.com/verl-project/verl/issues/6054, https://github.com/verl-project/verl/issues/6306
- NeMo-RL async/weight/generation issues: https://github.com/NVIDIA-NeMo/RL/issues/2244, https://github.com/NVIDIA-NeMo/RL/issues/1892, https://github.com/NVIDIA-NeMo/RL/issues/1906
- slime async/staleness/partial rollout issues: https://github.com/THUDM/slime/issues/1800, https://github.com/THUDM/slime/issues/1852
- ROLL weight-sync/async issues: https://github.com/alibaba/ROLL/issues/431, https://github.com/alibaba/ROLL/issues/394, https://github.com/alibaba/ROLL/issues/279
- SkyRL async/weight-sync issues: https://github.com/NovaSky-AI/SkyRL/issues/536, https://github.com/NovaSky-AI/SkyRL/issues/977, https://github.com/NovaSky-AI/SkyRL/issues/1623, https://github.com/NovaSky-AI/SkyRL/issues/1605
- TorchRL LLM/weight-update issues: https://github.com/pytorch/rl/issues/3462, https://github.com/pytorch/rl/issues/3032, https://github.com/pytorch/rl/issues/2872
