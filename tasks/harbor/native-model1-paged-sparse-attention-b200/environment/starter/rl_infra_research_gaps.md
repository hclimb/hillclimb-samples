# Practical SOTA RL Infra Research Gaps - 2026-05-11

Workspace clone root: `/Users/aumdesai/kernel+infra/sota-rl-infra-repos`

I treated "SOTA RL infra" as active, practical infrastructure rather than old algorithm-only repos. The audit covers LLM RL/post-training, agentic RL, embodied/VLA RL, distributed environment RL, safety RL, environment APIs, and baseline/reproducibility libraries. I intentionally deprioritized obsolete stacks such as legacy Gym, trlX-style RLHF stacks, DeepSpeed-Chat-style examples, and older monolithic research repos unless they still show current practical use.

## Cloned Repos

LLM/agentic RL:
- `verl-project/verl`
- `OpenRLHF/OpenRLHF`
- `huggingface/trl`
- `NVIDIA-NeMo/RL`
- `areal-project/AReaL`
- `alibaba/ROLL`
- `THUDM/slime`
- `RLinf/RLinf`
- `NovaSky-AI/SkyRL`
- `rllm-org/rllm`
- `pytorch/rl` as `torchrl`

Rollout/runtime backbones:
- `sgl-project/sglang`
- `vllm-project/vllm`

Embodied/general RL:
- `ray-project/ray` (RLlib)
- `Toni-SM/skrl`
- `leggedrobotics/rsl_rl`
- `Denys88/rl_games`
- `alex-petrenko/sample-factory`
- `thu-ml/tianshou`
- `DLR-RM/stable-baselines3`
- `vwxyzjn/cleanrl`
- `google/brax`
- `isaac-sim/IsaacLab`
- `google-deepmind/mujoco_playground`
- `Farama-Foundation/Gymnasium`
- `Farama-Foundation/PettingZoo`
- `PKU-Alignment/omnisafe`

## High-Conviction Research Opportunities

### 1. Staleness-Aware Async RL for Long-Horizon Agents

Why it matters:
- The active LLM RL repos are converging on async/decoupled rollout because synchronous rollout wastes GPUs on long-tail generations.
- The research gap is not "add async"; it is controlling policy staleness when trajectories contain tokens from different policy versions, partial rollouts, dynamic sampling, tool calls, and multi-turn interactions.

Evidence from repos:
- `verl` has experimental `fully_async_policy`, `one_step_off_policy`, `transfer_queue`, and docs explicitly discuss staleness control, partial rollout, and cross-node parameter sync.
- `OpenRLHF` documents async mode and partial rollout, noting that in-flight samples may mix old/new weights.
- `AReaL` is built around fully async RL and stores per-token output versions.
- `NeMo-RL` has async GRPO/replay support but active issues around GPU time sharing, async vLLM generation, fragmentation, and refit.
- `SkyRL` tracks fully async in-flight weight updates.
- `slime` and `ROLL` both expose decoupled async rollout paths and issues around hangs, timeout, abnormal outputs, and slow async training.

Research direction:
- Define behavior-policy provenance at token, segment, and trajectory levels.
- Learn or derive staleness budgets conditioned on reward sparsity, trajectory horizon, and generation latency.
- Combine importance correction, freshness-aware replay, partial rollout continuation, and adaptive discard/keep decisions.
- Evaluate not just final benchmark score, but GPU utilization, accepted-token-per-dollar, KL drift, and collapse rate.

Expected upside:
- 2-3x wall-clock throughput is already plausible from async systems; the gap is preserving or improving convergence at that speed.
- This is strong paper territory because it combines systems + RL algorithmic correction.

### 2. Universal Weight-Sync and Refit Engine

Why it matters:
- Weight transfer between trainer and rollout engines is one of the largest practical bottlenecks in LLM RL, especially for MoE, long context, VLM, LoRA, and multi-node runs.
- Every serious framework has bespoke code for FSDP/Megatron/DTensor to vLLM/SGLang/Megatron-inference handoff.

Evidence from repos:
- `verl` roadmap/issues include checkpoint engine, hybrid weight sync, and TransferQueue.
- `AReaL` has issues for elastic weight update, sparse delta compression, Ray RDT integration, and fault-tolerant inference service logic.
- `OpenRLHF` has an issue for actor rollout parameter sync optimization.
- `ROLL` has an issue reporting slow weight update progress.
- `slime` highlights DCS/TransferQueue-based async weight streaming in related systems.
- `NeMo-RL` issues include Megatron-vLLM refit crashes, memory leftovers after refit, deadlocks in checkpoint conversion, and direct Megatron checkpoint support.
- `SkyRL` has race conditions in new inference weight sync and tracks routing replay / in-flight updates.

Research direction:
- A backend-neutral weight plane with versioned tensors, delta/sparse updates, LoRA adapter deltas, RDMA/NIXL/Ray-RDT transport, and consistency contracts.
- Formal modes: strict barrier, one-step stale, bounded-stale streaming, and adapter-only update.
- Fault injection benchmark for deadlocks, partial transfer, lost rollout workers, and resumed checkpoints.

Expected upside:
- Reduced idle time, lower memory peaks, less OOM/deadlock risk, and scalable 70B-600B+ RL runs.
- This is likely a major infra contribution even if the RL algorithm is unchanged.

### 3. Long-Tail Rollout Scheduler for Multi-Turn Tool/Code/Web Agents

Why it matters:
- For LLM RL, generation dominates cost; OpenRLHF explicitly states sample generation takes most RLHF time, and slime points to rollout long-tail as the dominant bottleneck.
- Multi-turn tool agents amplify the tail: one task may finish in seconds, another may hang on a sandbox/tool/server.

Evidence from repos:
- `slime` supports dynamic sampling, partial rollout buffering, and request abort/resume.
- `OpenRLHF` supports oversampling and async partial rollout.
- `AReaL`, `SkyRL`, and `rLLM` are centered on long-horizon agent workflows.
- Issue surfaces include sandboxed execution, failed rollout crashing whole training steps, timeout/hang after rollout, and multi-turn trace bugs.

Research direction:
- Budgeted rollout allocation: adapt group size and max tokens based on early reward/verifier signal.
- Partial continuation with unbiased estimators: recover value from aborted requests instead of throwing them away.
- Queueing policy that jointly optimizes GPU batch efficiency and environment worker utilization.
- Verifier-aware scheduling: allocate more rollouts to prompts whose reward uncertainty is high.

Expected upside:
- Potentially larger than model-side kernel gains because it attacks wasted generation and stuck environments.
- Strong candidate for an "RL systems" paper and a reusable library.

### 4. Standard Trace Schema for Agentic RL

Why it matters:
- Each framework has its own notion of rollout, episode, trajectory, tool call, reward, token IDs, logprobs, masks, and model version.
- This fragmentation makes environment reuse and apples-to-apples benchmarking difficult.

Evidence from repos:
- `rLLM` wraps arbitrary agents and captures token IDs/logprobs through a gateway.
- `AReaL` has workflow APIs and output version metadata.
- `SkyRL` has `skyrl-gym`, custom generators, and Tinker API migration.
- `OpenRLHF` emphasizes token-in-token-out agent execution.
- `verl` has AgentLoop abstractions and multi-turn tokenization/masking references.
- Farama `Gymnasium`/`PettingZoo` remain standards for classic env APIs, but do not cover LLM token/logprob/tool provenance.

Research direction:
- A "Gymnasium for LLM agents" trace API:
  - immutable episode and trajectory records
  - per-token logprobs and model version
  - tool call spans, tool outputs, sandbox state hashes
  - multimodal payload references
  - reward components and verifier provenance
  - replay/serialization format that works across `verl`, `OpenRLHF`, `AReaL`, `SkyRL`, and `rLLM`

Expected upside:
- Enables backend portability and much faster experimentation.
- The contribution could be both infra and benchmark standard, not a narrow optimization.

### 5. Multimodal/VLA RL Transport and World-Model Rollout

Why it matters:
- VLM/VLA RL shifts data movement from token arrays to images, video, audio, robot observations, and action chunks.
- The current RL control planes were built mostly around text token tensors.

Evidence from repos:
- `RLinf` is aggressively expanding VLA, world-model, real-world robot, and embodied RL support.
- `verl`, `OpenRLHF`, `AReaL`, `ROLL`, and `NeMo-RL` all added VLM/VLA/multimodal paths recently.
- `RLinf` open issues include VLA with SGLang/vLLM rollout backend, tensor fast-path payloads for env-to-rollout observation transport, real-world VLA training, WAN world model instability, and slow world-model training.
- `verl` docs state that central controller data movement becomes a bottleneck for multimodal training and router replay.

Research direction:
- Zero-copy or reference-based media/tensor transport for observations.
- GPU-resident env buffers for simulated observations and action chunks.
- Confidence-aware world-model rollout: adaptively choose simulator, world model, or real robot based on estimated model error.
- Unified VLA action/reward schema across LIBERO, ManiSkill, IsaacLab, MuJoCo Playground, and real hardware.

Expected upside:
- This is one of the clearest places to get "substantial performance" because data payloads are much larger than text-only RL.
- Also high research value due to VLA+RL being early and fragmented.

### 6. Cross-Backend Numerical Equivalence and Masking Correctness

Why it matters:
- RL training is sensitive to logprob mismatch, tokenization, masking, truncation, and backend-specific behavior.
- vLLM vs SGLang vs HF vs Megatron can produce subtle differences that look like "algorithm instability" but are infra bugs.

Evidence from repos:
- `AReaL` has an issue where GRPO is stable on SGLang but collapses on vLLM.
- `verl` has issues around multi-turn response length, function calling support, garbled rollout text, and full-async VLM token crashes.
- `OpenRLHF` flags token-in-token-out consistency as a core design point.
- `slime` has VLM/multiturn image visibility and converter correctness issues.
- `NeMo-RL` has non-contiguous tensor crashes during Megatron-to-vLLM refit and context truncation issues.

Research direction:
- Backend equivalence harness for:
  - prompt rendering and chat templates
  - token/logprob equality bounds
  - truncation and loss masks
  - multimodal placeholder tokens
  - tool-call serialization
  - rollout-vs-train policy version checks
- Treat this as a formal verifier for RL data correctness before training.

Expected upside:
- Prevents expensive failed runs and collapse.
- This is "performance" in the real sense: fewer wasted GPU-days and more reproducible scaling.

### 7. Environment-Utilization RL for Code/Web/Terminal Agents

Why it matters:
- Sandboxed code, browser, terminal, and SWE environments are now major RL workloads, but environment execution is slow, flaky, and hard to reset.

Evidence from repos:
- `rLLM`, `SkyRL`, `AReaL`, `OpenRLHF`, and `verl` all expose agentic/tool workflows.
- `rLLM` and `OpenRLHF` have open issues around sandboxed execution for rollouts.
- `SkyRL` issues include failed rollout crashing whole steps and generator contracts.
- `slime` ecosystem projects include Triton kernel generation with compilation feedback.

Research direction:
- Snapshotting sandboxes with content-addressed state.
- Incremental reset and cache reuse across rollouts.
- Deterministic verifier cache keyed by task, code diff, dependency graph, and tool output.
- Curriculum/scheduler that balances expensive real environment calls against cheap proxy verifiers.

Expected upside:
- Major throughput gains for SWE/code RL without changing model architecture.
- Directly useful for training coding, browser, and terminal agents.

### 8. GPU-First Multi-Agent Vectorization for Classic and Embodied RL

Why it matters:
- Classic RL still matters for robotics, games, finance/control, and multi-agent simulation.
- The practical gap is not another PPO implementation; it is vectorized multi-agent simulation with GPU-resident observations/actions and clean distributed training.

Evidence from repos:
- RLlib docs state multi-agent setups are not vectorizable yet.
- IsaacLab comparison shows SB3 is much slower than RL-Games/SKRL/RSL-RL on Isaac-Humanoid-v0.
- `skrl` supports PyTorch, JAX, Warp, IsaacLab, MuJoCo Playground, and distributed runs, but JAX memory save/load and GPU preallocation issues remain.
- `IsaacLab` open issues include multi-GPU training bugs, reproducibility problems, and NCCL/CUDA issues.

Research direction:
- Multi-agent batched environment ABI that maps cleanly to GPU tensors.
- Shared env stepping API across IsaacLab, MuJoCo Playground, PettingZoo, TorchRL, and RLlib.
- Deterministic replay and seed protocol for multi-GPU sim.

Expected upside:
- Could close a large ergonomic/performance gap for embodied multi-agent RL and make MARL less bespoke.

### 9. Safe/Constrained RL for Agentic LLMs

Why it matters:
- `omnisafe` covers classic safe RL, but LLM RL systems mostly treat safety as static reward/preference modeling or post-hoc evaluation.
- Agentic systems need online constraints: tool risk, privacy leakage, irreversible actions, unsafe code execution, and budget limits.

Research direction:
- Bring constrained policy optimization, Lagrangian methods, and shielded exploration into LLM agent traces.
- Define cost signals over tool calls, sandbox violations, policy refusals, and externally visible actions.
- Multi-objective RL with verifiable constraints and dynamic reward weights.

Expected upside:
- More safety than raw benchmark gains, but likely high-impact because current agentic RL infra lacks a principled constraint layer.

## Lower-Priority or Less Differentiated Opportunities

- Adding another PPO/GRPO/DAPO trainer: most active repos already have this.
- A new clean baseline library: useful but unlikely to be a substantial performance opportunity.
- More hand-written recipes for a single benchmark: valuable engineering, weak research unless paired with new scheduler, trace, or correction mechanism.
- General "make it faster" kernel work: useful, but less distinctive than weight-sync, rollout scheduling, or multimodal transport unless tied to RL-specific workload structure.

## Best Bets

1. **Staleness-aware async RL**: strongest algorithm + systems research seam; broad across every LLM RL repo.
2. **Universal weight-sync/refit engine**: highest immediate infra leverage; repeated pain across `verl`, `AReaL`, `OpenRLHF`, `NeMo-RL`, `ROLL`, `slime`, `SkyRL`.
3. **Long-tail rollout scheduler**: likely biggest GPU-dollar improvement for agentic RL.
4. **Standard agentic RL trace schema**: ecosystem-level leverage; good open-source moat.
5. **VLA/multimodal tensor transport**: high upside for embodied RL where payloads dwarf text tokens.

## Sources

Primary repo/docs sources used:
- https://github.com/verl-project/verl
- https://github.com/OpenRLHF/OpenRLHF
- https://github.com/NVIDIA-NeMo/RL
- https://github.com/areal-project/AReaL
- https://github.com/alibaba/ROLL
- https://github.com/THUDM/slime
- https://github.com/RLinf/RLinf
- https://github.com/NovaSky-AI/SkyRL
- https://github.com/rllm-org/rllm
- https://docs.pytorch.org/rl/stable/
- https://docs.ray.io/en/latest/rllib/index.html
- https://isaac-sim.github.io/IsaacLab/v2.0.0/source/overview/reinforcement-learning/rl_frameworks.html
- https://github.com/sgl-project/sglang
- https://github.com/vllm-project/vllm
- https://github.com/google-deepmind/mujoco_playground
- https://github.com/Farama-Foundation/Gymnasium
- https://github.com/Farama-Foundation/PettingZoo
- https://github.com/PKU-Alignment/omnisafe
