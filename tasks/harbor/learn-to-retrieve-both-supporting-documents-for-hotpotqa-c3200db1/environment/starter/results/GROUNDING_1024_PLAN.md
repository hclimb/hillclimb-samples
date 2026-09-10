# Grounding & accuracy at 1024-doc MS MARCO — brainstorm, design, run matrix

**Goal:** train a memory-layer model that beats the baseline at **1024-doc MS MARCO corpus**
retrieval. Baseline (qhn64 topk64 @100k, `qa_hard_neg_think_sft4b`): ~**0.60 accuracy /
~0.48 lexical grounding** at 1024 docs (measured earlier; new eval harness below re-measures
it as the control).

## What we already know (diagnosis from the corpus sweep + mode-B evals)

- Reading is fine; **retrieval RANKING is the bottleneck**. At 10k docs, even measured at
  answer positions (mode B), `doc_access_acc` stays ~0.06 while `doc_token_hit_rate`~0.75
  (gold tokens are in the top-k pool but gold is not the argmax). Gold loses the softmax
  competition against many distractors.
- **Softmax is corpus-size sensitive**: its denominator grows with the number of competitive
  slots, so each added distractor dilutes the gold weight. Sweep: acc 0.82(16)→0.60(1024)→
  0.49(10k); `mem_pos_weight_mass` at answer positions 0.98(16 docs) → collapses at scale.
- Eval-time temperature sharpening gave only ~+5 pts (can't change the argmax); confirms the
  fix must be in **training / architecture**, not eval calibration.

## The three levers (user's ideas) + design

### 1. Memory-attention hparams
Current: **4 Q heads, 1 KV head, head_dim (k_dim) = 1024**, v_dim = 1024, mem_top_k=128,
mem_layers=[14]. Qwen3-4B itself uses 32 Q / 8 KV / head_dim 128.

- **head_dim 1024 → 128**: query/key effective rank rarely > 128, so 1024 is wasted capacity
  + dilutes the dot-product. `mem_k_dim` (drives Q & K dim) and `mem_v_dim` are independent in
  the code, so we can shrink the *key/query* head_dim to 128 while keeping value capacity.
- **Q heads 4 → 32**: more independent retrieval probes into the same key space → more chances
  gold lands in some head's top-k. Config-only: the scoring einsum `btnh,mh->bntm` broadcasts
  the single KV head across all Q heads. **Same param count** as 4×1024 (32×128 = 4096 = 4×1024).
- **KV heads 1 → 8**: HIGHER RISK — needs a multi-head key bank `[M, n_kv, head_dim]` and
  grouped-query gather in the sharded top-k path (`retrieval_ops.py`). Deferred: first run ships
  **32 Q / 1 KV / 128** (captures the effective-rank + more-probes wins at zero code risk). If
  time/verified, attempt 8-KV in its own worktree; otherwise it stays a documented follow-up.

**Run:** `hparams` = `mem_num_heads=32 mem_k_dim=128` (v_dim=1024, 1 KV). Config override only.

### 2. Activation function (replace softmax)
Softmax's corpus sensitivity comes from its cross-slot normalization (partition function).
**Element-wise activations are corpus-size invariant** — a slot's weight depends only on its own
score, not on how many other slots compete. This directly attacks the dilution mechanism.

- **ReLU**: `w_i = relu(score_i)`. Unnormalized; gold weight independent of #distractors. Read =
  Σ relu(score_i)·v_i. Output magnitude varies with #positive slots — the zero-init `mem_o_proj`
  + training learns the scale. Primary bet.
- **Sigmoid**: `w_i = sigmoid(score_i)` ∈ [0,1] per slot, independent. Bounded (more stable than
  ReLU), still corpus-invariant. Alternative.

Note the top-k selection still uses raw scores (unchanged); only the *weighting* over the top-k
changes. If ReLU/sigmoid works, a model trained on the small in-batch bank should transfer to
1024 docs **without** any corpus emulation, because the read no longer dilutes with corpus size.

**Impl:** `mem_score_activation: softmax|relu|sigmoid` (default softmax), threaded into every read
path (`memory.py` mem_lookup replicated softmax + `retrieval_ops.py` `_replicated_top_k` /
`_sharded_top_k`), mirroring the existing `temp` knob. **Runs:** `relu`, `sigmoid`.

### 3. Cheap long-corpus emulation (NO more hard-neg mining)
From the earlier brainstorm, the cheapest lever that changes the *partition* (not just sharpness)
without mining or a data-pipeline change:

- **Phantom-mass / logQ denominator term** (softmax only): add a constant background mass to the
  softmax denominator, `softmax_i = exp(z_i) / (Σ_j exp(z_j) + exp(C))`, with `C ≈ log(N_target)`.
  This makes the gold logit clear a partition inflated to `N_target` competitors at TRAIN time,
  using only the small in-batch bank — emulating corpus-scale competition for ~free (one scalar,
  no extra keys, no data change). This is the temperature-shaped idea done *correctly* (it hits
  the denominator, which is where corpus size actually bites, instead of rescaling all logits).

**Impl:** `mem_phantom_log_n: <float, default 0 = off>`. In the softmax path, augment the
denominator with `exp(phantom_log_n)`. **Run:** `phantom` with `phantom_log_n = ln(1024) ≈ 6.93`.

(Interaction: phantom-mass only makes sense with softmax; ReLU/sigmoid are already corpus-invariant
so they don't get it. Random-distractor-bank injection and MoCo-queue were considered but rejected
for now: both need data-pipeline / state machinery and the user asked for cheap/easy this round.)

## Eval during training (the metric we actually care about)
Replace the training eval set so each `eval_interval` logs the real objective:
**MS MARCO 1024-doc corpus** — `lexical_grounding` + `doc_access_acc` (mode B, answer positions)
+ (small-n) `llm_judge_accuracy`. Grounding + doc_access are cheap (computed from generations, no
API); llm_judge on a smaller n or less often. Exact wiring from the eval_set investigation
(`gen_large_mem_msmarco_v1_embed`, `target_docs=1024`, mode-B aux already landed on grounding-exps).

## Run matrix (single lever off baseline for clean attribution)

| # | name | change vs baseline | box | code |
|---|------|--------------------|-----|------|
| 0 | **baseline** | none — EVAL existing qhn64@100k with the new 1024 eval (no training) | any | none |
| 1 | **hparams** | `mem_num_heads=32 mem_k_dim=128` (32Q/1KV/128) | 0 | config |
| 2 | **relu** | `mem_score_activation=relu` | 1 | activation knob |
| 3 | **sigmoid** | `mem_score_activation=sigmoid` | 3 | activation knob |
| 4 | **phantom** | `mem_phantom_log_n=6.93` (softmax, emulate 1024) | 2 | phantom knob |
| 5 | **hparams+relu** (combo, likely-best) | 32Q/128 + relu | 4 | both |

Each training run starts from `qa_hard_neg_think_sft4b` dataset + recipe (per
`scripts/embed/train_hard_neg_think.sh`), same seq/chunks/bs, only the one lever changed. Second
wave (after overnight results) combines the winning levers.

## Implementation plan (worktrees)
- `wt-activation` → `mem_score_activation` knob (relu/sigmoid). Verify numerically on tiny input.
- `wt-phantom` → `mem_phantom_log_n` knob. Verify denominator math.
- Both knobs are backward-compatible (defaults reproduce softmax baseline) → **merge onto
  `grounding-exps`** so all boxes run one codebase and select behavior via CLI flags (robust for
  autonomous multi-box launch). hparams + baseline need no code (CLI overrides).

## Ops / autonomy
- Boxes 0,1,3 ready now; 2,4 provisioning (babysitter idx 0-4). Launch 3 now, add 2 when ready.
- Resilient launch via babysitter; per-box dispatch by RUN env. Preemption auto-resumes from
  latest checkpoint. Active monitoring loop (ScheduleWakeup) tracks the 1024 grounding/acc curve
  per run and reports.
