# Grounding Experiments — Fixing the Read Channel

> **Living document.** Keep hypothesis, plan, and results in this one file. As runs report and insights
> land, append to the **Results & insights log** at the bottom and update the affected sections in place
> (baselines, gates, stage changes). Don't fork a separate results doc.

## Problem

Trained `qwen3_mem_embed` retrieves the correct document (`doc_token_hit_rate` ≈ 100%) but still
answers grounded QA wrong — asks for a date, model emits a plausible but wrong date. Oracle-memory
diagnostic (commit `74e8f08`): oracle 0.281 ≈ corpus 0.289. Retrieval routing is fine; the
**read channel is the bottleneck** — the path `embed → K,V → top-k softmax mixture → o_proj → single
residual add at layer 14` loses the exact answer token.

Three suspected failure modes, to be separated by diagnostics before/while running the fixes:

- **Alignment** — the answer token (e.g. a bare date) has low query–key similarity, so even at
  100% `doc_token_hit_rate` it gets ~0 softmax weight in the 128-way mixture. Wrong *position* read.
- **Fidelity** — retrieved value vectors don't carry surface form. The shared embedding-model trunk
  (contrastively trained, homogenizes per-token states) collapses digit-level detail. Right position,
  garbled *content*.
- **Dilution** — one memory write at layer 14 gets washed out of the residual stream before the
  answer position. Nothing forces the read to survive to the decision.

Current config (`configs/model/qwen3_mem_embed.yaml`): `mem_layers: [14]`, `mem_top_k: 128`,
`mem_v_dim: 1024`, single shared `Qwen3-Embedding-0.6B` producing both K and V, `embed_conv`
kernel/stride 1 (no sequence compression — 1 doc token = 1 memory slot).

---

## Eval-time diagnostics (run alongside every experiment)

These decide which failure mode dominates. Cheap, additive, reused across all three stages.

1. **Oracle-memory eval** — build the memory bank from each question's OWN gold `pos_doc` (no
   negatives), so the model reads from the correct document only. Isolates the read channel from
   retrieval. (Oracle-*context* — doc in the main model's text context — was considered as a
   tighter isolator but **dropped** 2026-07-03; corpus + oracle-memory are the reference points.)
2. **Answer-slot softmax weight** — log the softmax weight mass on the answer-token memory slot vs
   `doc_token_hit_rate`, split by LLM-judge correct/incorrect. Prediction if alignment is the bug:
   incorrect cases have hit=1 but weight≈0. Machinery already plumbed via `pos_slot_indices` /
   `mem_pos_logits` in `mem_lookup_two_pass` (`models/memory.py`).
3. **Direct logit attribution (DLA)** — ablate (zero) the layer-14 memory output, measure change in
   the *answer-token logit*. Measures relevance, not magnitude. Do NOT use `||o_mem|| / ||x||` norm
   ratio — a small-norm write can still flip the answer. Frozen unembed makes DLA clean.

**Decision use:** weight≈0 on answer slot → alignment problem (Stage 3 span readout matters most).
Weight fine but value garbled → fidelity problem (Stage 2 K/V split matters most). DLA decays across
depth → dilution (Stage 1 multi-layer matters most).

---

## Training-time telemetry (wandb, logged every step / N steps)

Continuous per-step signals — cheaper than the eval diagnostics and let us watch the failure modes
evolve *during* training rather than only at checkpoints. Most are computable from `aux_data` already
plumbed in `models/memory.py` (`mem_scores`, `mem_top_k_logits`, `mem_top_k_indices`, `mem_pos_logits`)
where the aux losses run in the trainer. Log per memory layer (they'll diverge — that's the point).

**Write magnitude / dilution — Stage 1 (zero-init o_proj, multi-layer):**
- **Per-layer memory output norm `‖o‖`** (the requested one). Mean over batch·tokens, one series per
  mem layer. With zero-init, watch each layer climb off zero; watch whether later layers write less.
- **Relative write ratio `‖o‖ / ‖x‖`** (memory output vs residual-stream norm at that layer). Is the
  write meaningful relative to the stream, and does it shrink with depth (dilution)?
- **`o_proj` weight norm `‖W_o‖` per layer.** Direct read on zero-init growth; a layer whose `‖W_o‖`
  stays ~0 is a dead memory layer (collapsed) — tells us if 4 layers is 4 useful writes or 1 + 3 dead.
- **Grad norm into each `mem_o_proj`.** Which layers the loss actually uses (relevance, not magnitude).
- **Cross-layer output cosine** `cos(o_i, o_j)`. Are layers writing complementary vs redundant info?
  Redundant ⇒ multi-layer buys little; drop back to fewer.

**Retrieval sharpness / averaging blur — all stages (suspect #1, #2):**
- **Top-k softmax entropy** and **effective #slots** = `exp(entropy)` or participation ratio
  `1/Σwᵢ²`. Directly measures the 128-way averaging blur — how many slots actually contribute. Watch
  it sharpen (or not) over training. If it stays ~128, the mixture is mush regardless of hit rate.
- **Top-1 weight fraction** (max softmax weight). Complementary sharpness read.
- **Positive-slot weight mass** — softmax weight landing on the answer-region (positive-doc) slots,
  not just argmax `doc_access_acc`. Train-time version of the eval answer-slot-weight diagnostic; uses
  `mem_pos_logits`. The number that separates "right doc" from "right answer."
- **Train-time hit-rate** (top-k contains a positive-doc slot) — train analog of `doc_token_hit_rate`.

**Head redundancy — all stages (heads share K/V, only queries specialize):**
- **Cross-head query cosine** `cos(q_i, q_j)` averaged over head pairs. Heads here can only specialize
  their *query direction* (K/V are shared, `mem_k`/`mem_v` single bank). If this trends → 1, the N
  heads collapse to the same probe and extra heads buy nothing.
- **Per-head top-k overlap (Jaccard)** on `mem_top_k_indices` across head pairs. The direct question:
  are the 4 heads selecting *different* slots or the same ones? High overlap ⇒ redundant heads ⇒ more
  heads won't help and the shared-K/V design is the limiter (would argue for per-head keys, a larger
  change). Low overlap ⇒ heads add genuine coverage, and the `N·v_dim` readout bandwidth is earning
  its cost. Feeds the (deferred) 1-vs-4-vs-8 head ablation.

**Key/value specialization — Stage 2 (separate K/V):**
- **`cos(K_repr, V_repr)` for the same doc token.** The whole Stage-2 thesis is K and V should be
  *different objects*. This should **drop** as the two models specialize — the direct success metric.
- **Value-space anisotropy** — mean pairwise cosine of `mem_v` over a batch (or effective rank via
  singular-value participation ratio). Embedding models collapse values (high cosine); base-LM values
  should be more spread. Quantifies the fidelity win.
- **Value-reconstruction accuracy as a passive metric** (weight=0, not optimized) — decode `mem_v`
  through the value model's `lm_head`, top-1 match to the source doc token. Turns the deferred recon
  *loss* into a read-only fidelity probe: tells us if values carry surface form without letting the
  objective game it.

**Span readout — Stage 3:**
- **Boundary-straddle fraction** — share of retrieved windows crossing a doc boundary (should be ~0;
  correctness probe for the `mem_mask` respect).
- **Neighbor-vs-center weight mass** — how much of the window's contribution comes from the neighbors
  vs the matched center slot. Confirms the window actually pulls adjacent answer tokens in.

**Stage-boundary health:**
- At the Stage A→B transition, log per-param-group grad norms (mem vs embed) to confirm the embed
  unfreeze actually took (guards the resume/stage bugs that bit the sim-pairs campaign).

Touch-point: extend the aux-metric logging path in `trainer/trainer.py` (where `doc_access_acc` is
already logged as a weight-0 metric) to emit these. Per-layer series need the aux dict keyed by layer
index — `main_forward` already aggregates `all_aux_data` as a per-layer list.

---

## Training recipe (shared across all stages)

Two-stage schedule, **main model frozen throughout** (no 4B unfreeze). Replaces the historical
4-stage `staged_sim.yaml`: routing warmup is subsumed by zero-init `mem_o_proj`, and freezing main
removes the delicate low-LR stage-3 and its resume-boundary hazards (`1ec9d8c`, `b9df852`, `0d6cdf5`).
Fewer boundaries = fewer confounds = cleaner attribution.

- **Stage A (0 → 1000 steps):** `ce_weight: 1.0`, `doc_access_loss: 0.1`. Trainable = **memory
  projections only** (`.*mem_.*`); embed model(s) frozen. `learning_rate: 1e-4`. Wires the memory
  read/route on top of off-the-shelf embeddings before the embed model moves.
- **Stage B (1000 → 150000 steps):** `ce_weight: 1.0`, `doc_access_loss: 0.1`. Trainable = **memory
  projections + embed model(s)** (`.*mem_.*`, `.*embed_model.*`); **main model frozen**.
  `learning_rate: 5e-5`, cosine decay over the horizon.

Frozen main can't collapse to closed-book (its answer can't improve except through memory), so CE-from-
step-0 is safe; no ce=0 warmup needed. Higher LR than the old 1e-5 is fine — no 4B in the trainable
set to protect (`4a552c9`: 5e-5 accelerates doc_access). Config: new `configs/trainer/staged_ground.yaml`.

**Checkpoint frequently** (`checkpoint_interval: 2000`, matching `staged_sim.yaml`, or tighter) so the
continuous-eval box always has a fresh checkpoint to score — see Infra / ops below. Frequent saves are
a hard requirement for the training-curve-of-accuracy we want, not a nice-to-have.

**Training data (decided): `dataset=qa_hard_neg_think_sft4b`.** Hard-negative SFT with CoT ("think"),
ETD-free. Sources: `science_qa_hard_neg_think`, `diverse_qa_hard_neg_think`, `combined_hard_neg_sft4b`,
`multihop_qa_sft`. Launch template: `scripts/embed/train_hard_neg_think.sh` (batch_size 16, seq_len 512,
`num_chunks_per_doc: 16`). **Two overrides vs that script:**
- `checkpoint_interval: 2000` — NOT the script's 20000. Dense checkpoints are required for the
  continuous-eval accuracy curve (see Infra / ops).
- `mem_top_k: 128` — the script uses 64; we fix **128** across control + all stages for clean Stage-1
  attribution, and sweep top_k post-hoc via `MEM_TOP_K` on finished checkpoints (see Fixed knobs below).

Note: this mix does **not** include msmarco/hotpotqa-specific QA (`multihop_qa_sft` is musique-flavored),
so msmarco/hotpotqa reading is somewhat out-of-distribution — keep in mind when reading those two curves.

**Fixed knobs (hold constant across control + all stages):** `mem_top_k=128`, `mem_v_dim=1024`,
`mem_num_heads=4`. top_k and head-count are explored *separately* — top_k post-hoc via `MEM_TOP_K`,
heads via the deferred 1-vs-4-vs-8 ablation — never folded into a stage, to keep each stage's delta clean.

---

## Control run (recipe-only baseline)

Run the **new recipe with the old architecture** — the apples-to-apples reference the grounding stages
are measured against. Same 2-stage frozen-main schedule, same QA-only data, same continuous eval, but
the *original* memory setup:
- `mem_layers: [14]` (single layer)
- `mem_o_proj` **old init** (`* 0.02` normal, **not** zero-init)
- `mem_num_heads: 4`, `mem_top_k: 128`, single shared `Qwen3-Embedding-0.6B` (no K/V split)

**Why it's necessary:** it's the one baseline that holds recipe and data fixed against the grounding
runs, so any Stage-1 delta is *purely* zero-init + multi-layer. **The control is the baseline — no need
to compare against the original sim-pairs checkpoint** (that one confounds old recipe + similarity data
on top of the arch, so it's not a clean reference; ignore it). Two reference points only:
1. **Control run** — old arch, new recipe, QA-only. The baseline Stage 1/2/3 are measured against.
2. **Grounding Stages 1–3** — new arch, new recipe, QA-only.

Run the control alongside Stage 1 (same eval boxes, distinct wandb name `ground_control`).

## Infra / ops (TPUs, checkpointing, continuous eval)

**Goal:** a live LLM-judge **binary accuracy** curve on the eval QAs *as training progresses*, not just
at the end. Retrieval metrics (`doc_access_acc`) already stream to wandb; the thing that actually
answers "can it read" is judged QA accuracy per checkpoint, so we want it continuously.

**TPU allocation (decided).** 5 personal `rohun*` TPUs (`v6e-8`, europe-west4-a), one training run per
box, run all four **in parallel**:
| TPU | Run |
|-----|-----|
| `rohun`-1 | **Control** (`ground_control`) — old arch + new recipe |
| `rohun`-2 | **Stage 1** (`ground_s1_zeroinit_4layer`) |
| `rohun`-3 | **Stage 2** (`ground_s2_kv_split`) — stacks on Stage-1 arch |
| `rohun`-4 | **Stage 3** (`ground_s3_span`) — stacks on Stage-2 arch |
| `rohun`-5 | **Eval box** (always-on) |

- **Training boxes.** Launch per README: single-run via `scripts/embed/*.sh` (Hydra overrides), multi-VM
  via `scripts/infrastructure/multi-vm-tpu-run.sh`. Checkpoints → GCS (`gs://...`) where the eval boxes read them.
- **Eval boxes.** `rohun`-5 plus **2 more `v6e-8`s provisioned via the TPU-nanny package** to speed evals
  → **3 eval boxes total.** Needed because eval load = **4 runs × 3 datasets** (msmarco, hotpotqa, musique)
  per checkpoint cycle; one box can't keep up with 4 training runs saving every 2000 steps. Shard the
  work across the 3 boxes (e.g. by dataset, or by run). Each uses the sim-pairs eval-box pattern —
  `scripts/misc/box_run.sh`, `sim_eval_box.py`, `sim_find_latest.py` — and frees the TPU between evals
  (`free_tpu_devices()`, `89741af`/`560b8de`) so the judge vLLM doesn't wedge `/dev/vfio`.

**Parallel-run tradeoff (deliberate).** The stages are *stacked* (Stage 2 arch ⊃ Stage 1, Stage 3 ⊃
Stage 2), so launching all four at once means the later runs start **before** the earlier stage's decision
gate resolves. That's an accepted speed-for-compute trade: the gates become **keep/kill** decisions on
already-running experiments rather than launch gates. If Stage-1 diagnostics later reveal a design flaw,
kill Stages 2/3 and relaunch — cheaper in wall-clock than running them strictly sequentially. Watch the
live curves; don't wait for 150k to cut a losing run.

**Why dedicated boxes instead of in-loop eval:** in-training eval (`trainer.eval_interval`) stalls the
training step and fights the LLM-judge vLLM for the TPU — the sim-pairs campaign hit exactly this
(`a04fc00` vLLM-TPU-busy bug). Decoupling eval onto its own always-on `rohun*` box keeps training at
full throughput and gives a denser accuracy curve (every saved checkpoint, not every `eval_interval`).

**Cadence:** `checkpoint_interval` (2000, or tighter early) sets how dense the accuracy curve is. Eval
box loops: poll newest checkpoint → if unseen, eval → log → repeat. Checkpoint rotation window must
outlast one eval pass (orbax keep-N), a constraint the campaign already tuned (`8387d1e`).

**Metric of record:** binary LLM-judge accuracy on **MS MARCO / HotpotQA / MuSiQue** (per
checkpoint/step), alongside the retrieval + telemetry signals above. That curve is the top-line answer
to "does the read channel work."

---

## Stage 1 — Zero-init o_proj + multi-layer + frozen main

**Goal:** establish a clean-attribution baseline. Prove the memory channel alone can inject useful
info, and reduce dilution by writing at multiple depths.

**Changes** (architecture only — training schedule per the shared recipe above)
- **Zero-init `mem_o_proj`.** Memory branch starts as identity; contribution grows only as it earns
  CE reduction. Any CE drop is provably memory-sourced. Also subsumes the old routing warmup. Touch-
  point: memory-layer weight init in `models/memory_utils.py` (`add_memory_layer`). Add a config flag
  `memory.mem_o_proj_zero_init: true`.
- **Multiple memory layers.** `mem_layers: [14]` → `[9, 14, 20, 27]`. Two independent wins: more of
  the memory signal reaches the residual stream (dilution), and later layers query an `x` that already
  carries earlier reads → iterative re-retrieval (alignment), even with main frozen.
- **Main frozen throughout** (shared recipe). Caps the ceiling on purpose — this is the scientific
  control, not the final recipe.

**Known transient:** with `W_o = 0`, `dL/dy = W_o · dL/do = 0`, so q_proj / mem_k_proj / mem_v_proj
get zero gradient at step 0. Self-healing once `W_o` becomes nonzero (few steps). Expect a flat early
loss; not a bug. (Stage A trains only these projections, so the transient lives entirely inside the
1000-step Stage A before the embed model unfreezes.)

**Metrics / success**
- All three diagnostics above.
- CE / LLM-judge on the grounded QA eval set vs the current `[14]`-layer checkpoint.
- DLA per memory layer — does writing at 4 depths keep the read relevant at the answer position?

**Decision gate:** if zero-init + multi-layer gets meaningful lift **over the control run** (not the
historical sim-pairs checkpoint) and the diagnostics show a clean channel, proceed to Stage 2. If the
channel is still dead (oracle-memory ≫ corpus-retrieval read even here), the bottleneck is capacity/fidelity, not
optimization — Stage 2 becomes the priority.

---

## Stage 2 — Separate Key / Value models (stacked on Stage 1)

**Goal:** fix fidelity. Keys and values optimize different objectives — keys need topical document
info so queries route correctly; values need surface-specific info so dates/names reconstruct. Forcing
both through one contrastively-trained embedding trunk collapses the per-token detail values need.

**Changes**
- **Two embedding models.**
  - **Key model:** `Qwen3-0.6B` (base) or keep `Qwen3-Embedding-0.6B`, **bidirectional** attention
    (each token sees full doc → good contextual routing keys). Finetuned.
  - **Value model:** `Qwen3-0.6B` **base LM**, not the embedding model, **trainable**. Base-LM
    per-token hidden states are trained to carry next-token/surface info — exactly the starting point
    value reconstruction needs; finetuning then sharpens it for the read task. Embedding models
    homogenize per-token states toward a pooled summary; wrong init for values.
- **Value model trainable** (both K and V models unfrozen in Stage B, per the shared recipe), plus a
  learned value projection. Base-LM init preserves pretrained token fidelity; training adapts it.
**Deferred — value-reconstruction aux loss (not in the first Stage-2 run).** Idea: decode `mem_v`
through the value model's own `lm_head` back to the doc token id. A base LM's hiddens are *made* to be
decoded by their lm_head, so the loss is natural and it would directly train the reading fidelity that
nothing currently optimizes. Skipped for now to keep the K/V-decouple signal clean — add it only if
decoupling alone doesn't move fidelity. Caveat when we do add it: with a *trainable* value model, the
loss can be gamed (model shifts hiddens to be trivially self-decodable rather than useful for the read)
— stop-grad the `lm_head` and/or keep the recon weight low. Register in `losses/` alongside
`doc_access_loss`.

**Bidirectional-vs-causal for values — test, don't assume.** Causal saves compute but a causal value at
the cue position ("born") *cannot* carry a date that *follows* it. Bidirectional value can. Start
bidirectional for values too; ablate causal only if compute forces it.

**Code touch-points**
- `models/qwen3_mem_embed.py::embed_forward` — currently one `qwen3_forward` producing shared
  `last_hidden_states`; split into two embed passes (key model, value model) with independent weights.
- `models/qwen3_mem_embed.py::load` — load a second embed model; `split_weights` / `merge_weights` to
  add a `value_model` namespace alongside `embed_model`.
- Config: add `value_model` block; `embed_model` becomes the key model. Stage B `trainable_params`
  must cover both — extend `.*embed_model.*` to also match `.*value_model.*` (or rename to a shared
  prefix both hit).

**Metrics / success**
- Same diagnostics. Specifically watch **answer-slot value fidelity**: with clean values, does the
  value-recon loss converge and does judged accuracy on date/number questions rise?
- Separate correct/incorrect splits on numeric-answer questions (dates, quantities) — the class most
  sensitive to fidelity.

**Decision gate:** if fidelity improves but accuracy still lags and diagnostic #2 shows answer-slot
weight≈0, the residual is **alignment** — go to Stage 3. (Decoupling fixes content, not *which position*
gets read.)

---

## Stage 3 — Span readout (stacked on Stage 2)

**Goal:** fix alignment. Keys and values are position-aligned: retrieve slot `t` by its key, get slot
`t`'s value. If the high-query-similarity position is the cue token ("born"), you read the cue, not the
answer ("1863"). Span readout reads a *window* so the answer rides in even when keyed on context.

**Changes**
- **Neighbor-window value gather.** When position `t` is retrieved, also pull values of `t ± w`
  (small window). Concatenate or pool into the retrieved value. Clean per-token values (Stage 2) +
  neighbor window = the date surfaces even when the match was on the surrounding context.
- Alternative / complement: **span keys** — pool keys over a window so a single key represents a local
  span, reducing the low-similarity-lone-token problem at the source.

**Code touch-points**
- `models/memory.py` — the top-k gather (`mem_lookup` / `mem_lookup_two_pass`): after `top_k_indices`,
  expand to `[..., K, 2w+1]` neighbor indices, gather `mem_v` over the window, pool. Respect doc
  boundaries via `mem_mask` (don't span across documents).
- New config: `memory.span_window: w`.

**Metrics / success**
- Diagnostic #2 answer-slot weight should stop mattering — even a low-weight cue slot now drags in the
  adjacent answer value. Expect the largest jump on questions where the answer token is semantically
  dim relative to the question (dates, IDs, bare numbers).

---

## Sequencing summary

| Stage | Fixes | Key change | Gate to next |
|-------|-------|-----------|--------------|
| Diagnostics | isolates mode | oracle-memory, answer-slot weight, DLA | run continuously |
| Control | recipe-only baseline | old arch (`[14]`, old init, 4 heads, shared embed) + new recipe | reference for Stage 1 |
| 1 | dilution + attribution | zero-init o_proj, `mem_layers=[9,14,20,27]`, freeze main | lift **over control** + clean channel |
| 2 | fidelity | base-LM value model (trainable), K/V decouple (value-recon loss deferred) | values clean but weight≈0 |
| 3 | alignment | span/neighbor readout, respect doc boundaries | — |

Each stage stacks on the prior. Diagnostics run every stage so we always know which failure mode is
live and never fix the wrong thing.

## Open questions / risks

- Zero-init early-step zero-grad transient — confirm it self-heals; if training stalls, warm o_proj
  with a tiny nonzero init instead of exact zero.
- Two embed models double doc-encode compute; both trainable now (value model no longer frozen), so
  no forward-only offset — full 2× encode + backward. Watch memory/throughput.
- Span readout must not cross document boundaries (`mem_mask`) or it pollutes values with the next doc.
- Multi-layer memory multiplies retrieval cost per forward — watch throughput regression vs the
  Pareto numbers from the throughput campaign.

---

## Handoff — implementer guide

Everything an agent needs to build and run this without re-deriving the above.

### Repo orientation (read first)
- **Branch:** work on `grounding-exps` (branched off `sim-pairs`; a PR for it should target `sim-pairs`,
  not `main`, or the diff includes 44 unrelated commits). `sim-pairs` itself is PR #40.
- **README.md** — training/eval scripts, Hydra override syntax, launch instructions.
- **Auto-memory** (`memory/MEMORY.md` and linked files, esp. `sim-pairs-campaign.md`, `rohun-tpus.md`,
  `pareto-eval-study.md`) — TPU pool, campaign pitfalls, launcher scripts. Read before launching.
- Core code paths already cited per stage: `models/memory.py`, `models/memory_utils.py`,
  `models/qwen3_mem_embed.py`, `losses/` (`registry.py`, `doc_access_loss.py`), `trainer/trainer.py`,
  `configs/model/qwen3_mem_embed.yaml`, `configs/trainer/staged_sim.yaml`.

### Target evals
**The eval sets that matter for these runs: MS MARCO, HotpotQA, MuSiQue** (the MSA variants —
`gen_large_mem_msa_msmarco_v1`, `gen_large_mem_msa_hotpotqa`, `gen_large_mem_msa_musique` in
`configs/eval_set/msa_evals.yaml`). All metrics of record, oracle tasks, and the eval box target these
three. NQ was only the diagnostic that *found* the phenomenon — not a target here.

### Baselines / reference numbers (reproduce before changing anything)
Current model: `mem_layers=[14]`, `mem_top_k=128`, `mem_num_heads=4`, `mem_v_dim=1024`, single shared
`Qwen3-Embedding-0.6B`. The reading-is-the-bottleneck phenomenon was established on **NQ** (corpus 0.289
≈ oracle-memory 0.281 ≈ 0.250 bs=1 with `doc_access_acc=1.0`; `4903c27`, `1d4b888`, task
`gen_embed_msa_nq_oracle`). **Those NQ numbers are origin evidence, not targets.**

For MS MARCO / HotpotQA / MuSiQue, no reference numbers exist yet — measure them on the **control run**
(the baseline; ignore the original sim-pairs checkpoint):
- **corpus** (full retrieval) — run each of the three msa_evals tasks.
- **oracle-memory** (gold `pos_doc` only into memory) — DONE: `gen_embed_msa_{msmarco,hotpotqa,musique}_oracle.yaml`.
- ~~oracle-context~~ — dropped 2026-07-03.

**First task:** stand up the diagnostics + oracle-memory tasks, then measure them on the control run's
early checkpoints (corpus + oracle-memory on all three datasets) to lock the reference point.
Stage 1/2/3 runs can launch in parallel, but treat them as kill/keep against what the control + oracle
diagnostics reveal.

### Ordered implementation checklist
1. **Diagnostics harness (blocks everything).**
   - Oracle-memory eval task (memory bank from gold `pos_doc`) — DONE:
     `gen_embed_msa_{msmarco,hotpotqa,musique}_oracle.yaml` (ports of `gen_embed_msa_nq_oracle`).
   - Answer-slot softmax-weight logging (uses `mem_pos_logits`/`pos_slot_indices` already in
     `mem_lookup_two_pass`), split by judge correct/incorrect.
   - DLA / layer-14 memory-ablation hook.
   - wandb training telemetry (see the telemetry section) — extend the weight-0 metric path in
     `trainer/trainer.py` where `doc_access_acc` logs; per-layer series from the `all_aux_data` list.
2. **Training recipe config** — `configs/trainer/staged_ground.yaml`: 2 stages, frozen main,
   Stage A 1k @1e-4 (`.*mem_.*`), Stage B →150k @5e-5 cosine (`.*mem_.*` + embed), `checkpoint_interval: 2000`.
3. **Stage 1 arch** — `memory.mem_o_proj_zero_init` flag in `add_memory_layer` (`memory_utils.py`);
   `mem_layers: [9,14,20,27]`. Model-config variant for the run.
4. **Continuous-eval box** — stand up 1–2 `rohun*` TPUs on the eval-box loop (`scripts/misc/box_run.sh`,
   `sim_eval_box.py`, `sim_find_latest.py`), pointed at the run's GCS checkpoint dir.
5. **Launch control run** (`ground_control`: old arch + new recipe + QA-only) and **Stage-1 run**
   (`ground_s1_zeroinit_4layer`) together; both feed the eval box. Apply the Stage-1 gate = lift over control.
6. **Stage 2** — second embed model (base `Qwen3-0.6B`) for values, config `value_model` block,
   `embed_forward`/`load` split, extend `trainable_params`. (Value-recon loss deferred; add the passive
   recon-accuracy metric.)
7. **Stage 3** — span/neighbor gather in `memory.py`, `memory.span_window`, doc-boundary masking.

### How to run
- **Train** (single VM): adapt `scripts/embed/train_hard_neg_think.sh` — `uv run train.py
  model=qwen3_mem_embed dataset=qa_hard_neg_think_sft4b trainer=staged_ground
  model.memory.mem_top_k=128 trainer.checkpoint_interval=2000
  +trainer.run_name="ground_s1_zeroinit_4layer"` (+ per-stage model overrides: zero-init, mem_layers,
  K/V split, span). Multi-VM: `scripts/infrastructure/multi-vm-tpu-run.sh`. Checkpoints → GCS
  (`gs://memory-layers-training/<run_name>/...`).
- **Eval box:** the sim-pairs `box_run.sh` pattern — polls newest checkpoint, evals **msmarco + hotpotqa
  + musique**, scores with the local Qwen3-8B judge, logs binary accuracy to wandb keyed by step. Free
  the TPU between evals. (sim_eval was trimmed to 2 datasets for a rotation window — retarget to these 3.)
- **Eval a single checkpoint manually:** `uv run eval.py checkpoint_dir=gs://... '~eval_set@evals=pretraining'
  '+eval_set@evals=msa_evals' tp_devices=1` (README "Hydra Eval Set Override Syntax").

### Conventions
- **wandb run naming:** `ground_s{1,2,3}_<change>` (e.g. `ground_s1_zeroinit_4layer`) so accuracy curves
  are comparable across experiments. Same wandb project as the campaign.
- **Metric of record:** per-checkpoint binary LLM-judge accuracy on msmarco / hotpotqa / musique.
  Everything else is diagnostic.
- **Commit cadence:** small, described commits on `grounding-exps`; end messages with the Co-Authored-By
  trailer. Push when asked.
- **Env:** `.env` at repo root (`HF_TOKEN`, `HF_USERNAME`, `WANDB_API_KEY`, `GCS_*`, `GIT_PAT`). `uv` must
  be on PATH for the LLM judge (`76d0547`). **`gh` is NOT installed** — for PRs use `GIT_PAT` + the GitHub
  REST API.

### Known pitfalls (all bit the sim-pairs campaign — see memory)
- **Stage-boundary resume bugs** (`1ec9d8c`, `b9df852`, `0d6cdf5`): stage settings/`ce_weight` can be lost
  on mid-stage resume. The 2-stage frozen-main recipe minimizes boundaries; still verify grad-norms flip
  at A→B.
- **TPU `/dev/vfio` busy** (`a04fc00`, `560b8de`): lingering vLLM judge wedges the device — eval box must
  free the TPU (`free_tpu_devices()`) before each eval.
- **Orbax rotation window** (`8387d1e`): keep-N must outlast one eval pass or the box races a deleted
  checkpoint. Tune keep-N vs eval time.
- **HF 429 storms** (`2ae5336`, `a45256c`): huge datasets / too many grain workers trigger throttling.
- **Throughput:** multi-layer memory and two trainable embed models raise per-step cost — sanity-check
  against the Pareto/throughput baselines before committing to a 150k run.

### Decisions (all resolved)
- **Training data:** `dataset=qa_hard_neg_think_sft4b` (ETD-free hard-neg-think SFT; sources
  science/diverse hard-neg-think + `combined_hard_neg_sft4b` + `multihop_qa_sft`). Template
  `scripts/embed/train_hard_neg_think.sh`, overriding `checkpoint_interval=2000` and `mem_top_k=128`.
- **top_k:** fixed 128 across all runs; swept post-hoc via `MEM_TOP_K`, never folded into a stage.
- **Evals:** msmarco / hotpotqa / musique (`msa_evals`).
- **TPUs:** 4 parallel training runs (control + 3 stages) + 1 `rohun` eval box + 2 nanny-provisioned
  `v6e-8` eval boxes.

---

## Results & insights log (living — append as runs report)

**This is a living document.** As diagnostics land, runs report accuracy curves, or insights emerge,
**append them here** (dated, with the run name / wandb link / checkpoint step) and update the affected
plan sections above (baselines table, decision gates, stage changes). Don't spin up a separate results
doc — keep the hypothesis, the plan, and what actually happened in one place so the next reader sees the
full thread. Kill/keep decisions on the parallel runs go here too, with the reason.

**2026-07-03 — Foundation landed (code, not yet run).** Built the buildable, unambiguous
layer of the plan on `grounding-exps`. Nothing run yet (local env can't install — `libtpu`
is Linux/TPU-only; all compose/exec validation must happen on a TPU box). YAML syntax-checked
and Python byte-compiled locally.

- **Recipe:** `configs/trainer/staged_ground.yaml` — 2-stage frozen-main (A 0→1k @1e-4 `.*mem_.*`;
  B 1k→150k @5e-5 cosine `.*mem_.*`+`.*embed_model.*`), `checkpoint_interval: 2000`, in-loop
  eval off (`eval_interval` huge — judged accuracy comes from the eval boxes).
- **Stage-1 arch:** `mem_o_proj_zero_init` flag added to `memory_utils.py::add_memory_layer`
  (breaks `mem_o_proj` out of the shared `*0.02` init loop → `jnp.zeros`); default `false` added
  to `configs/model/qwen3_mem_embed.yaml`.
- **Launch scripts:** `scripts/embed/train_ground_control.sh` (`ground_control`, old arch + new
  recipe) and `train_ground_s1.sh` (`ground_s1_zeroinit_4layer`, `mem_o_proj_zero_init=true`,
  `mem_layers=[9,14,20,27]`). CLI-override style on the base model config.
- **Oracle-memory evals:** `gen_embed_msa_{msmarco,hotpotqa,musique}_oracle.yaml` (ported from
  `gen_embed_msa_nq_oracle`, doc→memory bank).
- **Oracle-context evals: dropped** (2026-07-03 decision). Not building the doc-in-context
  isolator. Reference points are **corpus + oracle-memory** only.
- **Eval suite:** `configs/eval_set/ground_evals.yaml` bundles the 6 tasks the eval box runs —
  3 corpus (existing `gen_large_mem_msa_{msmarco_v1,hotpotqa,musique}`) + 3 oracle-memory (new).
- **Write-magnitude telemetry:** `memory.py::memory_layer` now emits `mem_write_norm` (‖o‖) and
  `mem_write_ratio` (‖o‖/‖x‖) into `aux_data` under `collect_aux`; registered as weight-0 metrics
  in `losses/mem_telemetry.py` and enabled in `staged_ground.yaml`. Logged as
  `train/mem_write_norm` / `train/mem_write_ratio` (across-layer means). Under zero-init these
  start at 0 and should climb — the direct "channel is alive" check.

**2026-07-03 (pass 2) — all four runs + eval box implemented (code, not run).** Everything
needed to launch control + Stages 1–3 in parallel. Byte-compiled + YAML/bash syntax-checked;
NOT executed (TPU-only). All arch changes are gated so control/Stage-1 are byte-identical to
before. On-box validation still required (sharding on the new span gather, value-model
dims/tokenizer) before trusting a long run — smoke each with a tiny step count first.

- **Stage 2 (K/V split):** optional `value_model` namespace (base `Qwen3-0.6B-Base`, bidirectional,
  trainable) supplies memory VALUES via new `value_forward`; `embed_model` stays the key model.
  Gated on `'value_model' in cfg`. `load`/`forward`/`split_weights` extended to 3 namespaces
  (`models/qwen3_mem_embed.py`). Stage-B `trainable_params` gained `.*value_model.*` (no-op when
  absent). Configs `qwen3_mem_embed_ground_s2.yaml`, launch `train_ground_s2.sh`
  (`ground_s2_kv_split`). Value model uses no conv (1 token = 1 value slot, aligned with key path
  at conv stride 1). *Deferred:* value-recon passive metric + K/V-cosine/anisotropy (need mem_k/
  mem_v vectors or the value lm_head in the loss path).
- **Stage 3 (span readout):** `span_readout()` in `memory.py` — re-gathers `mem_v` over a t±`span_window`
  window from the retrieved indices, masks to valid + same-document slots (via `mem_mask` +
  injected `effective_doc_len`), mean-pools, replaces the retrieved values. Gated on
  `span_window>0` (default 0 = no-op). Config `qwen3_mem_embed_ground_s3.yaml` (span_window=1),
  launch `train_ground_s3.sh` (`ground_s3_span`). Emits `mem_boundary_straddle` telemetry.
- **Telemetry suite (weight-0, per-layer):** `memory.py` emits `mem_write_norm`, `mem_write_ratio`,
  `mem_topk_entropy`, `mem_effective_slots`, `mem_top1_weight`, `mem_o_proj_norm` (+ `mem_top_k_probs`)
  from the true post-softmax `top_k_scores` (path-agnostic — resolves the earlier logits-vs-probs
  ambiguity; on the TPU shard-map path `mem_top_k_logits` is probs, so I compute from `top_k_scores`
  instead). `losses/mem_telemetry.py` registers these + `mem_pos_weight_mass` + `mem_hit_rate`
  (joined to the positive-doc mask like `doc_access_acc`). Per-layer series enabled by a small
  `losses/registry.py` change: a metric may return `{l0..ln, mean}` → logged as `train/<name>/lN`.
  All enabled at weight 0 in `staged_ground.yaml`. *Deferred:* cross-head query cosine, per-head
  Jaccard, cross-layer output cosine (contracting the sharded head axis is risky untested), and
  per-param-group grad norms (needs the `grads` tree filtered in `_train_step`).
- **DLA hook:** env `MEM_ABLATE_LAYER=<idx>|all` zeros a chosen layer's memory write (`o`) via the
  per-layer `_mem_layer_idx` tag. Measure the answer-token logit/accuracy delta by running eval
  twice (with/without the env). The *hook* is in; the eval-time diff driver is a run-two-evals
  recipe, not new code.
- **Answer-slot weight (eval, correct/incorrect split):** still deferred — it's genuinely an
  eval-harness diagnostic needing the judge label per example; the train-time proxy
  (`mem_pos_weight_mass`) is implemented and covers most of the signal.
- **Continuous-eval box:** `scripts/misc/ground_eval_box.py` (+ `ground_box_run.sh`) — retargets the
  sim-pairs loop to `ground_evals` (3 corpus + 3 oracle-memory), polls newest GCS checkpoint,
  frees the TPU between evals (`free_tpu_devices`), uploads per-step JSON, optional wandb curve.
  Shard across 3 boxes by `GROUND_EVAL_DATASETS=msmarco|hotpotqa|musique`; all boxes watch all 4
  runs (GCS-result idempotency prevents double work). Default milestone 2000 for a dense curve.

**2026-07-03 (pass 3) — remaining telemetry + smoke test.** Implemented the previously-deferred
telemetry (except value-recon): K/V cosine + value anisotropy (`_kv_bank_telemetry`), cross-head
query cosine (safe head-axis reduction) + cross-layer output cosine (per-layer write-direction
vectors), per-param-group grad norms (mem/embed/value/main, in `_train_step`). Added the eval-time
answer-slot correct/incorrect split: `generation_embed` now attaches per-example positive-slot
weight mass to each result (`answer_slot_diag`, on for oracle tasks) for a judge-verdict join, and
— correctness fix found while wiring it — `generation_embed` now uses `value_forward` for values
when a `value_model` is present (else Stage 2/3 oracle eval read values off the key model).
**Smoke test on `rohun` TPUs (in progress):** caught two config bugs before wasting compile time —
(1) `standard.yaml` pulls a nonexistent `eval_set/standard` (fixed: `staged_ground` overrides the
evals default); (2) `steps=6` tripped the increasing-stage-max_step check (a smoke-arg issue, not
code). Re-running control/s1/s2/s3 (6 steps, crossing the A→B boundary) one per box.

**2026-07-03 (pass 3, cont.) — HF 429 root-cause + fix.** The smoke stalled on HuggingFace
`429 rate limit (1000 req / 5 min)`. Root cause: the training data (`qa_hard_neg_think_sft4b`,
~75 GB) is **streamed** (won't fit the 67 GB boot disk), so the grain pipeline re-lists shards
via the HF API — `num_workers: 16` × 4 datasets × 4 concurrent runs × auto-rebuild-on-timeout
blew the quota. Fix: the v6e-8 hosts have **709 GB `/dev/shm` (1.3 TiB RAM free)** — new
`scripts/misc/precache_hf.sh` snapshot-downloads the datasets into `/dev/shm/hf` (sequential,
few API calls), and the training launch scripts now run `HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1`
→ zero HF API calls, no 429 ever. tmpfs is volatile → re-run the pre-cache on a fresh/preempted
box (~76 GB, one-time, ~10-15 min). **Launch order per box: `precache_hf.sh` → train script.**
(Note: gcloud/IAP SSH also got flaky under my repeated connections — intermittent `code 255`;
back off between connections.)

**2026-07-03 (pass 4) — smoke PASSED + data-loading solved.** The Stage-2 smoke ran all 6
steps crash-free on `rohun-v6e-8-2`, crossing the A→B stage boundary: `Loss 2.17→2.03→2.23`,
`value_model.mem_k_proj/mem_v_proj` + `value_model.layers.*` in the Stage-B trainable set (value
model unfroze), all 4 mem layers present, `Saved step 6 to gs://...`. So config compose, model
load **incl. the Stage-2 value model**, forward+backward, the stage transition (embed+value
unfreeze + optimizer rebuild/recompile), and all weight-0 telemetry all execute. (s3/span smoke
running to validate `span_readout`; control/s1 are strict subsets of what s2 exercised.)

Getting there surfaced a chain of **infra** issues (none in the arch code) and their fixes:
- **HF 429 (1000 req/5min):** streamed 75 GB data × `num_workers:16` × 4 concurrent runs ×
  auto-rebuild re-listing blew the quota. → `eval_set/none` (no in-loop eval-dataset loads).
- **`/dev/shm` isn't durable:** a periodic cleanup on these VMs wipes it mid-run (confirmed: 341 GB
  gone at a timestamp with no reboot), so the RAM-cache route is unsafe.
- **Offline needs local files, not repo ids:** `load_dataset(repo_id)`/`snapshot_download` still
  hit the Hub in `dataset_module_factory` under `HF_HUB_OFFLINE`.
  → **Final approach:** `scripts/misc/precache_hf.sh` downloads a **subset of parquet shards**
  (`GROUND_DATA_SHARDS`/`GROUND_DATA_FRAC`, default half) to the **boot disk** (`~/hf_parquet`,
  durable, ~half of 75 GB fits the 67 GB disk) + the model checkpoints; `qa.py`'s offline branch
  streams them via `load_dataset("parquet", data_files=..., streaming=True)` → zero Hub calls, no
  arrow doubling, no shm dependency. Launch scripts run `HF_HUB_OFFLINE=1`.
- **gcloud ssh ops:** launch detached jobs via **`tmux new-session -d`** (bare `... &` hangs the
  ssh channel → generic exit 255); a wedged/overloaded box also 255s until killed/rebooted.

**Launch (user drives — needs the TPUs):** per the allocation table, one run per `rohun` box.
**Per box first:** `bash scripts/misc/precache_hf.sh` (half the shards; set `GROUND_DATA_FRAC=1`
for the full data if a data disk is attached), then the train script (already runs `HF_HUB_OFFLINE=1`):
`train_ground_control.sh`, `train_ground_s1.sh`, `train_ground_s2.sh`, `train_ground_s3.sh`;
`rohun`-5 + 2 nanny boxes run `ground_box_run.sh` sharded by dataset. **Smoke-test each run for a
few steps first** (config compose + Stage-2 value-model load + Stage-3 span gather are untested).
