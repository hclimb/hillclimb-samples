# multihop_hard_neg_full x train_ground_s1 architecture, 3 memory layers

**Date:** 2026-08-02 **Author:** rohunagrawal (with Claude) **Status:** RUNNING — launched,
confirmed stable through the initial compile *and* the Stage A→B transition (embed model
unfreeze) with no OOM; too early for any quality signal.

## Conclusion, first

`train_ground_s1.sh`'s architecture (zero-init `mem_o_proj`, multi-layer memory, frozen main
throughout) combined with the `multihop_hard_neg_full` recipe (200 hard negs/query,
`mem_batched_isolation`, `doc_access_per_query_loss` — see
[2026-08-02-hard-neg-full-efficient-retrieval.md](../implementations/2026-08-02-hard-neg-full-efficient-retrieval.md))
at **3** memory layers instead of the original 4: compiled and ran past the first
`checkpoint_interval` boundary (step 200) with no OOM — a real, flagged risk beforehand (3x the
per-row retrieval compute/memory of the single-layer recipe, which already ran close to the
~31GB/chip ceiling). Step time (~2.0-2.3s/it) is barely above the single-layer recipe's, despite
3x the memory-layer count — retrieval evidently isn't the dominant per-step cost here. `Loss`
(2.11-2.14, `CE` component 1.83-1.86 at `ce_weight=1.0` from step 0 — safe since main is frozen
throughout in this recipe) is real and non-degenerate. No accuracy signal yet; this is an
infra/launch result, not a quality one.

## Hypothesis & motivation

Rohun: try the grounding architecture (`train_ground_s1.sh` — zero-init memory write, multiple
memory layers spread across the network for iterative re-retrieval) on the multihop hard-negative
recipe instead of the single-memory-layer default, with 3 layers instead of the original 4, and
telemetry on. Open question going in: does 3x the per-row retrieval work (independent per-layer
banks) fit the same ~31GB/chip budget the 1-layer recipe was already close to.

## Setup

- **Model:** `qwen3_mem_embed` — main `Qwen/Qwen3-4B`, `mem_layers=[9,18,27]` (Qwen3-4B has 36
  hidden layers; `train_ground_s1.sh`'s 4-layer set is `[9,14,20,27]` — same start/end anchors,
  evenly trisected here instead of an uneven 4-way spread or an arbitrary 3-of-4 subset — not
  verified against any specific ablation), `mem_o_proj_zero_init=true`, `mem_top_k=128`,
  `mem_placement=after_attention` (model default).
- **Recipe:** `staged_ground_batched_isolation` (new, `configs/trainer/staged_ground_batched_isolation.yaml`)
  — `staged_ground`'s 2-stage, frozen-main-throughout schedule (Stage A 0-1000: `.*mem_.*` only,
  `ce_weight=1.0`; Stage B 1000-150000: `+.*embed_model.*` `+.*value_model.*`), with
  `doc_access_loss` → `doc_access_per_query_loss` in both stages (same reason as the 4-stage
  sibling: `doc_access_loss` needs a cross-batch grid `mem_batched_isolation` never builds).
  Telemetry (the full weight-0 `mem_*` diagnostic block) is inherited from `staged_ground`'s own
  base `aux_losses` — not re-specified, carries through via normal dict-merge (only the
  `training_stages` list is overridden).
- **Data:** `multihop_hard_neg_full` — unchanged from the single-layer recipe (`batch_size=8`,
  200 hard negs/query, `num_chunks_per_doc=256`, `doc_chunk_seq_len=256`).
- **What's deliberately NOT changed:** `trainer.stop_grad_frozen` (default `false`) — main is
  frozen in both stages here, so enabling it would skip backward through the entire 4B backbone
  (quality-neutral per `trainer.py`'s own docstring) and could matter more now that 3 memory
  layers compete for the same budget. Not enabled tonight — a new lever beyond what was asked for,
  and a clean follow-up if this run turns out to be memory/speed constrained.

## Reproducibility

- Command: `TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers bash scripts/infrastructure/multi-tpu-box-run.sh tpu-v6e-slice-mig-lhxn=scripts/embed/train_multihop_ground3layer.sh tpu-v6e-slice-mig-qvlq=scripts/embed/train_multihop_ground3layer.sh`
- Commit: `48a9041617aa914fae63ff5ae2b247ceca8f02b5` (`datagen` branch) + uncommitted working-tree
  changes (same tree as the single-layer recipe, plus `configs/trainer/staged_ground_batched_isolation.yaml`
  and `scripts/embed/train_multihop_ground3layer.sh`).
- wandb: run name `multihop_ground3layer_batched_iso_topk128_bs8`, `wandb_run_id=auto`.
- Checkpoint: `gs://memory-layers-training/multihop_ground3layer_batched_iso_topk128_bs8-2026-08-02-18-49-13/qwen3_mem_embed/200/` (first checkpoint, confirmed on both hosts).
- TPU: v6e-8 flex-start 2x4 slice, `tpu-v6e-slice-mig-{lhxn,qvlq}`, `europe-west4-a`. **Replaced**
  the single-layer `multihop_hard_neg_full_batched_iso_topk128_bs8_docaccesspq` run on this same
  slice (only one box available) — that run's checkpoints remain at
  `gs://memory-layers-training/multihop_hard_neg_full_batched_iso_topk128_bs8_docaccesspq-2026-08-02-17-31-31/`
  if resuming it later.

## Results so far

| Step | Loss | CE (w=1.0) | step time | Notes |
|------|------|------------|-----------|-------|
| ~210 | 2.1117-2.1377 | 1.8357-1.8645 | ~2.0-2.3s/it (post-compile) | First checkpoint at step 200, no OOM |
| 400 | 1.7160 | 1.5613 | ~2.0-2.3s/it | Genuine downward trend |
| 600 | 1.7903 | 1.5684 | ~2.0-2.3s/it | Roughly flat, normal step-to-step noise (log_interval samples a single step, not an average) |
| 800 | 2.3172 | 1.9256 | ~2.0-2.3s/it | Noisy uptick, same reason — not a real regression, confirmed by step 1000 |
| 1000 | 1.4106 | 1.2188 | ~2.3s/it | **Stage A→B transition** (embed_model + value_model unfrozen) — landed right at this step |
| 1011-1020 | 1.7963 | 1.6038 | 5.76s/it decaying to 2.49s/it | **Crossed the transition cleanly, no OOM** — confirms the general `trainer.py` opt_state fix (found for the single-layer recipe) applies here too, and that 3 memory layers + a newly-unfrozen embed model still fit the ~31GB/chip budget |
| 1200-4800 | 1.17-2.62 | 1.05-2.36 | steady ~3.08s/it | Noisy but bounded, single-step `log_interval` samples (not averaged) — no drift, no NaN, no error, both hosts symmetric throughout |
| 5001 | 1.6783 | 1.6204 | ~3.08s/it | **Passed step 5000 cleanly** — the exact step that OOM'd 3x on the single-layer recipe's 4-stage schedule (`staged.yaml`'s Stage 0→1 boundary). Not a landmine here: `staged_ground_batched_isolation` only has one transition, already crossed at step 1000. |
| 5200-7000 | 0.92-2.36 | 0.80-2.19 | steady ~3.08s/it | Same noise band, floor dropped slightly (two sub-1.0 samples: 0.9178 @ 5200, 0.9570 @ 2600) — plausibly early real signal but not distinguishable from noise yet at this sampling density |
| 7000-15000 | 0.60-2.39 | 0.45-2.24 | steady ~3.08s/it | Sub-1.0 samples now recurring regularly (7800, 9400, 10000, 12200, 13200, 13800 all <1.0, lowest 0.5961 @ 7800), but interleaved with samples back up at 2.0-2.4 — still reads as noise around a roughly flat mean rather than a monotonic trend; would need an actual moving-average or eval metric to say more |
| 15000-20000 | 0.28-2.35 | 0.24-2.24 | steady ~3.08s/it | The low tail is now setting new records repeatedly and fast: 0.4835 @ 17600, 0.4112 @ 18200, 0.2768 @ 19800 — three new floors inside 2200 steps, distinct from the flatter floor seen through step 15000. Upper end (2.0-2.35, e.g. step 20000) is unchanged though, so the distribution is widening, not just shifting down — consistent with real (if uneven) learning progress but still not a substitute for an eval number |
| 20000-29400 | 0.25-2.74 | 0.24-2.60 | steady ~3.08s/it (one qvlq-only 3.35s/it blip @ 20400, transient) | Both tails keep stretching: new floor 0.2459 @ 22400, new ceiling 2.7356 @ 25000 (previous ceiling was 2.62 @ step 2000). Consistent with widening per-batch difficulty variance rather than a shift in either direction — reads as real training dynamics, not an anomaly |
| 29400-36800 | 0.30-2.44 | 0.24-2.25 | steady ~3.03-3.08s/it | No new tail records in this window — distribution has settled into the wider band established by step 29400 rather than continuing to stretch. Still zero errors, both hosts symmetric throughout |
| 36800-44400 | 0.29-2.34 | 0.24-2.22 | steady ~3.08s/it | Same settled band continues, no new records either direction. Distribution has now held roughly steady for ~15000 steps (since ~step 29400) after the earlier 15000-29400 widening phase |

Compile took ~14 min for step 1 (consistent with every full-model compile tonight — this is a
fresh trainable-params mask + 3x the memory-layer graph vs. the single-layer recipe). The Stage
A→B recompile at step 1000 shows the same decay pattern (elevated step time immediately after,
settling back to steady-state within ~10 steps) as every other transition tonight.

## Interpretation

The main open question going in — does 3 memory layers' independent per-row retrieval fit the
budget — is answered: yes, at `batch_size=8`, with headroom apparently large enough that step
time barely moved versus 1 layer. That's a genuinely useful data point for how much of the
single-layer recipe's step cost was retrieval versus everything else (mostly *not* retrieval,
evidently). Through step 44400 (~30% of the planned 150000 steps, ~30.75h wall clock, ~30.5h
post-compile, 222 checkpoints), the run has been fully stable: no OOM, no error, both hosts
symmetric at every checkpoint throughout the entire run so far. `Loss`/`CE` noise widened over
steps 15000-29400 (floor 0.92→0.25, ceiling 2.35→2.74) then settled into that wider band, holding
steady through 44400 — consistent with a model that found a new (noisier but not worse) training
regime early in Stage B and has been stable in it since, but still not conclusive about actual
quality without an eval box.
