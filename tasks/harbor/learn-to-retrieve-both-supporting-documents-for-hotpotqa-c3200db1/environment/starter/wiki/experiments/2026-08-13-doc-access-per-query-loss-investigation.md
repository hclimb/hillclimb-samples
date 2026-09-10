# Why doc_access_per_query_loss won't go down during multihop hard-neg warm-start finetuning

**Date:** 2026-08-13 · **Author:** claude (session with rohunagrawal) · **Status:**
**RESOLVED** (the original bf16-restore-bug question, arms 0/1) — root cause found + fixed, fix
confirmed working with a large, confident sample. **Arm 2 RUNNING** — a new warm-start from a
different, further-along checkpoint (main_model unfrozen) picked up once rohunagrawal was awake;
see that section below for its own setup/debugging/status, kept live-updated separately from the
resolved arms 0/1 material above.

## Conclusion

**The loss formulation was never the problem — a checkpoint-restore bug was silently re-freezing
the exact weights the loss needed to move.**
`train_multihop_ground4layer_s1warmstart_no_multihop.sh` warm-starts from a checkpoint whose
trainable leaves are bf16, and `utils.py::load_checkpoint`'s restore path was silently discarding
`promote_trainable_to_fp32`'s fp32 promotion on restore — re-trapping `mem_q_proj` / `mem_o_proj`
/ `embed_model.mem_{k,v}_proj` in the bf16-ULP freeze documented in
[2026-08-07-bf16-ulp-freeze-empirical-confirmation.md](2026-08-07-bf16-ulp-freeze-empirical-confirmation.md).
Under that freeze, Adam's per-step update on these ~0.02-magnitude weights is smaller than
bf16's ULP at that magnitude, so it rounds to zero almost every step regardless of how large the
upstream gradient (or loss weight) is — exactly consistent with "reweighting
`doc_access_per_query_loss` 0.1→1.0 had no visible effect." Fixed in one line in
[wiki/implementations/2026-08-13-warmstart-restore-dtype-fix.md](../implementations/2026-08-13-warmstart-restore-dtype-fix.md).

**With the fix, at ~8500 total steps (large samples, not noise):**
- `doc_access_per_query_loss`: **~2.4-2.5 at start → ~1.1-1.5 and still falling**, monotonic by
  thirds in both stage 1 and stage 2 (see Results below).
- `mem_pos_weight_mass` (the requested byproduct metric): **~0.42 → ~0.58-0.65**, same
  monotonic pattern.
- Confirmed via the dispositive `weight/wnorm/*/rms` metric that this is real weight
  accumulation, not noise or an artifact — RMS moves smoothly and continuously under the fix,
  and is frozen to 4 decimal places on the pre-fix baseline.

**One important nuance, not to be lost:** the pre-fix (bugged) arm ALSO improved somewhat over
its ~2900-step run (loss 2.48→1.48ish, `mem_pos_weight_mass` 0.42→0.59) — via `mem_o_proj`'s
zero-init escape hatch, which moves under bf16 regardless. So the dtype fix is a real,
confirmed, worthwhile improvement (faster, more stable, and removes a structural landmine
that silently defeats a previously-shipped fix for any future warm-start), not a case where
nothing was working before. Don't oversell it as "the reason literally nothing was learning."

## Hypothesis & motivation

rohunagrawal's first-pass fix (adding a `staged_ground_batched_isolation_docaccess_warmstart`
stage: `ce_weight` 1.0→0.1, `doc_access_per_query_loss` weight 0.1→1.0 for the first 5000 steps)
didn't move the needle — loss kept fluctuating 1–3 with no downward trend over the first ~50
logged steps of the relaunched run. Asked to (a) figure out if something structural is broken in
how the loss/gradients interact, being creative/exploratory rather than assuming the obvious
"needs more weight/different temperature" framing, and (b) if so, fix it and verify with real
training data overnight.

## Investigation trail

1. **Read the loss + retrieval code first**, to rule out a formulation bug before looking
   anywhere else: `losses/doc_access_per_query_loss.py`, `losses/doc_access_loss.py`,
   `losses/registry.py`, `models/memory.py::mem_lookup_batched`. Structurally sound: the FULL
   pre-top-k logits (`aux_data["mem_scores"]`, shape `[B,T,N,m_per_query]`) are exposed
   specifically so this loss's gradient reaches every candidate doc, including positives outside
   the current top-k — not just the discretely-selected top-k values that also feed CE. Weight
   bump from 0.1→1.0 is a real, well-formed gradient signal at the loss level. This is what
   pointed the investigation away from "the loss is wrong" and toward "the weights aren't moving
   regardless of loss signal."
2. **Recognized the symptom** (noisy, fluctuating, no downward trend despite a real gradient
   signal, on retrieval-scoring projections at ~0.02 magnitude) as the exact shape of the
   already-documented bf16-ULP freeze
   ([2026-08-07-bf16-ulp-freeze-empirical-confirmation.md](2026-08-07-bf16-ulp-freeze-empirical-confirmation.md),
   [2026-08-07-doc-access-jump-is-data-not-model.md](2026-08-07-doc-access-jump-is-data-not-model.md)) —
   both explicitly call out `mem_q_proj`/`embed_model.mem_k_proj` at ~0.02 magnitude as the
   "moves but doesn't accumulate" category.
3. **Checked whether the known fix (`promote_trainable_to_fp32`) is even active on this
   branch** — yes, `train.py` calls it unconditionally before `setup_optimizer`. So the fix
   should already apply. Confirmed it DID run: the live run's log shows
   `[promote_fp32] promoted 344 leaves bf16 -> fp32`, including the exact leaves above.
4. **But the run's own weight-monitor snapshot, taken moments later (post-checkpoint-restore),
   showed those same leaves back at `dtype=bfloat16`.** That's the smoking gun: something
   between the promotion and the first training step silently downcasts them again. Traced it to
   `utils.py::load_checkpoint`'s warm-start restore path (`restore_and_reshard`), which re-applies
   sharding via `jax.device_put(val, a.sharding)` but never casts `val`'s dtype to match the
   (already fp32-promoted) target `a`. A checkpoint saved in bf16 — this warm-start's source,
   `ground_s1_zeroinit_4layer_no_multihop` step 26000, evidently is — restores as bf16 no matter
   what the current run just promoted its OWN fresh init to.
5. **Fixed + tested** — see the implementation note for the one-line fix, the regression test
   (`tests/test_warmstart_restore_dtype.py`), and confirmation that the test fails without the
   fix and passes with it.

## Setup

- **Architecture:** `qwen3_mem_embed`, `mem_layers=[9,14,20,27]`, `mem_o_proj_zero_init=true`,
  `mem_top_k=128`, `per_query_isolation=true`, `isolation_group_size=1`,
  `mem_batched_isolation=true`, `mem_collect_full_scores=true`.
- **Dataset:** `multihop_hard_neg_full` (200 hard negs + ~4 positives per question,
  `batch_size=8`, `num_chunks_per_doc=256`).
- **Trainer:** `staged_ground_batched_isolation_docaccess_warmstart` (new tonight) — stage 1
  (0→5000 steps): `ce_weight=0.1`, `doc_access_per_query_loss` weight `1.0`; stage 2
  (5000→`trainer.steps`): back to `ce_weight=1.0`, weight `0.1`. Both stages: trainable
  `.*mem_.*|.*embed_model.*|.*value_model.*`, `learning_rate=2e-4` cosine, `warmup_frac=0.005`.
- **Warm start:** `ground_s1_zeroinit_4layer_no_multihop` step 26000
  (`gs://memory-layers-training/ground_s1_zeroinit_4layer_no_multihop-2026-08-11-00-52-33`).
- **Box:** `tn-v6e-8-0`, project `memorylayers`, zone `europe-west4-a`.

## Results

### Arm 0 — pre-fix (bugged), for reference

Run `multihop_ground4layer_s1warmstart_no_mul-2026-08-13-00-32-22`, ran to step 2917 before
being killed. Weight-monitor at step 0 (post-restore): `mem_q_proj`/`mem_o_proj`/
`embed_model.mem_{k,v}_proj` all `dtype=bfloat16` despite `promote_fp32` having just promoted
them.

**Correction to my initial read:** I first eyeballed the console `Loss:` print near step 2917
(2.5346) against a couple of early values and called the trend "flat." Pulling the actual
isolated `train/doc_access_per_query_loss` history from wandb (not the combined, CE-mixed
console `Loss`) shows that's an overstatement — there IS a real, if slow and very noisy (std
0.94, range 0.005–6.33 per logged point), downward trend across the full run, by decile
(steps, ~290 each):

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `doc_access_per_query_loss` | 2.48 | 2.10 | 2.11 | 1.90 | 1.89 | 1.98 | 1.75 | 1.70 | **1.48** | 1.97 |
| `mem_pos_weight_mass` | 0.42 | 0.51 | 0.50 | 0.55 | 0.55 | 0.51 | 0.55 | 0.58 | 0.59 | 0.54 |

So **even under the bf16-restore bug, this run was learning** — `mem_o_proj` (zero-init, always
escapes the ULP trap regardless of dtype) plus whatever fraction of `mem_q_proj`/`mem_k_proj`
elements happened to flip bits (~0.02–0.04%/step, accumulating over thousands of steps) was
apparently enough to move `mem_pos_weight_mass` from 0.42→~0.55-0.59 and the loss down by roughly
a third. Slower and much noisier than it should be with proper fp32 throughout, almost certainly,
but "no real downward trend" doesn't hold up against the full-run numbers — it likely reads that
way only over a short/early window, which is consistent with what I saw when comparing the first
~800 steps of both arms head-to-head below.

### Arm 1 — post-fix

Killed arm 0 (had reached step 2917, loss still flat at 2.5346 — 3000 steps of stage-1 warmup
with no visible improvement, consistent with the freeze) after confirming no checkpoint from it
is worth keeping (all trapped in the same bug). Verified clean TPU release (`tmux kill-session`,
no orphaned `venv/bin/python`, `fuser -v /dev/vfio/*` empty) before relaunching identically
(same command, same warm-start checkpoint) with the dtype fix synced.

**dtype fix confirmed live**, straight from the new run's weight-monitor snapshot:

| weight | before (arm 0) | after (arm 1) |
|---|---|---|
| `main_model.layers.14.mem_q_proj` | `dtype=bfloat16` `ulp=1.414e-4` MIXED | `dtype=float32` `ulp=2.157e-9` OK |
| `main_model.layers.14.mem_o_proj` | `dtype=bfloat16` `ulp=4.565e-5` OK | `dtype=float32` `ulp=6.966e-10` OK |

ULP dropped ~5 orders of magnitude — now far below any realistic per-step Adam update, so the
truncation-to-zero failure mode is gone regardless of what the loss/gradient trend turns out to
be. `wandb`: `multihop_ground4layer_s1warmstart_no_mul-2026-08-13-02-47-25`.

**Step-0 `wchanged` confirms it's not just the dtype label — elements are actually accumulating
now**, a much stronger signal than the dtype alone:

| weight | arm 0 (bf16) `n_changed` | arm 1 (fp32) `n_changed` |
|---|---|---|
| `main_model.layers.14.mem_o_proj` | 4317/10,485,760 (0.04%) | **10,475,813/10,485,760 (99.9%)** |
| `embed_model.mem_k_proj` | 179/1,048,576 (0.02%) | **1,048,055/1,048,576 (100.0%)** |
| `embed_model.mem_v_proj` | 242/1,048,576 (0.02%) | **1,048,015/1,048,576 (99.9%)** |

Frozen backbone leaves (`main_model.norm`, layer 9/17/26 norms, `q_proj`/`up_proj`) correctly stay
`dtype=bfloat16` with `n_changed=0` — confirms the fix is precisely scoped to the leaves that are
actually supposed to be promoted, not an accidental blanket fp32 upcast.

**Head-to-head, first ~800 steps of each arm** (matched row counts, thirds):

| | arm 0 (bf16, bugged) | arm 1 (fp32, fixed) |
|---|---|---|
| `doc_access_per_query_loss` thirds | 2.55 → 2.15 → 2.04 | 2.41 → 2.29 → 2.28 |
| `mem_pos_weight_mass` thirds | 0.41 → 0.50 → 0.50 | 0.42 → 0.49 → 0.48 |

**Honest read: over this short a window, the two arms look similar** — both show a modest early
improvement, arm1 not obviously faster or better yet. That's not the dramatic before/after I was
expecting given the wchanged evidence (99.9% vs 0.02% of elements moving), and it means the
restore-dtype bug, while real, confirmed, and worth fixing regardless (it was silently discarding
a deliberate, previously-validated fix — that's wrong on principle even if its practical impact
here turns out to be modest), **may not be the dominant reason the loss "isn't decreasing"** —
or its benefit only shows up over a longer horizon than 800 steps. Arm 0's own full-run deciles
above suggest real improvement needs ~1000+ steps to separate from noise at this variance level
(std ~0.9-1.0 per point), so the fair test is arm1 at a comparable step count to arm0's full
~2900, not 800. Continuing to let it run and will re-compare at parity.

**But the dispositive weight-accumulation test (per the project's own 2026-08-07 methodology —
`weight/wnorm/*/rms`, not `wchanged`/gradient-response metrics) confirms the fix DOES do what it's
supposed to, even though the loss-level effect isn't dramatic yet:**

| | arm 0 (bf16, thirds) | arm 1 (fp32, thirds) |
|---|---|---|
| `wnorm/main_model.layers.14.mem_q_proj/rms` | 0.021636 → 0.021650 → 0.021666 (flat to 4 decimals — matches the FROZEN/MIXED verdict from 2026-08-07 exactly) | 0.021619 → 0.021579 → 0.021538 (**monotonically decreasing, real drift**) |
| `wchanged/main_model.layers.14.mem_q_proj` (frac/step) | 0.36 → 0.39 → 0.28 | **~1.0 → 1.0 → 1.0** (essentially every element every step) |
| `doc_access_acc` | 0.423 → 0.455 → 0.494 | 0.352 → 0.424 → 0.427 |
| `mem_hit_rate/mean` | 0.820 → 0.867 → 0.855 | 0.833 → 0.785 → 0.831 (noisier, dips mid-window) |
| `grad_norm_mem` | 1.02 → 0.51 → 0.39 | 1.40 → 0.67 → 0.65 |

So arm 1's weights are unambiguously accumulating real change (RMS moving, not just bit-noise);
arm 0's are not (RMS frozen, exactly the "MIXED — moves but doesn't accumulate" signature).
`grad_norm_mem` shrinking similarly in both arms suggests a shared underlying dynamic (very
plausibly `mem_o_proj`'s zero-init escape hatch, which moves under bf16 too) is doing a
meaningful share of the early improvement in BOTH arms, which is consistent with why the
loss-level trend hasn't diverged yet — a genuinely-moving `mem_q_proj`/`mem_k_proj` needs more than
~800 steps at this RMS drift rate to compound into a visibly different retrieval outcome. Next
check compares at step-count parity with arm 0's full ~2900-step run to see if that's actually
where arm 1 pulls ahead.

**Update at near-parity (~2230 steps each, 223 logged rows):** arm 1 has pulled slightly, but not
dramatically, ahead —

| | arm 0 (bf16, thirds) | arm 1 (fp32, thirds) |
|---|---|---|
| `doc_access_per_query_loss` | 2.26 → 1.95 → **1.81** | 2.35 → 1.76 → **1.71** |
| `mem_pos_weight_mass` | 0.470 → 0.535 → 0.550 | 0.467 → 0.549 → 0.553 |

Last-third gap is ~0.10 (1.81 vs 1.71, ~5.5% relative) — roughly one standard error at this
sample size (std ~0.9-1.0 per point, ~74 points/third → SE ≈ 0.11), so directionally consistent
with the fix helping but not yet a clean, unambiguous separation. The trend is moving the right
way as more steps accumulate (statistically indistinguishable at 800 steps → a modest, consistent
edge at 2230), consistent with the RMS-drift-needs-time-to-compound theory above.

**Update at FULL parity with arm 0's complete run (2930 steps, matched row counts):**

| | arm 0 (bf16, full run, thirds) | arm 1 (fp32, same step range, thirds) |
|---|---|---|
| `doc_access_per_query_loss` | 2.19 → 1.91 → 1.71 | 2.19 → 1.79 → **1.65** |
| `mem_pos_weight_mass` | 0.483 → 0.536 → 0.572 | 0.485 → 0.537 → **0.580** |

Arm 1 is now ahead on **both** metrics in the later thirds, and it's a bit more consistent about
it than the single-point comparison earlier suggested. Arm 1's own full trajectory to its current
3640 steps, by decile: `2.45 → 2.27 → 1.66 → 1.83 → 1.86 → 1.70 → 1.57 → 1.61 → 1.69 → 1.56`
(latest 20-row mean: **1.58**) — noisy but genuinely trending down and roughly stabilizing
1.5-1.7 in deciles 6-9, without the late-run reversal arm 0 showed (its decile 9 jumped back to
1.97 after a decile-8 low of 1.48 — arm 0's trajectory looks less stable, not just slower).
Meanwhile `weight/wnorm/main_model.layers.14.mem_q_proj/rms` in arm 1 is STILL smoothly,
monotonically decreasing every single decile (0.021613 → 0.021377) — no sign of the fix's effect
plateauing.

**Read: the fix produces a real, modest, and now fairly consistent improvement — not a dramatic
transformation.** Both arms learn something (loss ~2.4-2.5 → ~1.5-1.9 over ~3000 steps either
way), which reinforces that `mem_o_proj`'s zero-init escape hatch was already doing real work
under the bug. The fp32 fix adds a genuine but incremental edge on top of that, and arm 1's
trajectory looks somewhat more stable in addition to being somewhat lower. Worth watching what
happens at the stage 1→2 transition (step 5000): `ce_weight` 0.1→1.0, `doc_access_per_query_loss`
weight 1.0→0.1 — a substantial change in gradient allocation that could either continue the trend
or stall it out as CE starts competing harder for the same gradient budget.

**Stage 1→2 transition (step 5000) happened cleanly.** Confirmed via the console `(w=...)` value
(this is `ce_weight`, verified: 0.10000000149011612 throughout stage 1, exactly `1.0` from step
5001) and a fresh weight-monitor snapshot at the boundary (frozen backbone leaves correctly still
`dtype=bfloat16`, unaffected). The transition itself cost ~5 min of recompile (new optimizer/mask
→ new jitted train step), same order as the original first-step compile — not a hang, just
expected JIT cost. No crash, no traceback, no restart.

**Stage 2 update (~1400 steps in, 141 logged rows):** the improving trend from stage 1 does
**not** carry through — `doc_access_per_query_loss` thirds `1.52 → 1.56 → 1.59` (flat to
slightly worse), `mem_pos_weight_mass` `0.580 → 0.545 → 0.555` (flat/slightly down),
`doc_access_acc` dead flat at `~0.455`. Last 10 rows average worse still (`dapq=1.79`,
`mpwm=0.506`) though that's a small, noisy sample (std ~0.9-1.0/point over just 100 steps) —
not yet claiming a reversal, just that the stage-1 improvement has clearly **stopped**, not
continued. Meanwhile `weight/wnorm/main_model.layers.14.mem_q_proj/rms` is STILL moving
(0.021328 → 0.021204 → 0.021012) — the fix keeps doing its job mechanically, the weights keep
changing, but that movement isn't translating into further retrieval improvement once CE gets
10x the gradient weight back.

**Reading:** this suggests the recipe design itself — 5000 steps at high `doc_access_per_query_loss`
weight, then a hard drop to the normal 1.0/0.1 split — may be losing the momentum stage 1 built,
not because the dtype fix stopped working (RMS drift proves it hasn't) but because CE dominating
the gradient budget again gives the model much less incentive to keep sharpening retrieval
specifically. Two candidate follow-ups if this holds up over more steps: (a) a longer stage 1
(the improvement in stage 1 hadn't clearly plateaued by step 5000 — deciles 6-9 were still in the
1.5-1.7 range, not obviously flattening), or (b) a gentler stage-2 weight (e.g. 0.3-0.5 instead of
0.1) so CE doesn't fully crowd out the retrieval signal. Watching a few hundred more stage-2 steps
before deciding whether to launch either as a new arm — the last-10-rows uptick could just be
noise at this sample size.

**Correction with more data (~2140 stage-2 steps, 214 rows): the "stall" was noise, not a real
plateau.** `doc_access_per_query_loss` thirds now read `1.516 → 1.572 → 1.468` — the last third
is BELOW the first, not above it; the middle-third dip that looked like a stall at 141 rows was
exactly that, a dip, not a trend. `mem_pos_weight_mass` `0.577 → 0.549 → 0.576` shows the same
dip-and-recover shape, ending back near its stage-1 level. Last 10 rows: `dapq=1.36` (the best
value seen yet in this run), `mpwm=0.559`. `mem_q_proj` RMS keeps decreasing smoothly
(`0.021302 → 0.021058 → 0.020770`). **Lesson for the rest of the night: at this noise level
(std ~0.9-1.0/point), don't trust a trend read off fewer than ~200 samples** — the previous
"stalled" conclusion was reached at 141 stage-2 rows and didn't survive 73 more. Not launching a
new experiment arm; the current run is doing fine, just noisily.

**Confirmed with a large sample (353 stage-2 rows, ~3530 stage-2 steps, total step ~8523):** the
trend is now unambiguously, monotonically positive —

| | thirds |
|---|---|
| `doc_access_per_query_loss` | 1.539 → 1.472 → **1.466** |
| `mem_pos_weight_mass` | 0.562 → 0.577 → **0.583** |
| `doc_access_acc` | 0.458 → 0.465 → 0.461 |
| `mem_q_proj` rms | 0.021226 → 0.020782 → **0.020445** |

Last 10 rows: `dapq=1.11`, `mpwm=0.646` — both new bests for the run. This is the point the
Conclusion at the top of this doc is based on. No new experiment arm needed — this recipe,
with the dtype fix, works.

**Health-check log (routine monitoring, no further re-analysis unless something changes):**
- 09:20 UTC — step 9928, `Loss=1.59`/`CE=1.39` (w=1.0, stage 2), box READY, same tmux session
  since launch, no tracebacks. Healthy.
- 10:25 UTC — step ~10870 (587 stage-2 rows). **Update: the loss appears to have plateaued, not
  continued falling.** `doc_access_per_query_loss` thirds `1.529 → 1.449 → 1.483` and
  `mem_pos_weight_mass` `0.566 → 0.583 → 0.567` are both oscillating in a band (~1.45-1.53,
  ~0.57-0.58) rather than trending further down/up — and `mem_q_proj` rms broke its smooth
  monotonic decrease this window (`0.02108 → 0.02048 → 0.02091`, ticked back up). Given the
  established noise floor (std ~0.9-1.0/point) this could still be within-band noise rather than
  a genuine plateau, but it's a big enough window (587 rows) that it's worth flagging honestly
  rather than assuming the earlier improving trend just continues indefinitely. Net overnight
  result stands regardless: `doc_access_per_query_loss` ~2.4 → **~1.4-1.5** (~40% down),
  `mem_pos_weight_mass` ~0.42 → **~0.57-0.58**. Whether there's a further formulation change
  (longer stage 1, different stage-2 weight/LR) that pushes past this band, or this is close to
  a natural floor for a 200-hard-negative-per-query task, is an open follow-up — not chased
  further tonight since the box is a shared single resource and the core bug-fix question this
  investigation was about is answered.

**Decision: let it keep running rather than switch arms.** Given (a) the only real bottleneck
(compute — one box) argues against interrupting a run that's showing a positive trend to try
something else on a hunch, and (b) stage 1 (`ce_weight=0.1`, `doc_access_per_query_loss` weight
`1.0`) ends at step 5000 and the stage-1→stage-2 transition itself is worth observing end-to-end
for this recipe — continuing to let arm 1 run through the rest of the night, checking back
periodically, rather than killing it to test a different formulation. Will revisit the "design a
new arm" branch if the lead stops growing or reverses.

**Arm 1 stopped (not crashed) at ~11:57 UTC** — rohunagrawal asked to switch the warm-start
source to a further-along, different-lineage checkpoint (see Arm 2 below) once awake. Clean
shutdown: `tmux kill-session` + verified no orphaned `venv/bin/python` + `fuser -v /dev/vfio/*`
empty before freeing the box.

## Arm 2 — pf32/indexed/lr_masked source checkpoint, main_model unfrozen

New source, new architecture, same underlying goal (good multihop hard-neg retrieval). Source:
`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked` step 100000 — a
**different lineage** from arms 0/1: single memory layer (`mem_layers=[14]`, not `[9,14,20,27]`),
`mem_top_k=64` (not 128), and — critically — `main_model` **already unfrozen** through
`staged.yaml`'s own 4-stage recipe (stage D: full unfreeze, cosine LR, per-group
`learning_rates` from the just-merged `origin/lr` branch: mem=1e-4, embed=1e-5, main=1e-5).

**Decision (asked, not assumed):** the checkpoint's architecture doesn't match arms 0/1's
4-layer setup, and mismatches silently fall back to fresh-init leaves rather than erroring — a
previously-documented footgun in this codebase. Asked rohunagrawal directly rather than
guessing: (1) match the checkpoint's single-layer architecture exactly (chosen, over keeping 4
layers with 3 fresh-init) — nothing silently fresh-inits; (2) keep `main_model` trainable
(chosen, over re-freezing a model that's already past that stage) — continues the checkpoint's
own trajectory instead of reverting it.

**New files:** `configs/trainer/staged_batched_isolation_docaccess_warmstart.yaml` (built on
`staged`, not `staged_ground_batched_isolation` — main_model trainable via `learning_rates.main`
instead of frozen; same 2-stage doc-access-warmup shape as arm 0/1's trainer config),
`scripts/embed/train_multihop_lrmasked_warmstart.sh`. Kept `per_query_isolation` +
`mem_batched_isolation` + `doc_access_per_query_loss` (not `doc_access_loss`) — required for
multihop_hard_neg_full's 200-hard-neg scale regardless of what the source checkpoint used, since
those only change the runtime retrieval mode, not any weight shape.

**Seed:** `dataset.shuffle_seed=20260813` (repo default is 42) — specifically so this run's data
order doesn't replay whatever a same-seed run already walked through (the "weird epoch
artifacts" rohunagrawal flagged from two runs sharing a seed against the same deterministic
interleave/pool).

**Four launch attempts before a clean start — each failure diagnosed and fixed, not retried
blindly:**

1. **`MissingConfigException: eval_set/standard`** — `staged.yaml` (unlike `staged_ground.yaml`)
   doesn't have the `override /eval_set@evals: none` fix for this branch's missing
   `eval_set/standard`. Fixed in the new trainer config's own `defaults`, not in shared
   `staged.yaml`.
2. **`RESOURCE_EXHAUSTED` compiling `jit__train_step`**: needs 23.06G, 20.35G free (short
   ~2.7G) — unfrozen `main_model` + multihop's large activation footprint (256 chunks/doc, bs=8)
   doesn't fit replicated on one v6e chip.
3. **Tried `tp_devices=2`** (the standard fix per `wiki/training/optimizer.md`) — hit a hard
   structural wall instead: Qwen3's vocab dim (151669 = 7×21667, no factor of 2) can't be
   sharded 2 ways. `jax.device_put` raised a divisibility `ValueError` resharding
   `embed_tokens` the moment checkpoint restore ran. Not a workaround-able OOM; reverted.
4. **Excluded `embed_tokens`/`lm_head`** (both models) from `trainable_params` via negative
   lookahead (`main_model\.(?!embed_tokens|lm_head)`, `embed_model\.(?!embed_tokens)`) — bf16,
   frozen, no optimizer moments for those two vocab-sized tables. Closed ~0.8G of the gap (down
   to "need 22.75G, 20.86G free", short ~1.89G) — not enough alone. `jax.remat` is already
   applied to every transformer layer (`models/qwen3.py`) — gradient checkpointing wasn't an
   available additional lever, it's already on.
5. **Asked rohunagrawal** how to close the remaining ~1.89G: cut hard-neg coverage
   (`num_chunks_per_doc` 256→224, ~204→~180 docs/query) or revert `main_model` to frozen
   (keeping full 200-neg coverage). Chose the coverage cut to preserve the "keep main_model
   trainable" decision. This closed the gap — **step 0→1 completed successfully.**
6. **Two of the crashed attempts left the TPU held by an orphaned process** even after their
   tmux session was killed (`tmux_launch.sh`'s session-kill doesn't reach orphaned grain/JAX
   workers) — `ABORTED: The TPU is already in use by process with pid ...` on the next launch
   attempt both times. Fixed with `pkill -9 -u $(id -u) -f venv/bin/python` +
   `fuser -v /dev/vfio/*` empty-check before relaunching, per the launch runbook.

**Confirmed healthy at launch:** step 0→1 completed (no OOM, no crash). Weight-monitor confirms
both things at once: (a) the dtype fix from arms 0/1 still works on this new lineage — `mem_q_proj`
/ `mem_o_proj` / `embed_model.mem_{k,v}_proj` are `dtype=float32` with ~99.9-100% of elements
changing; (b) `main_model` really is unfrozen now — layers 9/17/26 (which stayed bit-exact
FROZEN all through arms 0/1) now show `dtype=float32` with nonzero `n_changed`, confirming they're
genuinely training, not silently falling back to fresh-init. First loss:
`Loss: 0.8731 | CE: 0.9744 (w=0.1)`.

No trend data yet — just launched. Next check should compare against arm 1's own trajectory at
matched step counts, same as the arm-0-vs-arm-1 comparisons above.

**Arm 2 stopped (not crashed) at step 4401** (`Loss: 0.0961 | CE: 0.4868`) — rohunagrawal
reclaimed `tn-v6e-8-0` for the CoT-injection ablation (see below): it doesn't use CoT, so it's
lower priority than getting that ablation running. Clean shutdown verified (no tmux session, no
orphaned `venv/bin/python`, `fuser -v /dev/vfio/*` empty) before reuse. Checkpoints up to step
4400 remain on GCS if this arm is resumed later.

## Ablation — inject CoT as a positive doc, test whether doc_access_per_query_loss can collapse

**Status: CONFIRMED.** `doc_access_per_query_loss` collapses from ~0.069 to ~0.00001-0.00005
within a few hundred steps once the CoT (which trivially reveals the answer) is inserted as a
retrievable positive document, and stays collapsed through step 3730 — see the confirmed result
below. The plateau in arms 0-2 (~1.1-1.5) is real task difficulty against genuinely hard
negatives, not a structural bug in the loss/pipeline. One open nuance (`doc_access_acc` staying
~0.53-0.56 despite the near-zero loss) is flagged below, not yet explained.

**Hypothesis:** arms 0-2 all show `doc_access_per_query_loss` improving but plateauing well
above zero (~1.1-1.5). Is that a natural floor for a genuinely-hard 180-204-negative
discrimination task, or is something structurally capping it regardless of task difficulty? Test:
insert each example's CoT (chain-of-thought) text as an *additional positive document* in the
memory bank. If the model can trivially retrieve it (the answer is right there in the text) and
`doc_access_per_query_loss` collapses toward zero, that's evidence the pipeline/objective is
sound and the plateau in arms 0-2 reflects real task difficulty, not a bug. If it *doesn't*
collapse even with the answer trivially present, that points to something structural.

**Data problem found before writing any code:** the `multihop_hard_neg_full` dataset's staged
parquet (`mihir-1999/multihop_qa_sft-hard-neg-train`, built by
`datagen/download_multihop_hardneg.py`) carries only `question | answer | pos_doc_ids |
neg_doc_ids` — no CoT column at all, confirmed directly against the HF schema (checked the
actual `rows/row_pos_doc_ids.parquet` schema: `row_id, question, answer, pos_doc_ids`, no
`think`). The *original* `ragrawal36/multihop_qa_sft` source has a `think` field, but there's no
shared id between the two datasets — question-text is the only available join key.

**Join script:** `datagen/join_multihop_cot.py` (+ box runner
`scripts/misc/run_join_multihop_cot.sh`) downloads all 176 train shards of
`ragrawal36/multihop_qa_sft` (~5.44 GB; row count ~1,352,000 vs. mihir-1999's 1,341,045 — close
enough to be encouraging, not proof), builds a `question -> think` map (first-occurrence wins on
duplicate questions, with duplicate/conflict counts reported), joins onto the locally-staged
mihir-1999 rows, reports the match rate, writes the joined parquet, and (unless `--dry-run`)
publishes it as a new private HF dataset: `ragrawal36/multihop_qa_sft-hard-neg-cot`.

**Compute churn before this got running** — `us-east1-d` (where `rohun-v6e-8-0/1` live) hit two
independent zone-wide spot-preemption waves in short succession (confirmed by *other people's*
`john-v6e-8-*` boxes going down in the same instant both times — not anything specific to this
session). Tried `rohun-v6e-8-0` (preempted mid-run), then `rohun-v6e-8-1` (preempted before the
join even started). rohunagrawal then chose to stop Arm 2 (above) and free `tn-v6e-8-0` for the
whole CoT pipeline instead of continuing to fight that zone's spot pool — the join is pure
CPU/network work, no TPU chip needed, so `tn-v6e-8-0` running it doesn't conflict with anything.

**Dry-run join result: 100.00% match (1,341,045/1,341,045 rows)** — every mihir-1999 row found a
`think` value by exact question-text match, confirming the two datasets share the same
underlying question set (not just similar row counts). Caveat: 70,029 questions (~5.5% of
1,271,016 unique) appeared as duplicates within `ragrawal36/multihop_qa_sft` itself, and 69,518
of those had *conflicting* `think` text (same question string, different sampled reasoning
trace) — kept first occurrence. Doesn't threaten the ablation's validity (any real CoT for that
question is still directly answer-revealing for retrieval purposes), just means the attached CoT
isn't guaranteed to be *the* canonical one for ~5% of rows. Proceeding to publish the full join.

**Published**: [`ragrawal36/multihop_qa_sft-hard-neg-cot`](https://huggingface.co/datasets/ragrawal36/multihop_qa_sft-hard-neg-cot)
(private, 1.85 GB, all 1,341,045 rows: `question | answer | pos_doc_ids | neg_doc_ids | think`).

**Wiring** (kept minimal per the original ask, two independent knobs added to the data pipeline):
- `data/utils.py::_docid_normalize` gained a `cot_field` param — if set and present on the row,
  forwards the raw text under a new `cot_doc` key. Deliberately independent of the pre-existing
  `think_field` param (which prepends CoT into the *answer*, i.e. the CE target) so the two don't
  interact.
- `data/qa.py::qa_transform_item` appends `item["cot_doc"]` (if present) to `pos_docs_raw` before
  the `pack_docs` call — `pack_docs` marks every entry in that list `pos_mask=1` uniformly, so no
  separate `pos_doc_mask` edit was needed.
- New source `configs/dataset/sources/multihop_qa_sft_hard_neg_cot.yaml` (points at the new HF
  repo, sets `cot_field`) and dataset `configs/dataset/multihop_hard_neg_full_cot_ablation.yaml`
  (literal copy of `multihop_hard_neg_full.yaml` with the source swapped).
- Verified with a standalone CPU-only smoke test (`scripts/debug/smoke_test_cot_ablation.py`,
  no TPU) against 5 real rows before spending any TPU time: `cot_doc` present and byte-identical
  to the raw `think` field, and `pos_doc_mask`'s positive-chunk count strictly higher than the
  same row processed without the CoT append.
- **rohunagrawal's follow-up decision**: also set `think_field: "think"` on the new source, so
  CoT feeds CE too (prepended into the answer as `<think>...</think>`), not retrieval-only. This
  means the ablation is no longer single-variable against the base `multihop_hard_neg_full` arm
  — both the retrieval objective and the CE target now differ. Re-verified with the same smoke
  test (extended to also assert the answer is `<think>`-prefixed with the matching raw text)
  before relaunching.

**Launch**: `scripts/embed/train_multihop_lrmasked_warmstart_cot_ablation.sh` — same warm-start
checkpoint, architecture overrides, trainer, and `shuffle_seed` as
`train_multihop_lrmasked_warmstart.sh` (Arm 2), dataset swapped to the ablation config. Ran on
`tn-v6e-8-0` after Arm 2 was stopped to free the box (see above) — two launches: first without
`think_field` (killed cleanly at step 126 once rohunagrawal asked for `think_field` too), then
the final version with both knobs set. Both launches passed step 0→1 with no OOM/crash (same
`num_chunks_per_doc=224` HBM fix as Arm 2 applies here too — identical architecture).

- **wandb (CoT ablation, final):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-15-44-42`

**Third arm added — CE-only, isolating the two effects.** rohunagrawal asked for a second
parallel run, "Arm 2 but with `think_field` on" — I initially (mistakenly) relaunched the full
CoT-ablation script, which also has `cot_field` set (memory-bank injection). Corrected: a new
source (`configs/dataset/sources/multihop_qa_sft_hard_neg_cot_ce_only.yaml`) sets **only**
`think_field`, leaving `cot_field` unset — same joined dataset, but `qa_transform_item` never
sees a `cot_doc` key, so the memory bank is untouched and only the answer/CE target changes.
Verified both configs with the (now dataset-parameterized) smoke test before launching:
`multihop_hard_neg_full_cot_ablation` → `cot_doc` present + `<think>`-prefixed answer;
`multihop_hard_neg_full_cot_ce_only` → no `cot_doc` on any row + `<think>`-prefixed answer.
Launched on `rohun-v6e-8-1` (staged fresh — the box had been preempted and wiped since Arm 2's
staging) — logged a clean step 0→1 with no traceback, but rohunagrawal reported it didn't
actually look like it was running (unconfirmed why — didn't dig into it further at their
direction). Relaunched instead on `rohun-v6e-8-0` (also re-staged from scratch — the whole
`us-east1-d` spot pool had cycled again): confirmed step 0→1 completed with byte-identical
`Loss`/weight-monitor numbers to the `rohun-v6e-8-1` attempt (expected — same checkpoint, seed,
code, so deterministic replay), so this box's copy is the one actually running.

This gives a clean 3-way comparison, all from the same warm-start checkpoint / architecture /
`shuffle_seed`:

| Arm | Memory bank | CE target | Box | wandb |
|---|---|---|---|---|
| Arm 2 (base) | no CoT | no CoT | `tn-v6e-8-0` (stopped, ckpt @ step 4400) | `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-12-20-07` |
| CoT ablation | CoT extra positive doc | `<think>`-prefixed | `tn-v6e-8-0` (stopped, see below — ran to step 3730) | `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-15-44-42` |
| CE-only | unchanged | `<think>`-prefixed | **running** — `rohun-v6e-8-0`, past step 6600, sustained recovery confirmed (see point 12) | steps 0-600: `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-16-45-43`; steps 600-2300: `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-18-27-09`; steps 2200-2400 (telemetry): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-22-21-38`; steps 2400-4000 (telemetry, rohun-v6e-8-0): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-00-28-34`; steps 4000-6200: `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-14-24-00`; steps 6200-6400 (6600 dir on GCS incomplete, see point 10): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-17-01-00`; step 6400 (brief): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-01-00-55`; step 6400 (brief, maintenance-event): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-02-03-08`; steps 6400+ (sustained, rohun-v6e-8-0): `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-03-08-02` |

**Confirmed against the actual isolated `doc_access_per_query_loss` metric (not just the
combined console `Loss`)** — pulled the full wandb history after the arm was stopped:
`doc_access_per_query_loss` collapses from ~0.069 (earliest steps) to **~0.00001–0.00005** by
the first few hundred steps and stays there through step 3730 (min observed: ~0.000003; a single
outlier spike to 0.86 aside, the trajectory is a clean, sustained collapse, not noise settling
near a small-but-nonzero floor like arms 0-2). **This confirms the hypothesis**: when the answer
is trivially present as a retrievable positive document, the retrieval objective collapses
toward zero almost immediately — so the plateau at ~1.1-1.5 in arms 0-2 reflects genuine
discrimination difficulty against 180-204 hard negatives, not a structural ceiling in the
loss/pipeline.

**One open nuance, not yet explained**: `train/doc_access_acc` sits around **0.53-0.56**
throughout the same window — nowhere near the ~1.0 you'd expect alongside a near-zero
contrastive loss. Possible explanations not yet checked: `doc_access_acc` may be computed
per-token-position across ALL positive-marked chunks (including the original real positive docs,
which the near-zero loss doesn't guarantee are individually top-ranked — the softmax mass could
concentrate almost entirely on the CoT doc specifically while still leaving the original
positives under-retrieved), or there may be a temperature/threshold mismatch between how the loss
and the accuracy metric each define "retrieved." Worth resolving before treating the collapse as
proof the *original* documents became easier to retrieve, rather than just the CoT text itself.
(`train/mem_pos_weight_mass/mean` wasn't present in this run's logged keys — couldn't cross-check
that corroborating metric.)

**Box consolidation, `us-east1-d` preempted again**: mid-session, `rohun-v6e-8-0` (running
CE-only) got swept by another zone-wide spot preemption (confirmed: `john-v6e-8-5` went
`PREEMPTED` in the same instant). Rather than keep fighting that pool, rohunagrawal had me stop
the CoT-ablation arm on the stable `tn-v6e-8-0` and continue the CE-only arm there instead —
**as a full resume of the preempted run**, not a fresh warm-start (checkpoints existed at steps
200/400/600 in `rohun-v6e-8-0`'s run-dir).

Two mistakes on the way to the correct resume, both caught before wasting real compute:
1. First relaunch attempt set `RESUME_FROM` as a **local shell prefix** to the `multi-vm-tpu-run.sh`
   invocation — ssh doesn't forward the local environment, so the box-side script silently fell
   back to its default (the *original* lr_masked warm-start checkpoint, not the resume target).
   Caught by checking the printed `resume_from:` in the composed config before trusting it —
   killed at step 1.
2. Second attempt used `RUN_ENV="RESUME_FROM=..."` correctly (config showed the right value), but
   the path was missing the `/qwen3_mem_embed` subdirectory — `load_manager.latest_step()` found
   no checkpoints under the bare run-dir, so `ckpt_step` stayed `None` and the entire restore
   block was silently skipped (train.py has no "checkpoint not found" error path for this — it
   just proceeds as a fresh run). **Caught via the dispositive tell**: `mem_q_proj` and
   `mem_o_proj` showed **byte-identical** `rms`/`mean|w|` (0.01997/0.01596 both) — real trained
   parameters never coincide like that; only a fresh, same-scale random init would. Killed at
   step 1 again (before `checkpoint_interval=200`, so nothing was written to collide with the
   good checkpoints).
3. Third attempt (`RESUME_FROM=.../qwen3_mem_embed`, no trailing step) worked: weight-monitor
   labels read `[wmon step=600]` (not `step=0`), `mem_q_proj`/`mem_o_proj` show distinct trained
   values (0.01915/0.01712), and previously-frozen backbone norms show accumulated change from
   real training. The tqdm progress-bar counter cosmetically resets to `1/100000` on a full
   resume (it's not step-aware for its `n` display) — harmless, the `step=600` weight-monitor
   label and the actual training step are correct; don't mistake the tqdm counter for the real
   step when checking a resumed run.

A full resume mints a **new wandb run** (new run-dir → new derived id — documented behavior, see
`wiki/infrastructure/experiment-launch-instructions.md` §5), so this arm's history is split
across two run ids at the step-600 boundary (both linked above).

No trend data yet on the CE-only arm post-resume. Once enough steps accumulate on `tn-v6e-8-0`,
pull the CoT-ablation arm's full `doc_access_per_query_loss`/`mem_pos_weight_mass` history first
(it already ran to step 3730 with a very promising `Loss` reading) to confirm or refute the
collapse hypothesis, then compare CE-only against Arm 2 at matched step counts to see whether the
CE-side change alone moves retrieval metrics.

**All three arms stopped, then CE-only relaunched with telemetry enabled.** After the CE-only
arm reached step 2300 (ckpt @ 2200), rohunagrawal stopped it (and had already stopped Arm 2 @ 4400
and the CoT ablation @ 3730 — see above) and asked for a resume with "telemetry on".

This closes the gap noted above (line 536): `train/mem_pos_weight_mass` (and the rest of the
weight-0 read-channel telemetry block — `mem_write_norm`, `mem_hit_rate`, `mem_topk_entropy`,
etc., see `wiki/training/auxiliary-losses.md`) was **never being logged** for any of the three
arms. `staged_batched_isolation_docaccess_warmstart.yaml` is built on plain `staged` (not
`staged_ground`/`staged_telemetry`, the only two configs that carry the telemetry block), so it
inherited none of it — this, not a metric-computation bug, is why the CoT-ablation cross-check
came up empty.

Fix: added the same weight-0 `aux_losses` block used by `staged_telemetry.yaml`/`staged_ground.yaml`
to `staged_batched_isolation_docaccess_warmstart.yaml` as a top-level key (sibling to
`training_stages`). Confirmed via `trainer/trainer.py` (`base_aux_loss_dict` +
`_apply_stage_loss_overrides`) that this top-level dict merges with each stage's
`doc_access_loss`/`doc_access_per_query_loss` override rather than being replaced by it, so no
other behavior changes.

Relaunched the CE-only arm as a **full resume** from the step-2200 checkpoint
(`RESUME_FROM=.../multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-13-18-27-09/qwen3_mem_embed`)
onto `tn-v6e-8-0`. Confirmed genuine resume (not fresh init): weight-monitor label read
`[wmon step=2200]` matching the checkpoint, and `mem_q_proj`/`mem_o_proj` showed distinct trained
rms values (0.01928 / 0.01714) rather than the byte-identical fresh-init tell. Confirmed telemetry
is now live by pulling the new run's wandb history directly on the box (`uv run python3` +
`wandb.Api()`): all 13 `train/mem_*` keys present and non-null, e.g.
`mem_pos_weight_mass/mean=0.846`, `mem_hit_rate/mean=1.0`, at `doc_access_per_query_loss=0.248`,
`doc_access_acc=0.578` — consistent with the pre-stop trend. New wandb run id (full resume always
mints one, see above): `multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-22-21-38`.

**`tn-v6e-8-0` lost entirely, moved to `rohun-v6e-8-0`.** The telemetry-enabled run crashed;
`gcloud alpha compute tpus tpu-vm describe tn-v6e-8-0 --zone=europe-west4-a` came back
`NOT_FOUND` — the node itself was gone (spot reclamation deleting the resource, not just
preempting it in place), not a recoverable in-place preemption. `rohun-v6e-8-0`/`rohun-v6e-8-1`
(both `us-east1-d`, `memorylayers` project) were READY, so resumed there instead.

`rohun-v6e-8-0` was a brand-new box, so the full one-time setup was needed before the launch
script's own `HF_HUB_OFFLINE=1` could work: `scripts/misc/stage_qwen3_4b_weights.sh` (main +
embed model weights) and `scripts/misc/stage_multihop_hardneg_data.sh` (doc corpus + hard-neg
rows). Hit and fixed one real bug in the process: `scripts/misc/stage_multihop_hardneg_cot_data.sh`
never sourced `.env`, unlike its two sibling staging scripts, so `HF_TOKEN` was never set and the
private `ragrawal36/multihop_qa_sft-hard-neg-cot` download 401'd. Fixed to match the sibling
scripts' `set -a; . .env ...` pattern.

Relaunched as a full resume from the step-2400 checkpoint
(`RESUME_FROM=.../multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-13-22-21-38/qwen3_mem_embed`)
onto `rohun-v6e-8-0`. Confirmed genuine resume the same way as before: `[wmon step=2400]` matching
the checkpoint, `mem_q_proj`/`mem_o_proj` distinct trained rms (0.0193 / 0.01713), first step
compiled (~7 min, consistent with prior launches) with no tracebacks, checkpoint re-saved at step
2400. New wandb run: `multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-00-28-34`.

rohunagrawal asked for an autonomous `/loop` to keep this arm alive going forward: periodically
check the run is healthy, and on a crash/preemption wait a few minutes for the TPU to come back
before resuming (full resume from the latest GCS checkpoint each time), alternating between
`rohun-v6e-8-0` and `rohun-v6e-8-1` if a box doesn't come back.

**`rohun-v6e-8-0`'s disk got wiped mid-run, discovered via a step-count check.** Some time after
the step-2400 resume above, `rohun-v6e-8-0` was recreated (new `createTime`, empty home dir, no
`~/memory-layers`, no `~/runs` log) — not a preemption-in-place, the whole VM disk was gone, same
underlying failure mode as the earlier `tn-v6e-8-0` loss but this time the *node name* survived
(`describe` still reported `READY`) while its local state didn't. The local log was gone, so
current progress was checked against GCS instead (authoritative regardless of what happened to
the VM): `gsutil ls .../multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-00-28-34/qwen3_mem_embed/`
showed checkpoints through **step 4000** — training had kept going well past the last step this
doc had recorded (2400) before the wipe.

Recovery: verified no orphaned TPU-holder (box was freshly booted, so trivially clean), re-ran the
full one-time staging (`stage_qwen3_4b_weights.sh`, `stage_multihop_hardneg_data.sh`,
`stage_multihop_hardneg_cot_data.sh`), then relaunched a full resume from step 4000
(`RESUME_FROM=.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-00-28-34/qwen3_mem_embed`).
One false start: the first launch attempt fired *before* staging had finished (same missing-weights
crash as the original `tn-v6e-8-0` loss, since `HF_HUB_OFFLINE=1` doesn't fetch), so staging must
complete before the launch command, not just be kicked off in parallel. Second attempt (staging
finished first) came up clean: `[wmon step=4000]` matching the checkpoint, `mem_q_proj`/`mem_o_proj`
distinct trained rms (0.01937 / 0.01704), first `Loss:` print at ~8.5 min compile, no tracebacks.
New wandb run: `multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-14-24-00`.

**Lesson for the standing `/loop`**: a box reporting `READY` is not sufficient evidence the run
survived — this box never went `NOT_FOUND`/`PREEMPTED`, it just silently lost its disk and came
back `READY` with nothing on it. Always check for the actual log/repo before trusting a "healthy"
box, and treat GCS (not the local log) as the source of truth for how far training actually got.

**Health check, step 4400**: still on `rohun-v6e-8-0`, no interruption since the step-4000
relaunch — `[wmon step=4400]`, `Loss` fluctuating 0.05-0.46 (normal per-batch variance),
`doc_access_per_query_loss=0.171`, `doc_access_acc=0.527`, `mem_pos_weight_mass/mean=0.893`,
`mem_hit_rate/mean=1.0`. Consistent with the established trend, nothing new to report.

**Health check, step ~4900**: still on `rohun-v6e-8-0`, one continuous run since the step-4000
relaunch (checkpoints saved sequentially 4000→4200→4400→4600→4800, single wandb tracking
session, no restarts) — `doc_access_per_query_loss=0.294`, `doc_access_acc=0.665`,
`mem_pos_weight_mass/mean=0.866`, `mem_hit_rate/mean=1.0`. `doc_access_acc` climbing (0.527→0.665)
is the first real movement seen in that metric across any arm — worth watching, not yet
conclusive. (Note: `tail -c <N>` on this log can land inside one of the large periodic
weight-monitor dumps rather than the true end of file — use `grep`/full-file `tail -n` instead of
byte-based tail when checking recency.)

**Health check, step ~5500**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
5400, single wandb session, no restarts, no errors). Crossed the trainer's stage-1→stage-2
boundary at step 5000 (`ce_weight` 0.1→1.0, `doc_access_per_query_loss` weight 1.0→0.1, per
`staged_batched_isolation_docaccess_warmstart.yaml`'s two-stage schedule) — `doc_access_acc`
dropped back to 0.495 (from 0.665) right at this transition, which retroactively explains the
prior check's "climbing" reading as an end-of-stage-1 artifact (heavy retrieval-loss weighting)
rather than a lasting trend, not a regression. `doc_access_per_query_loss=0.288`,
`mem_pos_weight_mass/mean=0.836`, `mem_hit_rate/mean=1.0`.

**Health check, step ~6100**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
6000, single wandb session, no errors). Now well into stage 2 —
`doc_access_per_query_loss=0.330`, `doc_access_acc=0.428`, `mem_pos_weight_mass/mean=0.796`,
`mem_hit_rate/mean=1.0`. Both retrieval metrics drifting slightly worse than the stage-1→2
transition point, consistent with the expected stage-2 tradeoff (CE dominates at weight 1.0,
retrieval loss down to weight 0.1) rather than a new problem.

**`rohun-v6e-8-0` disk wiped a second time**, discovered the same way (fresh `createTime`,
1-minute `uptime`, empty home dir, missing log despite the node reporting `READY`) after progress
had reached step 6200 per GCS (`gsutil ls` across the current run-dir). Recovery followed the
same playbook: repo re-synced via one launcher pass (crashed on missing weights as expected, same
as the first wipe), re-staged `stage_qwen3_4b_weights.sh` + `stage_multihop_hardneg_data.sh` +
`stage_multihop_hardneg_cot_data.sh`, then relaunched a full resume from step 6200. Confirmed
genuine: `[wmon step=6200]`, distinct `mem_q_proj`/`mem_o_proj` rms (0.01929 / 0.01697), first
`Loss:` print at ~10 min compile, no tracebacks. New wandb run:
`multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-17-01-00`.

**False-hang-alarm gotcha, worth recording**: mid-compile, one `ps -o pid,time,pcpu,stat` reading
showed CPU time flat across a 10-second recheck (`00:05:56` both times), which looked like a
stall. A follow-up check ~90s later showed CPU time had actually jumped to `00:07:53` — the
process was just in a bursty/quiet phase (likely blocked briefly on GCS I/O during restore), not
hung. **Don't declare a hang from a single flat CPU-time snapshot** — take at least two readings
a minute or more apart, and only treat it as stuck if CPU time is flat across *that* longer
window.

**Eventful cycle: `rohun-v6e-8-0` fully deleted, then a genuine deadlock on `rohun-v6e-8-1`, then
a zone-wide preemption caught it mid-recovery.** In order:
1. `rohun-v6e-8-0` went `NOT_FOUND` (fully deleted, its third failure — see disk-wipe notes
   above) after reaching step 6200; GCS showed true progress at step 6600.
2. Switched to `rohun-v6e-8-1` (fresh box, re-staged weights/data). First launch attempt
   deadlocked mid-compile: `ps` showed 1499 threads, one blocked in `futex_wait_queue`, and CPU
   time was genuinely flat across a **3-minute** window (`00:05:48` → `00:05:49`) — a real stall,
   correctly distinguished from the false alarm above by using a longer recheck window (this is
   the actual rule going forward: 2-3+ minutes apart, not 10-90s).
3. Killed the hung process (`pkill -9`) and confirmed clean (`/dev/vfio/*` empty, no tmux
   session) — the kill succeeded even though the SSH command reporting it disconnected mid-way
   (`ssh exited 255`), likely because force-killing ~1500 threads causes a brief kernel-level
   stall on the box that drops the SSH session; re-checking afterward confirmed the process was
   actually gone.
4. Retried on the same box (`rohun-v6e-8-1`): CPU time advanced healthily (`00:04:15` → `00:05:52`
   over 3 min), `[wmon step=6600]` matched, distinct `mem_q_proj`/`mem_o_proj` rms (0.01928 /
   0.01698) — genuinely healthy, unlike step 2.
5. Before the first `Loss:` print landed, the SSH tunnel itself started failing with `[4003:
   'failed to connect to backend']` (IAP tunnel can't reach port 22) — `describe` then showed
   `rohun-v6e-8-1` in `DELETING` state. Checking `tpu-vm list` for the zone showed **another
   user's box (`john-v6e-8-5`) also `DELETING` at the same moment** — this is the same zone-wide
   `us-east1-d` spot-reclamation pattern documented earlier in this doc, not a bug in the recovery
   procedure. Both `rohun-v6e-8-0` (`NOT_FOUND`) and `rohun-v6e-8-1` (`DELETING`) are down as of
   this write-up; the standing `/loop` will wait for one to come back `READY` before relaunching
   from the step-6600 checkpoint again.
6. Follow-up check: both boxes now fully `NOT_FOUND`, and `gcloud alpha compute tpus
   queued-resources list --zone=us-east1-d` shows **all five** queued resources in the zone
   (`rohun-v6e-8-0`, `rohun-v6e-8-1`, `john-v6e-8-4/5/6`) stuck at `WAITING_FOR_RESOURCES` —
   a genuine zone-wide capacity crunch, not an isolated blip. Nothing to relaunch until capacity
   frees up; the loop will keep checking at a slightly longer interval.
7. After 3 no-change checks (spaced up to the ~3600s max), capacity freed: both nodes came back
   `READY`/`ACTIVE`, both fully fresh (re-created, no `~/memory-layers`). `rohun-v6e-8-0` flipped
   straight back to `DELETING` before the sync-and-launch attempt even finished. Switched to
   `rohun-v6e-8-1`; mid-launch (during the setup/sync pass) the SSH tunnel failed with the same
   `[4003: 'failed to connect to backend']` symptom as the earlier zone-wide event, and `describe`
   then showed `rohun-v6e-8-1` `PREEMPTED` while `rohun-v6e-8-0` showed `SUSPENDING` — both boxes
   churning READY→unstable within minutes of becoming available. Reads as heavy ongoing spot
   contention in `us-east1-d` (capacity briefly freeing up only to be immediately re-claimed), not
   a one-off blip. Not forcing a relaunch into this churn; waiting for one box to hold `READY` for
   more than a few minutes before the next attempt.
8. Next cycle: both boxes held `READY` through an initial check *and* a 45s re-check, so attempted
   a relaunch on `rohun-v6e-8-1` — repo synced, weights/data staged successfully (all completed
   without error) — but the box flipped to `DELETING` again at the exact moment of the actual
   training relaunch (SCP failed: `"This TPU has state DELETING"`). Both boxes came back
   `DELETING` on the immediate follow-up check. Even a clean 45s hold and a fully successful
   staging pass didn't guarantee the box would survive to the relaunch step — the churn is bad
   enough that "READY for 45s" isn't a reliable go/no-go signal right now. Backing off further
   before the next attempt rather than continuing to retry into this.
9. After several no-change checks spaced up to the ~3600s max, both boxes came back `READY`.
   This time waited out a **~2.5 minute** stability window (two rechecks, ~60s then ~90s apart)
   before doing anything — both held. Proceeded on `rohun-v6e-8-1`: re-checked state right before
   AND right after staging (both `READY`), repo sync + full staging pass + the actual training
   relaunch command all completed cleanly with **no SSH/connection errors at all** — the most
   stable sequence yet. New wandb run `multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-23-51-56`
   started, checkpoint restore began with no errors. Then, mid-compile (before the first `Loss:`
   print), the same `[4003: 'failed to connect to backend']` symptom hit again, and `describe`
   showed `rohun-v6e-8-1` back to `DELETING` (`rohun-v6e-8-0` was `READY` at that same moment).
   **This is the key new data point**: a launch that passed a validated multi-minute stability
   check, a fully clean staging pass, AND a clean launch-command execution *still* got preempted
   mid-compile. The lesson isn't "check longer before launching" — pre-launch stability, however
   long, doesn't predict survival through the ~10-minute compile window itself. The underlying
   preemption rate in this pool is just high enough right now that any given multi-minute window
   (before OR during a launch) has a meaningful chance of hitting one. Not chasing this further
   within the same cycle per the standing rule; the next cycle will just try again.

`rohun-v6e-8-0` has now lost its disk twice in a row; next time this box needs a relaunch, try
`rohun-v6e-8-1` instead to see if it's more stable.

10. Next cycle: both boxes `READY` again, held through a brief ~60s recheck (per the point-9
    lesson, no longer over-investing in long pre-checks). Relaunched on `rohun-v6e-8-1` — repo
    sync, staging, and the launch command all completed with **zero connection errors**. Compile
    ran long (~10 min) with one false alarm along the way (a `ps` snapshot briefly looked flat,
    then a follow-up 2.5-min-apart reading showed a large CPU-time jump, confirming it was
    genuinely still working — same validated two-reading rule as before). First `Loss:` print
    landed clean, and the first checkpoint save revealed a real, useful finding: the restored step
    was **6400**, not the step-6600 checkpoint directory `gsutil ls` had shown existing. The 6600
    directory on GCS was apparently incomplete — almost certainly written by an earlier launch
    that got killed mid-write during this same extended churn period — and `CheckpointManager`
    correctly skipped it, falling back to the last fully-committed checkpoint. **Lesson**: a step
    directory appearing in `gsutil ls` is not proof that checkpoint is complete/usable; trust the
    `wmon step=` label after a resume (or the checkpoint manager's own resolution) over a raw GCS
    listing when determining true recoverable progress, especially following an abrupt kill.

    The run then hit a **new, distinct connectivity symptom**: `"!!! This TPU is going through a
    maintenance event, and might be unavailable !!!"` — a GCP-initiated maintenance/live-migration
    notice, not spot reclamation. `describe` kept reporting `READY` through this, suggesting the
    node itself intended to survive; retried the SSH check, and although the training process was
    still presumably alive at that instant, the box formally flipped to `PREEMPTED` shortly after
    on a follow-up check. Net progress this attempt: one checkpoint saved (step 6400, the resume
    point itself) and no further checkpoint intervals reached before preemption — effectively a
    wash on real training progress, though the recovery mechanics (multi-min stability check,
    clean staging, clean launch, genuine resume verification) all worked as designed. Not chasing
    further within this cycle; the next cycle will try again from whatever checkpoint is latest.

11. Next cycle: both boxes `READY`, held a brief ~60s recheck, relaunched on `rohun-v6e-8-1` again
    (resume from the same step-6400 checkpoint — no newer one existed). Repo sync + staging +
    launch all clean again. Compile ran long this time too (~10.5 min), including a two-stage
    false-alarm pattern — first a near-flat `ps` reading over ~2:50 (5:44→5:45), **then a second
    near-flat reading right after that** (5:45, unchanged over another ~3 min), before a third
    check finally showed a huge jump (5:45→33:07) confirming genuine activity. Worth noting: two
    consecutive quiet readings did NOT mean a stall this time either — the bursty compile pattern
    can apparently span more than one 3-minute window before the CPU-heavy phase kicks in, so even
    two flat readings in a row aren't automatically conclusive; a third check settled it.
    `[wmon step=6400]` confirmed, first `Loss:` printed, first checkpoint saved (step 6400) with
    no errors — genuine resume confirmed the same way as before.

    Then, waiting for a SECOND checkpoint (the actual bar for "sustained" this cycle), the
    **exact same `"TPU is going through a maintenance event"` SSH symptom hit again** — the second
    occurrence of this specific failure mode, and both times it happened shortly after the first
    checkpoint save on `rohun-v6e-8-1` specifically. This time `describe` initially still showed
    `READY` (unlike the first occurrence, which showed `PREEMPTED` right away) — a retry was
    attempted to see if the process had survived, but by the next check the box had moved to
    `DELETING`. Net progress: identical outcome to the prior cycle — one checkpoint at step 6400,
    no further intervals reached. **This is now a confirmed repeating pattern specifically on
    `rohun-v6e-8-1`** (twice, both landing right around/after the first post-resume checkpoint
    save) — worth trying `rohun-v6e-8-0` for the next attempt instead, in case this is somehow
    specific to that node's current maintenance schedule rather than the pool in general.

12. **Resolution: `rohun-v6e-8-0` gave a genuinely sustained recovery.** Both boxes were `READY`;
    held a brief ~60s recheck; relaunched on `rohun-v6e-8-0` this time (resume from the same
    step-6400 checkpoint). Repo sync + staging + launch all clean. Compile again ran long (~10.5
    min) with a modest-but-real CPU-time increase on the first check (not the dramatic
    flat-then-huge-jump pattern this time — just steady, moderate progress), confirming activity
    without a false alarm. `[wmon step=6400]` confirmed, first `Loss:` printed with normal values
    (~0.28-0.42, fluctuating batch-to-batch as expected), first checkpoint saved (step 6400).

    Kept waiting past the "first checkpoint" bar this time — training continued steadily at a
    consistent **~2s/step** once past the initial compile, loss values stayed in the normal
    range, and **a second checkpoint saved at step 6600** (confirmed via `[wmon step=6600]` and
    the tqdm counter reaching 201/100000) with the process still healthy (CPU actively running,
    `RLl+`/131% CPU) and the box still `READY` throughout. **Critically, `rohun-v6e-8-0` did NOT
    hit the "maintenance event" symptom** that killed `rohun-v6e-8-1` twice in a row — supporting
    the theory that failure mode was specific to `rohun-v6e-8-1`'s current maintenance schedule,
    not a pool-wide issue. This is the first genuinely sustained recovery since the extended
    `us-east1-d` capacity crunch began (see points 6-11): **CE-only arm is running again**, on
    `rohun-v6e-8-0`, past step 6600, wandb run
    `multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-03-08-02`.

**Health check, step ~7200**: still on `rohun-v6e-8-0`, one continuous run since the sustained
recovery above — checkpoints saved sequentially 6600→6800→7000, no restarts, no errors.
`doc_access_per_query_loss=0.215`, `doc_access_acc=0.472`, `mem_pos_weight_mass/mean=0.887`,
`mem_hit_rate/mean=1.0` — in line with the established stage-2 trend from before the churn period.

**Health check, step ~7800**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
7600, no restarts, no errors). `doc_access_per_query_loss=0.233`, `doc_access_acc=0.514`,
`mem_pos_weight_mass/mean=0.859`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~8300**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
8200, no restarts, no errors). `doc_access_per_query_loss=0.228`, `doc_access_acc=0.583`,
`mem_pos_weight_mass/mean=0.868`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~8900**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
8800, no restarts, no errors). `doc_access_per_query_loss=0.277`, `doc_access_acc=0.466`,
`mem_pos_weight_mass/mean=0.882`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~9500**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
9400, no restarts, no errors). `doc_access_per_query_loss=0.181`, `doc_access_acc=0.534`,
`mem_pos_weight_mass/mean=0.892`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step 10000**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
10000, no restarts, no errors). `doc_access_per_query_loss=0.208`, `doc_access_acc=0.515`,
`mem_pos_weight_mass/mean=0.887`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~10600**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
10600, no restarts, no errors). `doc_access_per_query_loss=0.234`, `doc_access_acc=0.457`,
`mem_pos_weight_mass/mean=0.854`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~11200**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
11200, no restarts, no errors). `doc_access_per_query_loss=0.294`, `doc_access_acc=0.530`,
`mem_pos_weight_mass/mean=0.816`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~11800**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
11600, no restarts, no errors). `doc_access_per_query_loss=0.320`, `doc_access_acc=0.559`,
`mem_pos_weight_mass/mean=0.860`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~12300**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
12200, no restarts, no errors). Latest single-row `doc_access_per_query_loss=0.473` looked like an
outlier at first glance, but the last-20 window (0.046-0.473) confirms it's normal batch-to-batch
noise, not a regression — the point right after it already dropped back to 0.243.
`mem_pos_weight_mass/mean=0.764`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~12900**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
12800, no restarts, no errors). `doc_access_per_query_loss=0.125`, `doc_access_acc=0.557`,
`mem_pos_weight_mass/mean=0.919`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~13500**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
13400, no restarts, no errors). `doc_access_per_query_loss=0.194`, `doc_access_acc=0.546`,
`mem_pos_weight_mass/mean=0.897`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~14100**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
14000, no restarts, no errors). `doc_access_per_query_loss=0.221`, `doc_access_acc=0.610`,
`mem_pos_weight_mass/mean=0.878`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~14700**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
14600, no restarts, no errors). Latest single-row `doc_access_per_query_loss=0.528` again just
batch noise (last-20 window 0.08-0.53, consistent with the established range).
`doc_access_acc=0.486`, `mem_pos_weight_mass/mean=0.786`, `mem_hit_rate/mean=1.0` — steady.

**Health check, step ~15200**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
15200, no restarts, no errors). Latest single-row `doc_access_per_query_loss=0.691` is the highest
spike seen yet, but the last-30 window (0.054-0.691) confirms it's still within the established
noisy range, and the point right after already dropped back to 0.137 — not a regression.
`doc_access_acc`/`mem_pos_weight_mass` also within normal fluctuation bands across the same window.
`mem_hit_rate/mean=0.999` — steady.

**Health check, step ~15800**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
15800, no restarts, no errors). `doc_access_per_query_loss=0.542`, `doc_access_acc=0.486`,
`mem_pos_weight_mass/mean=0.735`, `mem_hit_rate/mean=1.0` — within established noisy range, no
new issues.

**Health check, step ~16400**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
16200, no restarts, no errors). `doc_access_per_query_loss=0.152`, `doc_access_acc=0.553`,
`mem_pos_weight_mass/mean=0.927`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~17000**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
16800, no restarts, no errors). `doc_access_per_query_loss=0.181`, `doc_access_acc=0.523`,
`mem_pos_weight_mass/mean=0.899`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~17500**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
17400, no restarts, no errors). `doc_access_per_query_loss=0.210`, `doc_access_acc=0.410`,
`mem_pos_weight_mass/mean=0.872`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~18100**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
18000, no restarts, no errors). `doc_access_per_query_loss=0.122`, `doc_access_acc=0.677`,
`mem_pos_weight_mass/mean=0.926`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~18700**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
18600, no restarts, no errors). `doc_access_per_query_loss=0.247`, `doc_access_acc=0.590`,
`mem_pos_weight_mass/mean=0.888`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~19380**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
19200, no restarts, no errors). `doc_access_per_query_loss=0.216` (last-30 range 0.041-0.506,
mean 0.185 — consistent with established noise band), `doc_access_acc=0.501`,
`mem_pos_weight_mass/mean=0.869`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~19940**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
19800, no restarts, no errors). `doc_access_per_query_loss=0.027` (last-30 range 0.017-0.327,
mean 0.165 — trending toward the good end), `doc_access_acc=0.769` (new high), `mem_pos_weight_mass/mean=0.981`,
`mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~20490**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
20400, no restarts, no errors). `doc_access_per_query_loss=0.308` (last-30 range 0.052-0.382,
mean 0.182 — consistent with established noise band), `doc_access_acc=0.588`,
`mem_pos_weight_mass/mean=0.866`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~21040**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
21000, no restarts, no errors). `doc_access_per_query_loss=0.167` (last-30 range 0.042-0.491,
mean 0.176 — consistent with established noise band), `doc_access_acc=0.639`,
`mem_pos_weight_mass/mean=0.906`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~21600**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
21400, no restarts, no errors). `doc_access_per_query_loss=0.146` (last-30 range 0.043-0.524,
mean 0.197 — consistent with established noise band), `doc_access_acc=0.532`,
`mem_pos_weight_mass/mean=0.911`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~22200**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
22000, no restarts, no errors). `doc_access_per_query_loss=0.112` (last-30 range 0.042-0.553,
mean 0.205 — consistent with established noise band), `doc_access_acc=0.427`,
`mem_pos_weight_mass/mean=0.926`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~22770**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
22600, no restarts, no errors). `doc_access_per_query_loss=0.048` (last-30 range 0.047-0.546,
mean 0.176 — consistent with established noise band), `doc_access_acc=0.745`,
`mem_pos_weight_mass/mean=0.968`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~23330**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
23200, no restarts, no errors). `doc_access_per_query_loss=0.123` (last-30 range 0.024-0.405,
mean 0.185 — consistent with established noise band), `doc_access_acc=0.580`,
`mem_pos_weight_mass/mean=0.940`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~23870**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
23800, no restarts, no errors). `doc_access_per_query_loss=0.204` (last-30 range 0.051-0.332,
mean 0.152 — consistent with established noise band), `doc_access_acc=0.524`,
`mem_pos_weight_mass/mean=0.886`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~24420**: still on `rohun-v6e-8-0`, one continuous run (checkpoints through
24400, no restarts, no errors). `doc_access_per_query_loss=0.146` (last-30 range 0.047-0.399,
mean 0.173 — consistent with established noise band), `doc_access_acc=0.524`,
`mem_pos_weight_mass/mean=0.907`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Incident, ~step 24800**: `rohun-v6e-8-0` came back `NOT_FOUND` (fully deleted, not just
preempted) on this health check, after checkpointing through step 24800. Switched to
`rohun-v6e-8-1` (READY, but a completely fresh box — no repo, no venv, no staged weights/data,
no tmux session). Re-staged from scratch via `multi-vm-tpu-run.sh`: weights
(`stage_qwen3_4b_weights.sh`), hard-neg parquet+corpus (`stage_multihop_hardneg_data.sh`), CoT
rows (`stage_multihop_hardneg_cot_data.sh`) — each run to completion (rc=0) before the next.
Relaunched as a full resume onto `rohun-v6e-8-1` with
`RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-03-08-02/qwen3_mem_embed`
(latest step visible via `gsutil ls` was 24800). First `ps` CPU-time reading right after launch
looked healthy (train.py at 401% CPU, actively compiling); awaiting the ~10-11 min compile
window and first post-resume checkpoint before calling this a confirmed recovery.

**New lesson**: this resume took ~23 minutes from launch to the first `Loss:` print/checkpoint
(vs. the usual ~10-11 min cold compile), because `[loader-restore]` had to fast-forward the
Grain dataloader stream all the way to step 24800's position before the first batch could be
produced — visibly different from a normal compile stall (multiple Grain multiprocess worker
processes stayed busy with real, steadily-growing CPU time throughout, and the main `train.py`
process's own CPU time kept climbing: 5:48 → 8:15 → 41:26 across three checks ~12-16 min apart).
Confirmed via the log: `[resume] re-applied stage 1 config at step 24800`, dataloader state
restored from the 24800 checkpoint's `dataloader_state.json`, weight-monitor rms values matching
the pre-crash values, and `Saved step 24800` as the very first checkpoint write post-resume — a
genuine, sustained recovery. New wandb run:
`multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-17-54-46`. Takeaway for future resumes at
high step counts: a long dataloader fast-forward with healthy, growing CPU time (not flat) should
not be treated as a hang even well past the usual ~10-11 min compile window.

**Health check, step ~24800**: sustained recovery confirmed on `rohun-v6e-8-1`, one continuous
run since the resume, first checkpoint saved at 24800 (matching resume point exactly), no
restarts, no errors. `doc_access_per_query_loss=0.071` (only 1 data point so far post-resume,
within established noise band), `doc_access_acc=0.707`, `mem_pos_weight_mass/mean=0.966`,
`mem_hit_rate/mean=1.0` — healthy, back to steady-state babysitting.

**Health check, step ~25210**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
25200, no restarts, no errors). `doc_access_per_query_loss=0.137` (last-30 range 0.029-0.408,
mean 0.171 — consistent with established noise band), `doc_access_acc=0.695`,
`mem_pos_weight_mass/mean=0.923`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~25800**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
25600, no restarts, no errors). `doc_access_per_query_loss=0.063` (last-30 range 0.029-0.827,
mean 0.171 — one isolated spike to 0.827 at step 25720, immediately followed by 0.039/0.029 at
the next two points, not a sustained trend), `doc_access_acc=0.712`, `mem_pos_weight_mass/mean=0.964`,
`mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~26400**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
26200, no restarts, no errors). `doc_access_per_query_loss=0.124` (last-30 range 0.039-0.304,
mean 0.179 — consistent with established noise band), `doc_access_acc=0.528`,
`mem_pos_weight_mass/mean=0.940`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~26980**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
26800, no restarts, no errors). `doc_access_per_query_loss=0.183` (last-30 range 0.035-0.769,
mean 0.220 — normal point-to-point noise, no sustained trend), `doc_access_acc=0.544`,
`mem_pos_weight_mass/mean=0.891`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~27560**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
27400, no restarts, no errors). `doc_access_per_query_loss=0.112` (last-30 range 0.026-0.493,
mean 0.181 — consistent with established noise band), `doc_access_acc=0.558`,
`mem_pos_weight_mass/mean=0.929`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~28120**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
28000, no restarts, no errors). `doc_access_per_query_loss=0.081` (last-30 range 0.046-0.393,
mean 0.175 — consistent with established noise band), `doc_access_acc=0.545`,
`mem_pos_weight_mass/mean=0.947`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~28670**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
28600, no restarts, no errors). `doc_access_per_query_loss=0.290` (last-30 range 0.011-0.657,
mean 0.209 — consistent with established noise band), `doc_access_acc=0.575`,
`mem_pos_weight_mass/mean=0.850`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~29210**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
29200, no restarts, no errors). `doc_access_per_query_loss=0.049` (last-30 range 0.039-0.440,
mean 0.164 — consistent with established noise band), `doc_access_acc=0.710`,
`mem_pos_weight_mass/mean=0.965`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~29800**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
29600, no restarts, no errors). `doc_access_per_query_loss=0.181` (last-30 range 0.057-0.558,
mean 0.195 — consistent with established noise band), `doc_access_acc=0.501`,
`mem_pos_weight_mass/mean=0.890`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Incident, ~step 29800**: both `rohun-v6e-8-0` AND `rohun-v6e-8-1` came back `NOT_FOUND` on
this health check. `gcloud alpha compute tpus queued-resources list --zone=us-east1-d` shows
both stuck at `WAITING_FOR_RESOURCES`, while other users' queued resources in the same zone
(`john-v6e-8-4/5`) are `ACTIVE` — a repeat of the earlier zone-wide `us-east1-d` capacity crunch
(see the Eventful cycle section), not something to fix on our end. Last confirmed checkpoint:
step 29800 on the `...-17-54-46` run-dir. Backing off with progressively longer checks rather
than aggressively retrying, per the established pattern for this condition.

**Recovery, ~25 min later**: `queued-resources list` showed both `rohun-v6e-8-0` and
`rohun-v6e-8-1` back to `ACTIVE`/READY — the zone-wide crunch resolved faster than the previous
occurrence. `rohun-v6e-8-1` was a completely fresh box (no repo/venv/tmux), re-staged from
scratch (weights, hard-neg data, hard-neg CoT data, all rc=0). Training had actually kept
checkpointing to step 30200 (4 more saves past the last-seen 29800) before the box was
reclaimed, so `gsutil ls` showed 30200 as the latest step in the `...-17-54-46` run-dir.
Relaunched a full resume from that run-dir; first `ps` check post-launch showed `train.py` at
472% CPU, actively compiling. Awaiting the first post-resume checkpoint before calling this a
confirmed sustained recovery. New wandb run:
`multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-23-34-53`. At the ~15-min check, still no
Loss print, but `train.py` CPU time grew 0:47→5:48 and worker processes stayed busy — same
dataloader-fast-forward pattern as the previous high-step-count resume, not a hang. This
fast-forward took ~26 min total before the first `Loss:`/`Saved step 30200` line appeared
(vs. ~23 min for the step-24800 resume) — CPU time kept climbing throughout (0:47→5:48→8:01→42:40)
right until the first print, confirming the "growing CPU time without a Loss print yet = healthy,
not hung" rule holds even for multi-tens-of-minutes fast-forwards at high step counts.

**Health check, step ~30200**: sustained recovery confirmed on `rohun-v6e-8-1`, one continuous
run since the resume, first checkpoint saved at 30200 (matching resume point exactly), no
restarts, no errors. `doc_access_per_query_loss=0.428` (only 1 data point so far post-resume,
within established noise band), `doc_access_acc=0.513`, `mem_pos_weight_mass/mean=0.765`,
`mem_hit_rate/mean=1.0` — healthy, back to steady-state babysitting.

**Health check, step ~30570**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
30400, no restarts, no errors). `doc_access_per_query_loss=0.255` (last-30 range 0.063-0.508,
mean 0.185 — consistent with established noise band), `doc_access_acc=0.450`,
`mem_pos_weight_mass/mean=0.861`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~31130**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
31000, no restarts, no errors). `doc_access_per_query_loss=0.058` (last-30 range 0.032-0.366,
mean 0.169 — consistent with established noise band), `doc_access_acc=0.649`,
`mem_pos_weight_mass/mean=0.960`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~31680**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
31600, no restarts, no errors). `doc_access_per_query_loss=0.287` (last-30 range 0.034-0.441,
mean 0.174 — consistent with established noise band), `doc_access_acc=0.523`,
`mem_pos_weight_mass/mean=0.862`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~32240**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
32200, no restarts, no errors). `doc_access_per_query_loss=0.120` (last-30 range 0.026-0.480,
mean 0.164 — consistent with established noise band), `doc_access_acc=0.525`,
`mem_pos_weight_mass/mean=0.928`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~32820**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
32800, no restarts, no errors). `doc_access_per_query_loss=0.100` (last-30 range 0.028-0.315,
mean 0.149 — consistent with established noise band), `doc_access_acc=0.514`,
`mem_pos_weight_mass/mean=0.937`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~33410**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
33400, no restarts, no errors). `doc_access_per_query_loss=0.017` (last-30 range 0.017-0.401,
mean 0.123 — consistent with established noise band), `doc_access_acc=0.648`,
`mem_pos_weight_mass/mean=0.986`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~34000**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
33800, no restarts, no errors). `doc_access_per_query_loss=0.081` (last-30 range 0.021-0.645,
mean 0.189 — consistent with established noise band), `doc_access_acc=0.610`,
`mem_pos_weight_mass/mean=0.942`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~34600**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
34400, no restarts, no errors). `doc_access_per_query_loss=0.105` (last-30 range 0.033-0.334,
mean 0.161 — consistent with established noise band), `doc_access_acc=0.600`,
`mem_pos_weight_mass/mean=0.934`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~35190**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
35000, no restarts, no errors). `doc_access_per_query_loss=0.091` (last-30 range 0.038-0.312,
mean 0.152 — consistent with established noise band), `doc_access_acc=0.718`,
`mem_pos_weight_mass/mean=0.949`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~35730**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
35600, no restarts, no errors). `doc_access_per_query_loss=0.141` (last-30 range 0.034-0.644,
mean 0.187 — consistent with established noise band), `doc_access_acc=0.464`,
`mem_pos_weight_mass/mean=0.905`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~36300**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
36200, no restarts, no errors). `doc_access_per_query_loss=0.170` (last-30 range 0.039-0.520,
mean 0.178 — consistent with established noise band), `doc_access_acc=0.525`,
`mem_pos_weight_mass/mean=0.900`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~36870**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
36800, no restarts, no errors). `doc_access_per_query_loss=0.295` (last-30 range 0.029-0.636,
mean 0.185 — consistent with established noise band), `doc_access_acc=0.588`,
`mem_pos_weight_mass/mean=0.864`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~37450**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
37400, no restarts, no errors). `doc_access_per_query_loss=0.413` (last-30 range 0.034-0.692,
mean 0.152 — point-to-point noise, not a sustained trend), `doc_access_acc=0.432`,
`mem_pos_weight_mass/mean=0.772`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~38020**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
38000, no restarts, no errors). `doc_access_per_query_loss=0.016` (last-30 range 0.016-0.620,
mean 0.171 — consistent with established noise band), `doc_access_acc=0.691`,
`mem_pos_weight_mass/mean=0.987`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~38600**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
38400, no restarts). A batch of ~10 `Traceback` lines appeared right after the step-38400
checkpoint save, all `Exception ignored in: SharedMemoryArray.__del__` /
`FileNotFoundError: [Errno 2] No such file or directory: '/psm_...'` — Python garbage-collector
noise from Grain's dataloader worker pool cleaning up already-unlinked POSIX shared-memory
segments (likely a routine worker recycle). Confirmed benign: no RESOURCE_EXHAUSTED/CRITICAL,
`grep -c "Tracking run with wandb"` still 1 (one continuous process), and training produced two
more normal checkpoints (38400, then progressed to `wmon step=38600`) right after with no gap in
Loss prints. New lesson: this specific traceback signature (`SharedMemoryArray.__del__` +
`shm_unlink`/`sem_unlink` `FileNotFoundError`) is safe to ignore in future cycles as long as
checkpointing and wmon progress continue uninterrupted around it — don't treat it as a crash.
`doc_access_per_query_loss=0.127` (last-30 range 0.022-0.480, mean 0.153 — consistent with
established noise band), `doc_access_acc=0.606`, `mem_pos_weight_mass/mean=0.919`,
`mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~39200**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
39000, no restarts, no RESOURCE_EXHAUSTED/CRITICAL). `doc_access_per_query_loss=0.050`
(last-30 range 0.017-0.393, mean 0.128 — consistent with established noise band),
`doc_access_acc=0.647`, `mem_pos_weight_mass/mean=0.973`, `mem_hit_rate/mean=1.0` — steady, no
new issues.

**Health check, step ~39760 — notable positive shift**: still on `rohun-v6e-8-1`, one continuous
run (checkpoints through 39600, no restarts, no RESOURCE_EXHAUSTED/CRITICAL). `train/ce_loss`
dropped sharply and sustainably around step ~39430, from its established ~0.25-0.33 band down to
~0.02-0.09, and has stayed there through step 39730 (a real trend across ~15 consecutive points,
not a single-point spike). `doc_access_per_query_loss=0.059` (last-30 range 0.018-0.178, mean
0.060 — also shifted down from the usual ~0.15-0.20 mean, consistent with the CE improvement),
`doc_access_acc=0.603`, `mem_pos_weight_mass/mean=0.956`, `mem_hit_rate/mean=1.0`. This reads as
genuine training progress (a real drop in main-task CE loss), not an anomaly — noting it here so
future cycles don't mistake the new lower baseline for missing data or a bug.

**Health check, step ~40320**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
40200, no restarts, no RESOURCE_EXHAUSTED/CRITICAL). `ce_loss` holding at the new lower baseline
(last-30 range 0.023-0.069, mean 0.043), `doc_access_per_query_loss=0.052` (last-30 range
0.007-0.173, mean 0.039 — continuing the downward shift from the ~39430 milestone),
`doc_access_acc=0.591`, `mem_pos_weight_mass/mean=0.967`, `mem_hit_rate/mean=1.0` — steady, no
new issues.

**Health check, step ~40880**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
40800, no restarts, no RESOURCE_EXHAUSTED/CRITICAL). `ce_loss` holding at the new lower baseline
(last-30 range 0.017-0.069, mean 0.033), `doc_access_per_query_loss=0.078` (last-30 range
0.005-0.100, mean 0.033 — consistent with the post-milestone range), `doc_access_acc=0.563`,
`mem_pos_weight_mass/mean=0.968`, `mem_hit_rate/mean=1.0` — steady, no new issues.

**Health check, step ~41450**: still on `rohun-v6e-8-1`, one continuous run (checkpoints through
41400, no restarts, no RESOURCE_EXHAUSTED/CRITICAL). Both `ce_loss` and `doc_access_per_query_loss`
continuing to trend down further (`ce_loss` last-30 range 0.010-0.034, mean 0.020;
`doc_access_per_query_loss` last-30 range 0.005-0.049, mean 0.019), `doc_access_acc=0.636`,
`mem_pos_weight_mass/mean=0.986`, `mem_hit_rate/mean=1.0` — steady, no new issues.

## Reproducibility

```bash
# commit: multihop-finetuning branch, includes utils.py::restore_and_reshard dtype fix
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_ground4layer_s1warmstart_no_multihop.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Arm 2 (pf32/indexed/lr_masked source, main_model unfrozen):
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh
```

- **TPU:** v6e-8, `tn-v6e-8-0`, `memorylayers` project, `europe-west4-a`.
- **Checkpoint (warm start):** `gs://memory-layers-training/ground_s1_zeroinit_4layer_no_multihop-2026-08-11-00-52-33/qwen3_mem_embed/26000`.
- **wandb (arm 0, pre-fix):** `johnzhang2366-columbia-university/memory-layers/multihop_ground4layer_s1warmstart_no_mul-2026-08-13-00-32-22`
- **wandb (arm 1, post-fix):** `johnzhang2366-columbia-university/memory-layers/multihop_ground4layer_s1warmstart_no_mul-2026-08-13-02-47-25`
- **Checkpoint (arm 2 warm start):** `gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45/qwen3_mem_embed/100000`.
- **wandb (arm 2):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-12-20-07`
- **CoT-joined dataset:** [`ragrawal36/multihop_qa_sft-hard-neg-cot`](https://huggingface.co/datasets/ragrawal36/multihop_qa_sft-hard-neg-cot) (private).
- **wandb (CoT ablation, final):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-15-44-42`
- **wandb (CE-only arm, steps 0-600):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-16-45-43`
  (ran on `rohun-v6e-8-0` until a zone-wide spot preemption in `us-east1-d`; checkpoints survived
  to step 600 on GCS. Superseded run id `.../multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-16-05-36`
  on `rohun-v6e-8-1` before that — logged a clean step 0→1 but rohunagrawal reported it wasn't
  actually running; not confirmed why, not chased further at their direction.)
- **wandb (CE-only arm, steps 600+):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-18-27-09`
  (full resume of the above onto `tn-v6e-8-0`, once the CoT-ablation arm was stopped to free it —
  new run-dir/wandb-id is expected behavior for a full resume, not a break in continuity of the
  underlying model/optimizer state).
- **wandb (CE-only arm, steps 2200-2400, telemetry on):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-13-22-21-38`
  (full resume from the step-2200 checkpoint above, after adding the weight-0 `mem_*` telemetry
  block to `staged_batched_isolation_docaccess_warmstart.yaml` — see note above; ran on
  `tn-v6e-8-0` until that node was deleted by spot reclamation).
- **wandb (CE-only arm, steps 2400-4000, telemetry on):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-00-28-34`
  (full resume of the above onto `rohun-v6e-8-0`, `us-east1-d`, after `tn-v6e-8-0` was deleted —
  a fresh box, so needed `stage_qwen3_4b_weights.sh` + `stage_multihop_hardneg_data.sh` +
  `stage_multihop_hardneg_cot_data.sh` run first; see note above for the `.env`-sourcing bug fixed
  in the last of those. This box's disk was later wiped mid-run — see below.)
- **wandb (CE-only arm, steps 4000-6200, telemetry on):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-14-24-00`
  (full resume onto the same `rohun-v6e-8-0` name after its disk was silently wiped — re-staged
  from scratch again; see the "disk got wiped mid-run" note above).
- **wandb (CE-only arm, steps 6200-6400, telemetry on):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-14-17-01-00`
  (full resume onto `rohun-v6e-8-0` after a SECOND disk wipe — see the "disk wiped a second time"
  note above; the 6600 checkpoint under this run-dir turned out incomplete, see point 10).
- **wandb (CE-only arm, step 6400, brief):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-01-00-55`
  (recovered from a zone-wide `us-east1-d` capacity crunch, see points 6-9; preempted shortly after
  the first checkpoint save).
- **wandb (CE-only arm, step 6400, brief, maintenance-event):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-02-03-08`
  (hit the recurring "TPU going through a maintenance event" symptom on `rohun-v6e-8-1`, see point 11).
- **wandb (CE-only arm, steps 6400-24800, sustained):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-03-08-02`
- **wandb (CE-only arm, steps 24800-30200, after rohun-v6e-8-0 was fully deleted):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-17-54-46`
- **wandb (CE-only arm, steps 30200+, after a zone-wide capacity crunch took down both boxes):** `johnzhang2366-columbia-university/memory-layers/multihop_lrmasked_warmstart_docaccess_ba-2026-08-15-23-34-53`
  (full resume onto `rohun-v6e-8-0` instead of `rohun-v6e-8-1` — genuinely sustained past step
  6600, no maintenance-event symptom; see point 12, the resolution of the extended churn saga).

```bash
# CE-only arm, full resume onto tn-v6e-8-0 (RESUME_FROM must include /qwen3_mem_embed, or the
# checkpoint manager silently finds nothing and trains from a fresh init instead of erroring):
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-13-16-45-43/qwen3_mem_embed" \
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed again from step 2200 with telemetry enabled:
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-13-18-27-09/qwen3_mem_embed" \
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed onto rohun-v6e-8-0 after tn-v6e-8-0 was deleted. On a fresh box, run
# stage_qwen3_4b_weights.sh + stage_multihop_hardneg_data.sh + stage_multihop_hardneg_cot_data.sh
# first (see wiki/infrastructure/experiment-launch-instructions.md) -- staging must FINISH before
# this launch command, not just be kicked off in parallel, or it 401s/crashes on missing weights.
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-13-22-21-38/qwen3_mem_embed" \
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed again from step 4000 after rohun-v6e-8-0's disk was silently wiped:
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-14-00-28-34/qwen3_mem_embed" \
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed again from step 6200 after rohun-v6e-8-0's disk was silently wiped a SECOND
# time:
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-14-14-24-00/qwen3_mem_embed" \
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed from step 6400 (current, sustained -- genuinely running past step 6600):
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-02-03-08/qwen3_mem_embed" \
TPU_NAME=rohun-v6e-8-0 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh

# Same arm, resumed from step 24800 after rohun-v6e-8-0 came back NOT_FOUND (fully deleted).
# Moved to rohun-v6e-8-1, a fresh box requiring full re-staging first:
TPU_NAME=rohun-v6e-8-1 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/misc/stage_qwen3_4b_weights.sh bash scripts/infrastructure/multi-vm-tpu-run.sh
TPU_NAME=rohun-v6e-8-1 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/misc/stage_multihop_hardneg_data.sh bash scripts/infrastructure/multi-vm-tpu-run.sh
TPU_NAME=rohun-v6e-8-1 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/misc/stage_multihop_hardneg_cot_data.sh bash scripts/infrastructure/multi-vm-tpu-run.sh
RUN_ENV="RESUME_FROM=gs://memory-layers-training/multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only-2026-08-15-03-08-02/qwen3_mem_embed" \
TPU_NAME=rohun-v6e-8-1 ZONE=us-east1-d PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ce_only.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh
```

```bash
# CoT-injection ablation (retrieval + CE both see CoT):
TPU_NAME=tn-v6e-8-0 ZONE=europe-west4-a PROJECT_ID=memorylayers \
  RUN_SCRIPT_PATH=scripts/embed/train_multihop_lrmasked_warmstart_cot_ablation.sh \
  FOLLOW=0 bash scripts/infrastructure/multi-vm-tpu-run.sh
```
