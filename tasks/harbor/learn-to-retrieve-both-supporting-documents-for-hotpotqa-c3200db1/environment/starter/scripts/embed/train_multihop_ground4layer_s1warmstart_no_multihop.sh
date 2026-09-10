#!/usr/bin/env bash
# Same recipe as train_multihop_ground4layer_s1warmstart.sh (multihop_hard_neg_full,
# WARM START not full resume -- see that script's header for the full mechanics/rationale), but
# warm-started from ground_s1_zeroinit_4layer_no_multihop instead of the original
# ground_s1_zeroinit_4layer, and using staged_ground_batched_isolation_docaccess_warmstart
# instead of the plain _stage2only trainer: a first 5000-step sub-stage pushes
# doc_access_per_query_loss to weight 1.0 (ce_weight down to 0.1) to prioritize retrieval
# quality against the harder multihop_hard_neg_full distribution before the normal
# ce_weight=1.0 / doc_access_per_query_loss=0.1 balance takes back over -- see that trainer
# config's header. Added 2026-08-13 as a first attempt at improving doc_access_loss/acc on
# this lineage; no results yet.
#
# Why a different source checkpoint: ground_s1_zeroinit_4layer_no_multihop
# (train_ground_s1_no_multihop.sh) trains the same zero-init 4-layer architecture on
# qa_hard_neg_no_multihop_sft4b (science_qa/diverseqa/combined, ragrawal36/multihop_qa_sft
# excluded) rather than the original qa_hard_neg_think_sft4b -- see that script's header for why
# the multihop source was excluded. This run "graduates" that checkpoint onto the actual
# multihop hard-neg objective (multihop_hard_neg_full, 200 hard negs/question) once grounding is
# established on the non-multihop mix, rather than mixing multihop in from the start.
#
# GROUND_S1_STEP: defaults to 26000, the step this lineage's warm start actually used (see
# ground_s1_zeroinit_4layer_no_multihop's stop point) -- override for a different step. Matters
# only for the warm-start path; when RESUME_FROM is set (see below), GROUND_S1_CKPT is computed
# but never used, so this default must stay resolvable (:-, not :?) or a RESUME_FROM relaunch
# would crash on an unset var it doesn't even need -- exactly the bug this comment is warning
# against (hit once already: FOUND 2026-08-11, see the no_multihop run's experiment notes).
# GROUND_S1_RUNDIR: the no_multihop run's GCS dir (defaults to the 2026-08-11-00-52-33 run
# launched on tn-v6e-8-0 -- override if resuming a different run of the same script).
#
# mem_layers=[9,14,20,27] / mem_o_proj_zero_init=true: MUST match the source checkpoint's
# architecture exactly (train_ground_s1_no_multihop.sh uses the same values) -- see
# train_multihop_ground4layer_s1warmstart.sh's header for why a mismatch silently falls back to
# fresh-init leaves instead of erroring.
#
# DATA: offline-parquet + doc-id corpus. Stage BOTH on the box first (idempotent):
#     uv run python datagen/download_multihop_hardneg.py
GROUND_S1_RUNDIR="${GROUND_S1_RUNDIR:-gs://memory-layers-training/ground_s1_zeroinit_4layer_no_multihop-2026-08-11-00-52-33}"
GROUND_S1_CKPT="${GROUND_S1_RUNDIR}/qwen3_mem_embed/${GROUND_S1_STEP:-26000}"
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    model.memory.mem_o_proj_zero_init=true \
    'model.memory.mem_layers=[9,14,20,27]' \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    +model.memory.mem_batched_isolation=true \
    model.memory.mem_collect_full_scores=true \
    dataset=multihop_hard_neg_full \
    trainer=staged_ground_batched_isolation_docaccess_warmstart \
    trainer.checkpoint_interval=200 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    trainer.resume_from="${RESUME_FROM:-$GROUND_S1_CKPT}" \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_ground4layer_s1warmstart_no_multihop_docaccess_warmstart_batched_iso_topk128_bs8"
# RESUME_FROM (env, run-dir with NO trailing step) takes priority over the warm start above and
# resumes THIS run (model + optimizer + dataloader) from its own latest checkpoint after a
# preemption -- same convention as train_multihop_ground4layer_s1warmstart.sh.
