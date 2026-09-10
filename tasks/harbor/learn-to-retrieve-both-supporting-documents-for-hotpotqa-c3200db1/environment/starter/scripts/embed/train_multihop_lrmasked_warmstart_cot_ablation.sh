#!/usr/bin/env bash
# CoT-injection ablation arm of train_multihop_lrmasked_warmstart.sh -- see that script's header
# for the full setup rationale (source checkpoint, architecture-match requirements, the OOM/
# tp_devices/embed_tokens/num_chunks_per_doc debugging), and the "Ablation" section of
# wiki/experiments/2026-08-13-doc-access-per-query-loss-investigation.md for the hypothesis this
# is testing (does doc_access_per_query_loss collapse toward zero when the answer is trivially
# present in the memory bank, vs. plateauing on genuinely-hard negatives regardless).
#
# The difference from train_multihop_lrmasked_warmstart.sh: dataset=multihop_hard_neg_full_
# cot_ablation instead of multihop_hard_neg_full. That dataset's source
# (multihop_qa_sft_hard_neg_cot) is byte-identical to multihop_qa_sft_hard_neg's mined rows,
# with each row's CoT text joined in, appended as an EXTRA positive document in the memory bank
# AND prepended into the answer as <think>...</think> (both `cot_field` and `think_field` set --
# see configs/dataset/sources/multihop_qa_sft_hard_neg_cot.yaml -- rohunagrawal wants CoT
# feeding CE too, not just retrieval, so this is not a single-variable ablation, both the
# retrieval objective and the CE target differ from the base arm). Same warm-start checkpoint,
# same architecture overrides, same trainer, same shuffle_seed otherwise.
#
# DATA: stage the shared multihop parquet+corpus first (datagen/download_multihop_hardneg.py),
# then the CoT-joined rows table (datagen/download_multihop_hardneg_cot.py) -- see
# scripts/misc/stage_multihop_hardneg_data.sh / stage_multihop_hardneg_cot_data.sh.
LRMASKED_RUNDIR="${LRMASKED_RUNDIR:-gs://memory-layers-training/qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16_pf32_indexed_lr_masked-2026-08-09-05-49-45}"
LRMASKED_CKPT="${LRMASKED_RUNDIR}/qwen3_mem_embed/${LRMASKED_STEP:-100000}"
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    +model.memory.mem_batched_isolation=true \
    model.memory.mem_collect_full_scores=true \
    dataset=multihop_hard_neg_full_cot_ablation \
    dataset.shuffle_seed="${MULTIHOP_SHUFFLE_SEED:-20260813}" \
    dataset.num_chunks_per_doc="${MULTIHOP_NUM_CHUNKS_PER_DOC:-224}" \
    trainer=staged_batched_isolation_docaccess_warmstart \
    trainer.checkpoint_interval=200 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    trainer.resume_from="${RESUME_FROM:-$LRMASKED_CKPT}" \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ablation"
# RESUME_FROM (env, run-dir with NO trailing step) takes priority over the warm start above and
# resumes THIS run (model + optimizer + dataloader) from its own latest checkpoint after a
# preemption -- same convention as the other multihop warm-start scripts.
