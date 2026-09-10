#!/usr/bin/env bash
# Third arm alongside train_multihop_lrmasked_warmstart.sh (no CoT) and
# train_multihop_lrmasked_warmstart_cot_ablation.sh (CoT in memory bank AND CE): this one feeds
# CoT into the CE target ONLY (think_field), never into the memory bank (cot_field unset) -- see
# configs/dataset/sources/multihop_qa_sft_hard_neg_cot_ce_only.yaml and the "Ablation" section of
# wiki/experiments/2026-08-13-doc-access-per-query-loss-investigation.md.
#
# Same warm-start checkpoint, architecture overrides, trainer, and shuffle_seed as the other two
# arms -- only the dataset (hence what the answer/CE target contains) differs.
#
# DATA: stage the shared multihop parquet+corpus (datagen/download_multihop_hardneg.py) and the
# CoT-joined rows table (datagen/download_multihop_hardneg_cot.py) -- same data as the full CoT
# ablation arm, this one just doesn't set cot_field so qa_transform_item never appends it to the
# memory bank.
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
    dataset=multihop_hard_neg_full_cot_ce_only \
    dataset.shuffle_seed="${MULTIHOP_SHUFFLE_SEED:-20260813}" \
    dataset.num_chunks_per_doc="${MULTIHOP_NUM_CHUNKS_PER_DOC:-224}" \
    trainer=staged_batched_isolation_docaccess_warmstart \
    trainer.checkpoint_interval=200 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    trainer.resume_from="${RESUME_FROM:-$LRMASKED_CKPT}" \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_lrmasked_warmstart_docaccess_batched_iso_topk64_bs8_cot_ce_only"
# RESUME_FROM (env, run-dir with NO trailing step) takes priority over the warm start above and
# resumes THIS run (model + optimizer + dataloader) from its own latest checkpoint after a
# preemption -- same convention as the other multihop warm-start scripts.
