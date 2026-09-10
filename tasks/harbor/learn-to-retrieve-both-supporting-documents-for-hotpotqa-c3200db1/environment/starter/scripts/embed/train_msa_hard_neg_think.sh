uv run train.py \
    model=qwen3_msa_train \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.msa.top_k_docs=16 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_msa \
    trainer.eval_interval=2000 \
    trainer.checkpoint_interval=10000 \
    eval_set@trainer.evals=msa_hard_neg_think_nll \
    +trainer.run_name="msa_hard_neg_think_sft4b_topk16_seq512_chunks16_bs16"
