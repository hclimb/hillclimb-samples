# multihop_hard_neg_full (200 hard negs/question) + train_ground_s1.sh's architecture, at 3
# memory layers instead of 4, with telemetry on: zero-init mem_o_proj (memory branch starts as
# identity; any CE drop is provably memory-sourced) + multi-layer memory. Everything else
# (mem_batched_isolation true per-row retrieval, doc_access_per_query_loss,
# mem_collect_full_scores, offline-parquet data) carries over from
# train_multihop_hard_neg_full.sh — see wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md
# for the full context (why mem_batched_isolation exists, and why doc_access_loss needed
# replacing with doc_access_per_query_loss under it).
#
# mem_layers=[9,18,27]: train_ground_s1.sh's 4-layer set is [9,14,20,27] (Qwen3-4B,
# num_hidden_layers=36) -- an even spread with the SAME start (9) and end (27) anchors. For 3
# layers, [9,18,27] keeps those same anchors and evenly trisects the gap (step 9, not the
# original's uneven 5/6/7) rather than just dropping one of the original 4 -- closer to the
# original "spread signal across the network" intent than an arbitrary subset. Not verified
# against any specific ablation; flag if a different placement was intended.
#
# NOT enabled here, deliberately, though it would apply: `trainer.stop_grad_frozen=true`. Main is
# frozen in BOTH staged_ground stages (never in trainable_params), and stop_grad_frozen (see
# trainer.py) skips backward through non-trainable weights -- quality-neutral per its own
# docstring, "saves ~2x the step in frozen-main stages" -- which could matter more here than in
# the 1-layer recipe (3x the memory-layer retrieval compute now competes for the same budget).
# Not added because it's a new lever beyond what was asked for tonight; a clean follow-up if this
# run is memory/speed constrained.
#
# UNTESTED: whether 3 memory layers' retrieval (each layer independently builds a full per-row
# score/gather tensor at m_per_query=65,536) fits the ~31GB/chip budget at batch_size=8 alongside
# everything else -- the single-layer recipe already ran close to that ceiling. Smoke-test this,
# don't assume it works from the 1-layer numbers.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    model.memory.mem_o_proj_zero_init=true \
    'model.memory.mem_layers=[9,18,27]' \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    +model.memory.mem_batched_isolation=true \
    model.memory.mem_collect_full_scores=true \
    dataset=multihop_hard_neg_full \
    trainer=staged_ground_batched_isolation \
    trainer.checkpoint_interval=200 \
    trainer.max_to_keep=20 \
    trainer.log_interval=10 \
    ${RESUME_FROM:+trainer.resume_from="$RESUME_FROM"} \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_ground3layer_batched_iso_topk128_bs8"
