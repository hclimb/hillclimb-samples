# multihop_hard_neg_full training run: 200 hard negatives/question (dataset=multihop_hard_neg_full,
# mihir-1999/multihop_qa_sft-hard-neg-train + doc-id corpus resolution — see
# configs/dataset/sources/multihop_qa_sft_hard_neg.yaml), with TRUE per-row memory retrieval
# (mem_batched_isolation) instead of the default cross-batch bank. See
# wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md for the full sweep and for
# why this is batch_size=8, not 16: batched isolation's own retrieval kernel is flat in B (fits at
# 16 in ~23.7G, retrieval-only), but the FIRST launch attempt at batch_size=16 OOM'd end-to-end
# (33.66G vs 31.25G) — the isolated benchmark didn't include the embed model's cost, which scales
# with B and apparently accounts for the gap. batch_size=8 is the verified-working point; getting
# 16 working end-to-end is a follow-up (remat on the embed model, most likely).
#
# DATA: offline-parquet + doc-id corpus, same offline-only rule as train_hard_neg_think.sh (HF's
# per-account rate limit turns live streaming into a livelock, wiki/data/hf-rate-limits.md).
# Stage BOTH artifacts on the box first (idempotent, skips if already staged):
#     uv run python datagen/download_multihop_hardneg.py
# Must run on EVERY host of the slice (each host's disk is separate) — see
# wiki/infrastructure/experiment-launch-instructions.md §2.3.
#
# ISOLATION: mem_batched_isolation requires per_query_isolation + isolation_group_size=1 set
# alongside it (models/memory.py::mem_lookup_batched raises NotImplementedError otherwise) —
# each query attends ONLY to its own ~204 docs (200 hard negs + positives), never another row's.
# That's exactly right for this recipe: with 200 dataset hard negatives per query there's no need
# for additional in-batch negatives from other rows.
#
# SMOKE-TEST ANYWAY: this is a brand-new config (new retrieval kernel + a dataset no one has
# trained on before) — watch for step 1 and check peak-memory headroom before walking away, don't
# just trust the isolated-retrieval benchmark numbers (this exact script OOM'd end-to-end once
# already at batch_size=16 despite the retrieval kernel fitting fine in isolation).
#
# EVALS: in-loop eval OFF (eval_set=none), same rationale as train_hard_neg_think.sh — a dedicated
# eval box is the intended path but wasn't stood up this session; add one as a follow-up.
#
# checkpoint_interval=200 + max_to_keep=20 (4000-step window): deliberately smaller than
# train_hard_neg_think.sh's 2000/16 — this recipe's per-step doc-token count is far higher (16
# docs -> 256, batch 16 -> 4M+ doc tokens/step vs that recipe's ~1M), so wall-clock/step is much
# higher and unknown until measured; checkpoint more often (in step count) until real step timing
# is known, then widen the interval to match trainer=staged_telemetry's usual cadence.
#
# FOUND (2026-08-02): this run OOM'd at EXACTLY step 5000, THREE times across three separate
# process launches, with near-identical "Attempting to allocate 5.94M. There are 4.0x M free" —
# not a leak (ruled out by recurring at the same step in fresh processes with fresh checkpoint
# managers), but configs/trainer/staged.yaml's Stage 0 -> Stage 1 boundary: max_step: 5000 is
# exactly where trainable_params grows from [".*mem_.*", ".*embed_proj_conv.*"] to
# [".*mem_.*", ".*embed_model.*"] — the WHOLE embed model (Qwen3-Embedding-0.6B) needs gradients
# from Stage 1 on, a real, deterministic memory step-up, not gradual growth.
#
# Attempt 1 fix (trainer=staged_telemetry -> staged, dropping aux telemetry) was NOT enough on
# its own — crashed again with essentially the same numbers. Attempt 2 fix (this one):
# checkpoint_interval=200 evenly divides max_step=5000, so a checkpoint SAVE (a real, sizeable
# device-side operation) was landing on the EXACT SAME STEP as the Stage 1 recompilation/
# optimizer-rebuild every single time — two demanding operations stacked on one step, right at
# the point where headroom is already thinnest. checkpoint_interval=333 (a number that does NOT
# evenly divide 5000, 10000, or 15000 — the stage boundaries in staged.yaml) decouples them: the
# nearest saves land at step 4995/5328 instead of exactly 5000, so the transition gets to run
# without a concurrent save. trainer.max_to_keep lowered to 15 to match (still a healthy
# multi-thousand-step retention window at this interval). mem_top_k/batch_size left untouched —
# unlike telemetry, they're part of the actual experiment this note is about, and this fix didn't
# need them. See wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md for the full
# diagnosis chain (including whether this decoupling theory actually holds).
# RESUME_FROM (env, no trailing step — see other train_*.sh scripts' convention) resumes model +
# optimizer + dataloader state from the latest checkpoint of an earlier run of this same script,
# so a crash loses at most checkpoint_interval steps, not the whole run.
#
# FOUND (2026-08-02, separate from all of the above): the entire run to this point trained on
# EXACTLY ZERO gradient signal. `Loss: 0.0000` in every single progress line, every stage, the
# whole session -- missed until directly asked why. Stages 0-1 have ce_weight=0.0, and
# doc_access_loss (the only other nonzero-weight aux loss under `trainer=staged`) needs
# aux_data["mem_scores"] as the FULL CROSS-BATCH [B,T,N,B*m_per_query] grid, which
# mem_lookup_batched structurally never builds -- so it silently hit its own `if not
# mem_scores_list: return 0.0` guard, every step. trainer=staged_batched_isolation (below) swaps
# in doc_access_per_query_loss (losses/doc_access_per_query_loss.py) -- the same contrastive
# objective restricted to a query's own slots, fed by mem_lookup_batched's new (opt-in,
# mem_collect_full_scores-gated) per-row mem_scores. See
# wiki/implementations/2026-08-02-hard-neg-full-efficient-retrieval.md.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
MULTIHOP_CORPUS="${MULTIHOP_CORPUS:-$HOME/hf_parquet/multihop_doc_corpus.arrow}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=128 \
    +model.memory.per_query_isolation=true \
    +model.memory.isolation_group_size=1 \
    +model.memory.mem_batched_isolation=true \
    model.memory.mem_collect_full_scores=true \
    dataset=multihop_hard_neg_full \
    trainer=staged_batched_isolation \
    trainer.checkpoint_interval=333 \
    trainer.max_to_keep=15 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    ${RESUME_FROM:+trainer.resume_from="$RESUME_FROM"} \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="multihop_hard_neg_full_batched_iso_topk128_bs8_docaccesspq"
