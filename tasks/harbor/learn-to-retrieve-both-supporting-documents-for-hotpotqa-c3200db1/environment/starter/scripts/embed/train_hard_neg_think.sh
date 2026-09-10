# Hard-neg (think) SFT4B training run. Evals do NOT run in this loop — see "Evals" below.
#
# DATA: offline-parquet. BOTH env vars below are REQUIRED and neither is optional:
#   * GROUND_HF_PARQUET alone does NOTHING — data/qa.py:313 gates the offline branch on
#     HF_HUB_OFFLINE == "1", so you would silently stream live.
#   * live-HF is not a fallback, it is a LIVELOCK. wiki/data/hf-rate-limits.md: 16 grain workers x
#     4 sources = 64 shard resolutions blow HF's per-account 1000-req/5-min quota during pipeline
#     build; qa.py:552 then catches the 429 with an `except Exception` written for worker OOM and
#     REBUILDS the pipeline, re-blowing the quota. Measured on this exact script: 12 rebuild cycles,
#     47 min, still step 0, no checkpoint. Offline-parquet reads local disk: 49.7 batch/s, 0 stalls.
# Populate the cache first (on the box) — ~13 GB, one balanced 100k-step epoch:
#     bash scripts/misc/download_hard_neg_data.sh
# It sizes by rows needed (100k x 16 = 1.6M, 400k per source) rather than a blanket shard fraction,
# so no source runs dry and gets silently repeated by interleave stopping_strategy='all_exhausted'.
# See wiki/data/epoch-sized-data.md. Verified on rohun-v6e-8-0: 455,647 / 441,691 / 509,858 /
# 502,988 rows per source, 13G on disk.
# SMOKE-TEST ANYWAY: watch for step 1 before walking away.
# trainer.log_interval=10 pipelines the loop (pull losses/log every 10 steps) for ~+10% throughput
# with no change to training math (Bottleneck B); set =1 for per-step logging.
# trainer.stop_grad_frozen (Axis A): ~2.12x faster in the frozen-main warmup stages 0-2. It is
# quality-neutral in EXACT arithmetic (proven; stop_gradient on a frozen weight can't change a
# trainable weight's grad) — but at bf16 it perturbs trainable grads ~3% (reduction-order noise on
# the M=65k score backward). Left OFF here pending a short on-vs-off loss A/B; flip to =true to enable.
#
# TELEMETRY: trainer=staged_telemetry is the usual `staged` recipe + the weight-0 read-channel
# block (train/mem_pos_weight_mass, mem_hit_rate, mem_top1_weight, mem_topk_entropy, ...). Weight 0
# => stop_gradient'd, so it logs every log_interval steps and changes the training math not at all.
#
# EVALS: in-loop eval is OFF (eval_set=none + eval_interval huge), and that is deliberate, not an
# omission. llm_judge_accuracy / lexical_grounding are computed by evals/shared.py::
# run_metrics_pipeline in eval.py's PARENT process, after the JAX worker exits and frees the TPU for
# the vLLM judge; trainer._run_evals only keeps evaluator.evaluate()'s inference_metrics and never
# calls run_metrics, so a task's `metrics:` block is INERT in-loop. Judged accuracy comes from a
# dedicated eval box instead, which also keeps training at full throughput and off the judge's TPU:
#     RUN_NAME=... bash scripts/embed/hard_neg_eval_box_run.sh     (on a SEPARATE rohun-* box)
# It evaluates eval_set=hard_neg_think_c512 (msmarco/hotpotqa/musique @512 docs + scienceQA NLL,
# n=128) at every checkpoint and logs eval/* INTO THIS wandb run against a `train_step` x-axis.
#
# wandb_run_id=auto is what makes that one-run merge possible: it derives a deterministic wandb id
# from run_name so the eval box can attach to this exact run as a shared-mode secondary writer
# (and so a preemption-resume continues this curve instead of opening a second run).
#
# checkpoint_interval=2000 + max_to_keep=16. These two MUST be set together. Orbax keeps only the
# last max_to_keep checkpoints, so a checkpoint stays evaluable for max_to_keep * interval steps.
# At the default max_to_keep=4, a 2000-step interval gives an 8000-step window (~65 min at
# ~490ms/step) — SHORTER than the ~1-1.5h an eval cycle takes, so the box would lose checkpoints to
# rotation before scoring them and the eval curve would come out full of holes. 16 x 2000 = 32k
# steps (~4.4h) restores the margin. Cost is GCS: ~16 x 24G live at once.
# The eval box still uses HARD_NEG_EVAL_MILESTONE=10000 (a new milestone every ~82 min vs ~75 min
# of eval work), so it keeps up; the 2000-step checkpoints in between are for resume granularity,
# not eval. Evaluating every 2000 would be ~50 cycles x ~1.25h = far longer than the run itself.
# RUN_START_TIME (optional): pins the run-dir's timestamp instead of minting it at startup, so a
# launcher can compute <run_name>-<stamp> up front and point an eval box at the same identity
# without waiting for this to print it. multi-tpu-box-run.sh sets it for every box. Unset => the
# usual hydra-derived stamp.
HF_HUB_OFFLINE=1 GROUND_HF_PARQUET="${GROUND_HF_PARQUET:-$HOME/hf_parquet}" \
uv run train.py \
    ${RUN_START_TIME:+"+trainer.run_start_time=$RUN_START_TIME"} \
    model=qwen3_mem_embed \
    model.main_model.model_id="Qwen/Qwen3-4B" \
    model.memory.mem_top_k=64 \
    dataset=qa_hard_neg_think_sft4b \
    trainer=staged_telemetry \
    trainer.checkpoint_interval=2000 \
    trainer.max_to_keep=16 \
    trainer.log_interval=10 \
    trainer.eval_interval=1000000000 \
    'eval_set@trainer.evals=none' \
    +trainer.wandb_run_id=auto \
    +trainer.run_name="qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16"
