#!/bin/bash
# Runs ON a box. Scores ONE checkpoint on the MuSiQue @512-doc corpus eval, judge included.
#
# Unlike scripts/misc/hard_neg_eval_box_run.sh this is a ONE-SHOT eval of a specific checkpoint,
# not a watcher over a run-dir, and it is fine to run on the TRAINING box once training has
# exited. The "use a separate eval box" rule exists because the judge's vLLM needs the TPU that a
# live training run is holding — with training finished, the chips are free.
#
#   CKPT=<gs://…/qwen3_mem_embed/<step>> bash scripts/embed/eval_musique_sft_midtrain.sh
#
# ⚠️ MuSiQue is IN-DOMAIN for the midtrained checkpoint — this dataset is what it was trained on.
# The number here measures fit, not generalisation, and is NOT comparable to the same metric on
# the pre-midtraining checkpoint. Run the msmarco/hotpotqa c512 tasks for the honest read
# (EVAL_TASK=gen_large_mem_msmarco_c512 / gen_large_mem_hotpotqa_c512).
set -uo pipefail
set -a; . "$HOME/.env"; set +a
export PATH="$HOME/.local/bin:$PATH"
cd "${REPO_DIR:-$HOME/memory-layers}"

# resolve_train_cfg reads the checkpoint's .hydra from GCS BEFORE setup_gcs_credentials() runs,
# so the ADC has to be exported up front or it 404s long before training config is loaded.
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL:?GCS_USER_EMAIL not set in ~/.env}/adc.json"
export GCLOUD_PROJECT="${GCS_BUCKET_PROJECT}"

CKPT="${CKPT:?set CKPT to a gs://…/qwen3_mem_embed/<step> path}"
EVAL_TASK="${EVAL_TASK:-gen_large_mem_musique_c512}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"

# Generation budget. The c512 task configs ship max_new_tokens=512, which fits the hard-neg model
# (trained with think+answer <= 400 at seq_len 512). THIS checkpoint was midtrained at seq_len
# 1024 with a 950-token think+answer budget, and at 512 the model runs out of tokens mid-<think>:
# measured on the first eval, 71% of generations never emitted a closing </think>, and because
# the answer follows that tag, the same 71% had an EMPTY generated_answer. That dragged
# llm_judge_accuracy to 0.25 while the 29% that completed scored 0.43 — i.e. the cap was
# measuring the budget, not the model. Keep this >= the training think+answer budget.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"

# Results are keyed by <task>[-<RUN_TAG>], not by task alone: the GCS filename and the wandb
# artifact name both derive from it, so two runs of the same task at the same step would silently
# overwrite each other. Set RUN_TAG for a repeat (e.g. RUN_TAG=rerun2 to measure run-to-run
# spread) or for a variant (RUN_TAG=exacttopk).
RUN_TAG="${RUN_TAG:-}"

# Decoding temperature. 0.0 => inference.py::_sample returns argmax immediately and top_k/top_p
# are NEVER reached (dead settings). Any value > 0 activates the task's top_k=20 / top_p=0.8 and
# makes the run stochastic — so a temperature run is not comparable to a greedy one, and repeats
# of it will differ from each other. Worth trying because the SFT data itself was generated at
# temperature 0.6, so sampling there matches the training distribution.
TEMPERATURE="${TEMPERATURE:-0.0}"

# Judge sizing. The metric defaults (evals/metrics/llm_judge.py) are Qwen3-8B at
# tensor_parallel_size=8 — that is a v6e-8 assumption and will NOT start on a 4-chip box, where
# TP must divide the chip count. Qwen3-4B at TP=4 is what the other judged task configs
# (gen_large_mem_science_qa.yaml, …_hotpotqa_distractor.yaml) already use, and it is already on
# disk from precache_hf.sh.
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-4B}"
JUDGE_TP="${JUDGE_TP:-$(ls /dev/vfio 2>/dev/null | grep -c '^[0-9]*$')}"
JUDGE_TP="${JUDGE_TP:-4}"

# NO HF_HUB_OFFLINE here, deliberately: the corpus eval streams msa-musique-{qa,docs}-with-ids
# from the Hub. This is a single eval, not a 16-worker training loop, so it does not approach the
# rate limit that forces training offline (wiki/data/hf-rate-limits.md).
#
# vllm-tpu 0.12.0's HTTP judge needs these pins or it dies on
# "_IncludedRouter has no attribute 'path'" — see the pyproject.toml note. evals/vllm.py launches
# with `uv run --no-sync` so they survive.
echo "[eval] pinning vllm HTTP server deps..."
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

# A leftover worker from a previous run still holds /dev/vfio/*; evals/vllm.py frees them itself
# (free_tpu=True) but the JAX eval worker runs first and would hit device-busy before that.
pkill -f "[v]llm serve" 2>/dev/null || true
sudo fuser -k -9 /dev/vfio/[0-9]* 2>/dev/null || true
sleep 3

echo "[eval] ckpt=$CKPT"
echo "[eval] task=$EVAL_TASK  n=$NUM_SAMPLES  max_new_tokens=$MAX_NEW_TOKENS  temp=$TEMPERATURE  judge=$JUDGE_MODEL tp=$JUDGE_TP"

uv run --no-sync eval.py \
    checkpoint_dir="$CKPT" \
    '~eval_set@evals=pretraining' \
    "+eval/tasks@evals.musique=$EVAL_TASK" \
    "evals.musique.eval.num_samples=$NUM_SAMPLES" \
    "evals.musique.eval.max_new_tokens=$MAX_NEW_TOKENS" \
    "evals.musique.eval.temperature=$TEMPERATURE" \
    "+evals.musique.eval.metrics.llm_judge_accuracy.model_id=$JUDGE_MODEL" \
    "+evals.musique.eval.metrics.llm_judge_accuracy.tensor_parallel_size=$JUDGE_TP"
rc=$?

# eval.py leaves the results JSON on LOCAL disk and its own wandb run carries the config but no
# metrics: llm_judge_accuracy/lexical_grounding are computed by evals/shared.py::
# run_metrics_pipeline in the PARENT process, after the JAX worker (which owns that wandb run)
# has already exited. scripts/misc/hard_neg_eval_box.py normally closes the loop; a one-shot eval
# bypasses it, so do the same two things here — otherwise the only copy of the numbers dies with
# the box, which for a flex VM is a hard deadline, not a hypothetical.
[ $rc -eq 0 ] && CKPT="$CKPT" EVAL_TASK="${EVAL_TASK}${RUN_TAG:+-$RUN_TAG}" \
  uv run --no-sync python scripts/misc/log_eval_to_wandb.py
exit $rc
