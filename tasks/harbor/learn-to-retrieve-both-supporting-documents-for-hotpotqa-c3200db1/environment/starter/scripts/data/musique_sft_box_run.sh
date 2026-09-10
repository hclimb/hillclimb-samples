#!/bin/bash
# Runs ON a TPU box. Builds the MuSiQue SFT finetuning set (both stages).
#
#   PHASE=smoke (default) — tests, then stage 1, then stage 2 on ONE shard, so the
#                           yield and hop_grounding histogram can be inspected before
#                           committing the TPU to all ten.
#   PHASE=full            — stage 2 over every remaining shard. Resumable: it lists the
#                           output repo and skips shards already uploaded.
#
# Launch (note TRANSPORT=gce — tpu-v6e-vm is a Compute Engine VM with an attached TPU,
# not a Cloud TPU node, so the tpu-vm ssh path can't see it):
#   TPU_NAME=tpu-v6e-vm ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce \
#   RUN_SCRIPT_PATH=scripts/data/musique_sft_box_run.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
set -euo pipefail
set -a; . $HOME/.env; set +a
export PATH=$HOME/.local/bin:$PATH
cd $HOME/memory-layers
. .venv/bin/activate

PHASE="${PHASE:-smoke}"
echo "[musique] phase=$PHASE  user=${HF_USERNAME:?HF_USERNAME not set in ~/.env}"
df -h / | tail -1

# vllm-tpu 0.12.0's HTTP server breaks on the fastapi/starlette versions it resolves to
# (_IncludedRouter has no attribute 'path') — see the pyproject.toml note. These pins are
# deliberately NOT in pyproject (they conflict with vllm-tpu's starlette>=0.49.1 and would
# break `uv run`), so install them here; datagen/vllm_inference.py launches vllm with
# --no-sync so they don't get reverted.
echo "[musique] pinning vllm HTTP server deps..."
uv pip install --quiet fastapi==0.115.6 starlette==0.41.3 prometheus-fastapi-instrumentator==7.0.0

# A vllm left over from a previous attempt still holds the TPU chips, and the next launch
# dies on device-busy well after the model has downloaded.
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Token budgets, overridable so a training seq_len change is an A/B rather than a source edit.
# THINK_TOKENS must match the training seq_len: seq_len 512 -> ~400, seq_len 1024 -> ~950
# (the chat-template prefix costs ~33 tok and the <think> wrapper ~8).
THINK_TOKENS="${THINK_TOKENS:-400}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
MODEL_LEN="${MODEL_LEN:-8192}"
BUDGETS="--max-think-ans-tokens $THINK_TOKENS --max-output-tokens $OUTPUT_TOKENS --max-model-len $MODEL_LEN"
echo "[musique] budgets: think+ans<=$THINK_TOKENS output<=$OUTPUT_TOKENS model_len=$MODEL_LEN"

if [ "$PHASE" = "test" ]; then
  echo "[musique] ─── tests only ───"
  uv run --no-sync python tests/test_musique_sft.py
elif [ "$PHASE" = "all" ]; then
  # tests -> stage 1 (shuffled shards) -> stage 2 over every shard, one session.
  echo "[musique] ─── tests ───"
  uv run --no-sync python tests/test_musique_sft.py
  echo "[musique] ─── stage 1: MuSiQue -> ${HF_USERNAME}/musique-sft-base ───"
  uv run --no-sync python datagen/musique/prepare_musique_sft_base.py
  echo "[musique] ─── stage 2 (all shards) ───"
  uv run --no-sync python datagen/musique/generate_musique_sft.py $BUDGETS
elif [ "$PHASE" = "probe" ]; then
  # Generate ONE not-yet-done shard to A/B a change against a previous shard's accepted%.
  # Shards already in the output repo are skipped, so this picks the next one.
  echo "[musique] ─── stage 2 (probe: next single shard) ───"
  uv run --no-sync python datagen/musique/generate_musique_sft.py --limit 1 $BUDGETS
elif [ "$PHASE" = "smoke" ]; then
  echo "[musique] ─── tests ───"
  uv run --no-sync python tests/test_musique_sft.py

  echo "[musique] ─── stage 1: MuSiQue -> ${HF_USERNAME}/musique-sft-base ───"
  uv run --no-sync python datagen/musique/prepare_musique_sft_base.py

  echo "[musique] ─── stage 2 (ONE shard, for yield inspection) ───"
  uv run --no-sync python datagen/musique/generate_musique_sft.py --limit 1 $BUDGETS
  echo "[musique] smoke done — check accepted count and the hop_grounding histogram above."
else
  echo "[musique] ─── stage 2 (all remaining shards) ───"
  uv run --no-sync python datagen/musique/generate_musique_sft.py $BUDGETS
fi
