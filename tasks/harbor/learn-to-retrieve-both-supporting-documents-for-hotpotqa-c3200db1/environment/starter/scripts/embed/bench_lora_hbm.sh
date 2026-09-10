#!/bin/bash
# Does the LoRA stage-C configuration FIT IN HBM, and at what step time?
#
# Uses the bench harness (real jitted Trainer._train_step, synthetic batch of the true shapes) so
# it needs NO checkpoint restore and NO dataloader — which matters because the real resume path is
# currently blocked on an unrelated issue: the optax.masked optimizer changes the opt_state pytree
# (MaskedState/MaskedNode), and orbax cannot reconcile that with checkpoints written under the old
# chain. Memory footprint does not depend on restored VALUES, only on shapes/dtypes, so this
# measures the thing we actually want to know.
#
# Arms:
#   full   stage-C trainable set = the real 4B unfreeze  -> expected OOM on v6e (~32 GB/chip)
#   lora   stage-C trainable set = adapters only         -> the arm under test
# Both at batch 16 (dataset default). Run on EVERY worker of the slice (runbook §2.3).
#
#   TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers \
#   bash scripts/infrastructure/multi-tpu-box-run.sh \
#     tpu-v6e-slice-mig-1wjb=scripts/embed/bench_lora_hbm.sh \
#     tpu-v6e-slice-mig-1z9d=scripts/embed/bench_lora_hbm.sh
BENCH_LOG="${BENCH_LOG:-$HOME/bench_lora_hbm.log}"
GROUND=(
  model.memory.mem_top_k=128
  model.memory.mem_o_proj_zero_init=true
  'model.memory.mem_layers=[9,14,20,27]'
  trainer=staged_ground_mainunfreeze
)
# setup_optimizer() uses training_stages[0], so put the stage-C mask there.
FULL_MASK="trainer.training_stages.0.trainable_params=['.*mem_.*','.*embed_model.*','.*value_model.*','.*main_model.*']"
LORA_MASK="trainer.training_stages.0.trainable_params=['.*mem_.*','.*embed_model.*','.*value_model.*','.*main_model.*_a_proj','.*main_model.*_b_proj']"
{
  echo "############ host $(hostname)  $(date -u) ############"
  echo "############ ARM 1: LoRA adapters (model=qwen3_mem_embed_mainlora) ############"
  uv run python scripts/embed/bench_approx_topk.py --mode time --iters 10 --warmup 3 \
      --overrides model=qwen3_mem_embed_mainlora "${GROUND[@]}" "$LORA_MASK" || echo "ARM1_FAILED rc=$?"

  echo "############ ARM 2: full unfreeze (control — expected to OOM on v6e) ############"
  uv run python scripts/embed/bench_approx_topk.py --mode time --iters 10 --warmup 3 \
      --overrides "${GROUND[@]}" "$FULL_MASK" || echo "ARM2_FAILED rc=$?"

  echo "############ DONE  $(date -u) ############"
} 2>&1 | tee "$BENCH_LOG"
