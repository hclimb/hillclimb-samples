#!/bin/bash
# Runs ON a box. Composes a training config and prints the values that actually matter, WITHOUT
# starting a run — so a typo'd override or a missing config file fails in seconds instead of
# after a model load and a JAX compile.
#
# Pass Hydra overrides as CHECK_OVERRIDES, CARET-separated (spaces also work when run directly;
# a separator is needed because multi-vm-tpu-run.sh's RUN_ENV splits on whitespace):
#   CHECK_OVERRIDES="model=qwen3_mem_embed^dataset=musique_sft^trainer=midtraining_telemetry" \
#     bash scripts/misc/check_train_config.sh
#
# Caret, having ruled out the obvious candidates: comma shatters Hydra list overrides
# (model.memory.mem_layers=[9,14,20,27] -> four bogus overrides), semicolon is a shell command
# separator and breaks when RUN_ENV is exported on the box, and colon appears in gs:// paths.
#
# NOTE most trainer configs inherit `standard`, whose defaults reference a nonexistent
# `eval_set/standard` — so nearly every compose needs `eval_set@trainer.evals=none` (or a real
# eval_set) or Hydra dies before printing anything. Same override train.py runs need.
#
# Prints dataset shape, trainer stages, the trainable-param regexes per stage, aux-loss weights,
# and derived quantities (memory-bank slots per batch, epochs over a given row count).
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
export PATH="$HOME/.local/bin:$PATH"
set -a; . .env 2>/dev/null || . "$HOME/.env" 2>/dev/null || true; set +a

DATASET_ROWS="${DATASET_ROWS:-0}" CHECK_OVERRIDES="${CHECK_OVERRIDES:-}" \
uv run --no-sync python - <<'PY'
import os, sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

raw = os.environ.get("CHECK_OVERRIDES", "").replace("^", " ")
overrides = [o for o in raw.split() if o]
rows = int(os.environ.get("DATASET_ROWS", "0"))
print("overrides:", " ".join(overrides) or "(none)")
print()
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
    cfg = compose(config_name="train", overrides=overrides)

d, t = cfg.dataset, cfg.trainer
print("DATASET")
print(f"  name                  : {d.name}")
print(f"  seq_len / doc_chunk   : {d.seq_len} / {d.doc_chunk_seq_len}")
print(f"  num_chunks_per_doc    : {d.num_chunks_per_doc}")
print(f"  batch_size            : {d.batch_size}")
print(f"  min_doc_length        : {d.get('min_doc_length', 64)}  (default 64)")
print(f"  provide_docs/mask_pfx : {d.get('provide_docs')} / {d.get('mask_prefix')}")
print(f"  chat_template/think   : {d.get('chat_template')} / {d.get('force_thinking')}")
print(f"  split                 : {d.get('split')}")
for k, s in (d.get("sources") or {}).items():
    print(f"  source[{k}]")
    print(f"      hf_name       : {s.get('hf_name')}")
    print(f"      field_map     : {OmegaConf.to_container(s.get('field_map')) if s.get('field_map') else None}")
    print(f"      think_field   : {s.get('think_field')}")
    print(f"      doc_separator : {s.get('doc_separator')}")
    print(f"      neg_threshold : {s.get('neg_score_threshold')}  min_neg={s.get('min_neg_docs', 0)}")

print("\nMODEL")
m = cfg.model
print(f"  main / embed          : {m.main_model.model_id} / {m.embed_model.model_id}")
print(f"  mem_layers/size/heads : {list(m.memory.mem_layers)} / {m.memory.mem_size} / {m.memory.mem_num_heads}")
print(f"  mem_top_k / approx    : {m.memory.mem_top_k} / {m.memory.mem_approx_topk}")

print("\nTRAINER")
print(f"  steps / lr / wd       : {t.steps} / {t.learning_rate} / {t.weight_decay}")
print(f"  ce_weight (base)      : {t.ce_weight}")
print(f"  resume_from           : {t.get('resume_from')}")
warm = str(t.get('resume_from') or '').rstrip('/').rsplit('/', 1)
print(f"     -> mode            : {'WARM START (weights only, step 0)' if len(warm)==2 and warm[1].isdigit() else 'FULL RESUME (weights+optimizer+step)' if t.get('resume_from') else 'from scratch'}")
print(f"  ckpt_interval/keep    : {t.checkpoint_interval} / {t.max_to_keep}  -> {t.checkpoint_interval*t.max_to_keep} step window")
stages = t.get("training_stages")
if stages:
    print(f"  stages                : {len(stages)}")
    for i, s in enumerate(stages):
        print(f"    [{i}] max_step={s.get('max_step')} ce={s.get('ce_weight')} "
              f"lr_sched={s.get('lr_schedule','const')} warmup={s.get('warmup_frac')}")
        print(f"         trainable={list(s.get('trainable_params', []))}")
else:
    print(f"  stages                : none (single-phase); trainable={list(cfg.model.trainable_params)}")
nz = {k: v.weight for k, v in t.aux_losses.items() if v.get("enabled") and v.get("weight")}
z  = [k for k, v in t.aux_losses.items() if v.get("enabled") and not v.get("weight")]
print(f"  aux losses (nonzero)  : {nz}")
print(f"  telemetry (weight 0)  : {len(z)} entries")

print("\nDERIVED")
print(f"  memory bank slots/batch: {d.batch_size} x {d.num_chunks_per_doc} x {d.doc_chunk_seq_len} = "
      f"{d.batch_size*d.num_chunks_per_doc*d.doc_chunk_seq_len:,}")
print(f"  max pos_doc tokens     : {d.doc_chunk_seq_len*d.num_chunks_per_doc:,}")
print(f"  samples seen           : {t.steps} x {d.batch_size} = {t.steps*d.batch_size:,}")
if rows:
    print(f"  epochs over {rows:,} rows : {t.steps*d.batch_size/rows:.2f}")
PY
