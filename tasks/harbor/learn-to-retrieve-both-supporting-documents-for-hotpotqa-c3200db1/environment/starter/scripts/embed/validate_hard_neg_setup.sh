#!/bin/bash
# Pre-launch smoke test for the hard-neg (think) run + its eval box. Runs ON a box (needs the venv
# + .env for HF_USERNAME, which the doc-source configs interpolate). Checks, in order:
#   1. the corpus mem_pos_weight_mass math (standalone unit test)
#   2. the training config composes and carries the telemetry block + wandb id
#   3. the eval_set composes with the right corpus sizing / metrics / n
#   4. the deterministic wandb id is stable and matches what the eval box will derive
# Nothing here touches the TPU or launches a run.
#
# multi-vm-tpu-run.sh SOURCEs this from ~/memory-layers, so $0 is the shell, not this file —
# `dirname $0` would resolve to /home. cd to the repo explicitly, like the other box scripts do.
set -e
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . .env 2>/dev/null || true; set +a

echo "=============== 1. corpus mem_pos_weight_mass unit test ==============="
uv run python tests/test_corpus_pos_weight_mass.py

echo
echo "=============== 2. training config composes ==============="
# yaml_only: train.py/eval.py call setup_gcs_credentials() at MODULE level, which prints
# "GCloud credentials set!" to stdout before hydra emits the config — that banner makes the
# captured file unparseable YAML. Drop everything before the first top-level key.
yaml_only() { awk 'f || /^[a-zA-Z_][a-zA-Z_0-9]*:/ {f=1; print}'; }

uv run python train.py --cfg job --resolve \
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
    +trainer.run_name="qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16" \
    | yaml_only > /tmp/train_cfg.yaml
[ -s /tmp/train_cfg.yaml ] || { echo "ERROR: train --cfg job produced no YAML"; exit 1; }
echo "--- telemetry aux_losses (expect the weight-0 mem_* block) ---"
uv run python - <<'EOF'
from omegaconf import OmegaConf
c = OmegaConf.load("/tmp/train_cfg.yaml")
aux = c.trainer.aux_losses
mem = {k: dict(v) for k, v in aux.items() if k.startswith("mem_")}
for k, v in sorted(mem.items()):
    print(f"  {k}: {v}")
assert "mem_pos_weight_mass" in mem, "mem_pos_weight_mass MISSING from aux_losses"
bad = {k: v for k, v in mem.items() if v.get("weight", 0) != 0.0}
assert not bad, f"telemetry entries must be weight 0, got {bad}"
assert dict(aux["doc_access_loss"])["weight"] == 0.1, "doc_access_loss weight changed!"
print(f"  trainer.evals: {dict(c.trainer.evals) if c.trainer.get('evals') else '{} (in-loop eval OFF)'}")
print(f"  wandb_run_id: {c.trainer.wandb_run_id}   run_name: {c.trainer.run_name}")
print(f"  steps: {c.trainer.steps}  checkpoint_interval: {c.trainer.checkpoint_interval}"
      f"  max_to_keep: {c.trainer.max_to_keep}")
print(f"  stages: {len(c.trainer.training_stages)}")
# The eval box must reach a checkpoint before orbax rotates it away. Window = max_to_keep x
# interval; an eval cycle is ~1-1.5h (~11k steps at ~490ms). Assert real margin over one cycle.
window = int(c.trainer.max_to_keep) * int(c.trainer.checkpoint_interval)
print(f"  rotation window: {window} steps (~{window*0.49/3600:.1f}h at 490ms/step)")
assert window >= 20000, (
    f"rotation window {window} steps is too tight for a ~1-1.5h eval cycle — "
    f"raise trainer.max_to_keep or checkpoint_interval")
print("  OK")
EOF

echo
echo "=============== 3. eval_set composes ==============="
uv run python eval.py --cfg job --resolve \
    '~eval_set@evals=pretraining' \
    '+eval_set@evals=hard_neg_think_c512' \
    | yaml_only > /tmp/eval_cfg.yaml
[ -s /tmp/eval_cfg.yaml ] || { echo "ERROR: eval --cfg job produced no YAML"; exit 1; }
uv run python - <<'EOF'
from omegaconf import OmegaConf
c = OmegaConf.load("/tmp/eval_cfg.yaml")
seen = {}
for key, t in c.evals.items():
    e = t.eval
    dd = e.get("doc_dataset") or {}
    n = e.get("num_samples")
    if n is None and e.get("eval_steps") is not None:
        n = e.eval_steps * t.dataset.batch_size
    seen[key] = dict(type=e.type, n=n, metrics=sorted((e.get("metrics") or {}).keys()),
                     max_docs=dd.get("max_docs", "-"), target_docs=dd.get("target_docs", "-"))
    print(f"  {key}: {seen[key]}")
assert len(seen) == 4, f"expected 4 tasks, got {list(seen)}"
for k in ("gen_large_mem_msmarco", "gen_large_mem_hotpotqa", "gen_large_mem_musique"):
    s = seen[k]
    assert s["n"] == 128, f"{k}: n={s['n']} != 128"
    assert s["metrics"] == ["lexical_grounding", "llm_judge_accuracy"], f"{k}: metrics={s['metrics']}"
    # max_docs MUST stay null: data/documents.py stops the generator at max_docs, which would cap
    # the inject_query_gold scan and silently drop golds past the cap. target_docs does the trim.
    assert s["max_docs"] is None, f"{k}: max_docs={s['max_docs']} would truncate the gold scan"
    assert s["target_docs"] == 512, f"{k}: target_docs={s['target_docs']} != 512"
assert seen["nll_science_qa"]["n"] == 128, f"scienceQA n={seen['nll_science_qa']['n']} != 128"
assert seen["nll_science_qa"]["type"] == "nll"
print("  OK")
EOF

echo
echo "=============== 4. wandb id is per-RUN-DIR, not per-run-name ==============="
uv run python - <<'EOF'
from utils import wandb_run_id_from_run_dir
NAME = "qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16"
# The three real dirs that share this run_name in gs://memory-layers-training.
april = f"{NAME}-2026-04-19-20-17-51"
first = f"{NAME}-2026-07-16-15-14-02"
second = f"{NAME}-2026-07-16-16-52-25"
ids = {d: wandb_run_id_from_run_dir(d) for d in (april, first, second)}
for d, i in ids.items():
    print(f"  {d}\n    -> {i}")

# THE regression this design exists for: same run_name, different launches => DIFFERENT wandb
# runs. Keyed on the name these were all one id, so a relaunch appended to the earlier run and an
# eval box could log April's model into today's curve.
assert len(set(ids.values())) == 3, f"run-dirs collided on one id: {ids}"
assert wandb_run_id_from_run_dir(first) == wandb_run_id_from_run_dir(first), "not deterministic"
# A full gs:// path must resolve to the same id as the bare basename (the box may hold either).
assert wandb_run_id_from_run_dir(f"gs://memory-layers-training/{first}/") == ids[first]
for i in ids.values():
    assert len(i) <= 64 and all(c.isalnum() or c in "-_" for c in i), f"bad id charset/len: {i}"
    assert i.endswith(("-2026-04-19-20-17-51", "-2026-07-16-15-14-02", "-2026-07-16-16-52-25")), \
        f"timestamp must survive truncation (it is what makes the id unique): {i}"
# A run_name where a run-dir belongs must RAISE, not silently produce a plausible id.
try:
    wandb_run_id_from_run_dir(NAME)
    raise SystemExit("FAIL: bare run_name accepted as a run-dir")
except ValueError:
    print("  bare run_name correctly rejected")
print("  OK")
EOF

echo
echo "=============== 5. run_start_time pins the identity ==============="
uv run python - <<'EOF'
from omegaconf import OmegaConf
from utils import run_dir_name, wandb_run_id_from_run_dir
NAME = "qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16"
cfg = OmegaConf.create({"trainer": {"run_name": NAME, "run_start_time": "2026-07-16-18-00-00"}})
d = run_dir_name(cfg)
print(f"  pinned run_dir : {d}")
print(f"  wandb id       : {wandb_run_id_from_run_dir(d)}")
assert d == f"{NAME}-2026-07-16-18-00-00", d
# The whole point: a launcher can compute the dir+id with NO hydra context and no training box,
# which is what lets multi-tpu-box-run.sh start training and an eval box in parallel.
assert wandb_run_id_from_run_dir(d).endswith("-2026-07-16-18-00-00")
# Same pin => same identity (both boxes must agree); different pin => different run.
assert run_dir_name(cfg) == d, "not deterministic"
cfg2 = OmegaConf.create({"trainer": {"run_name": NAME, "run_start_time": "2026-07-16-19-00-00"}})
assert wandb_run_id_from_run_dir(run_dir_name(cfg2)) != wandb_run_id_from_run_dir(d)
# A malformed pin must fail HERE, not later as an unparseable dir / a dir that can never exist.
for bad in ["2026-07-16", "16-07-2026-18-00-00", "2026-07-16 18:00:00", "nonsense"]:
    try:
        run_dir_name(OmegaConf.create({"trainer": {"run_name": NAME, "run_start_time": bad}}))
        raise SystemExit(f"FAIL: accepted malformed run_start_time {bad!r}")
    except ValueError:
        pass
print("  malformed run_start_time correctly rejected")
print("  OK")
EOF

echo
echo "=============== ALL CHECKS PASSED ==============="
