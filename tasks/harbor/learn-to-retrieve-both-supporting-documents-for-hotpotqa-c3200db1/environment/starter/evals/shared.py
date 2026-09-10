import os
import json
import sys
import wandb
from omegaconf import DictConfig, OmegaConf

def resolve_train_cfg(cfg):
    checkpoint_dir = cfg.get("checkpoint_dir")
    if not checkpoint_dir:
        return cfg, None, None
    
    parts = str(checkpoint_dir).rstrip("/").split("/")
    step = int(parts[-1])
    run_dir = "/".join(parts[:-2])
    train_config_path = f"{run_dir}/.hydra/config.yaml"
    
    if str(checkpoint_dir).startswith("gs://"):
        # NOTE: this reads GCS, so GOOGLE_APPLICATION_CREDENTIALS must already point at the user's
        # adc.json (via utils.setup_gcs_credentials()) — otherwise the read falls back to the box's
        # compute service account, which 403s on memory-layers-training. eval.py/rag_eval.py call
        # setup_gcs_credentials() before this. See wiki/evaluation/two-process-design.md.
        import gcsfs
        with gcsfs.GCSFileSystem().open(train_config_path[5:]) as f:
            train_cfg = OmegaConf.load(f)
    else:
        train_cfg = OmegaConf.load(train_config_path)
    return train_cfg, str(checkpoint_dir), step


# Architecture keys whose value changes which weights exist / their shapes. If the eval's default
# model config differs from the checkpoint on any of these, silently using the eval default builds
# the WRONG network — at best a shape error on load, at worst (mem_layers, matching dims) a silent
# partial restore that drops the extra layers' trained weights. We warn on each.
_ARCH_KEYS = (
    "memory.mem_layers", "memory.mem_num_heads", "memory.mem_k_dim", "memory.mem_v_dim",
    "memory.mem_size", "memory.mem_top_k", "memory.mem_use_product_keys", "memory.mem_placement",
    "main_model.model_id", "embed_model.model_id",
)


def apply_checkpoint_model_cfg(cfg, train_cfg, cli_overrides=None):
    """Make the CHECKPOINT'S saved model config authoritative when evaluating a checkpoint.

    Why: configs/eval.yaml always composes a full default `model` (qwen3_mem_embed), so cfg.model
    is never empty. The eval worker's `OmegaConf.merge(train_cfg.model, cfg.model)` then lets that
    default OVERRIDE the trained architecture for every key where they differ — e.g. a checkpoint
    trained with mem_layers=[9,14,20,27] gets silently rebuilt as mem_layers=[14], and partial_restore
    drops layers 9/20/27's weights with no error. This makes the trained config the base instead.

    Precedence, when a checkpoint is present:
      * explicit `model=<group>` on the command line  -> caller deliberately swapped the whole model;
        respect cfg.model unchanged (e.g. a hand-matched config; the old workaround still works).
      * explicit `model.<path>=<val>` (or +/~ variants) -> a knob tweak LAYERED on the trained arch.
      * everything else                                 -> comes from the checkpoint, not the default.

    No checkpoint (train_cfg has no model) -> cfg.model is the only source of truth; return unchanged.
    cli_overrides: the Hydra task-override strings (HydraConfig.get().overrides.task).
    """
    if train_cfg is None or train_cfg.get("model") is None:
        return cfg
    overrides = list(cli_overrides or [])
    keys = [o.lstrip("+~").split("=", 1)[0] for o in overrides]
    if "model" in keys:  # explicit whole-model group swap -> honor it as-is
        return cfg

    default_model = cfg.get("model")
    base = OmegaConf.create(OmegaConf.to_container(train_cfg.model, resolve=True))

    # Loud diff so a train/eval architecture mismatch can never again be silent.
    if default_model is not None:
        for ak in _ARCH_KEYS:
            d = OmegaConf.select(default_model, ak)
            t = OmegaConf.select(base, ak)
            if d is not None and t is not None and d != t:
                print(f"[eval] checkpoint-authoritative: {ak}={t} from checkpoint "
                      f"(eval default {d} would have clobbered it)", flush=True)

    applied = []
    for o in overrides:
        raw = o.lstrip("+~")
        key = raw.split("=", 1)[0]
        if not key.startswith("model."):
            continue
        subkey = key[len("model."):]
        if o.startswith("~"):
            OmegaConf.update(base, subkey, None)
            applied.append(f"~{subkey}")
            continue
        val_str = raw.split("=", 1)[1] if "=" in raw else "null"
        parsed_val = OmegaConf.create(f"v: {val_str}").v   # YAML-parse so 64->int, [1,2]->list, etc.
        OmegaConf.update(base, subkey, parsed_val, force_add=o.startswith("+"))
        applied.append(f"{subkey}={parsed_val}")
    if applied:
        print(f"[eval] explicit model overrides applied on top of checkpoint config: {applied}", flush=True)

    OmegaConf.set_struct(cfg, False)
    cfg.model = base
    return cfg

def init_wandb(cfg, train_cfg, prefix="eval"):
    if not cfg.get("use_wandb", True): return
    model_id = (cfg.get("model") or train_cfg.get("model", {})).get("main_model", {}).get("model_id", "unknown")
    model_name = model_id.replace("/", "-")
    eval_keys = "_".join(cfg.evals.keys()) if cfg.get("evals") else cfg.dataset.get("name", "unknown")
    run_name = f"{prefix}_{model_name}_{eval_keys}"
    wandb.init(project=cfg.get("wandb_project", "memory-layers-eval"), 
               config=OmegaConf.to_container(cfg, resolve=True), name=run_name)

def run_eval_worker(cfg, train_cfg, checkpoint_dir, step, output_dir):
    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = os.path.join(tmpdir, "eval_cfg.json")
        train_path = os.path.join(tmpdir, "train_cfg.json")
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(eval_path, "w") as f: json.dump(OmegaConf.to_container(cfg, resolve=True), f)
        with open(train_path, "w") as f: json.dump(OmegaConf.to_container(train_cfg, resolve=True), f)
        
        cmd = [sys.executable, "-m", "evals.eval_worker", "--eval-cfg", eval_path, "--train-cfg", train_path,
               "--output-dir", output_dir, "--manifest-out", manifest_path]
        if checkpoint_dir: cmd += ["--checkpoint-dir", checkpoint_dir]
        if step is not None: cmd += ["--step", str(step)]
        
        print(f"[{prefix_name(cfg)}] Spawning JAX worker...")
        subprocess.run(cmd, env=os.environ.copy(), check=True)
        # On a multi-host mesh only the JAX rank-0 worker writes a manifest, and rank order is
        # not host order (runbook §2.3) — the other hosts' parents get an empty manifest and
        # their metrics pipeline no-ops (it is already gated on result files existing locally).
        if not os.path.exists(manifest_path):
            print(f"[rag_eval] no manifest on this host (non-rank-0 worker of a multi-host slice)")
            return {}
        with open(manifest_path) as f: return json.load(f)

def prefix_name(cfg): return "rag_eval" if "rag" in cfg or "evals" in cfg else "eval"

def run_metrics_pipeline(manifest, cfg, checkpoint_dir, step, output_dir):
    from evals.gen_base_model import run_generation
    from evals.metrics import run_metrics
    all_metrics = {}
    for eval_key, entry in manifest.items():
        if entry.get("deferred_type") == "generation_base":
            run_generation(eval_cfg=entry["deferred_eval_cfg"], dataset_cfg=entry["deferred_dataset_cfg"], output_file=entry["output_file"])
        
        inf_metrics = entry.get("inference_metrics") or {}
        for k, v in inf_metrics.items(): all_metrics[f"{eval_key}/{k}"] = v
        
        out_file = entry.get("output_file")
        m_cfg = entry.get("metrics_cfg")
        if out_file and m_cfg and os.path.exists(out_file):
            with open(out_file) as f: data = json.load(f)
            annotated, scores = run_metrics(data.get("samples", []), m_cfg)
            data["samples"] = annotated
            data["metrics"].update(scores)
            with open(out_file, "w") as f: json.dump(data, f, indent=2)
            for k, v in scores.items(): all_metrics[f"{eval_key}/{k}"] = v
    return all_metrics
