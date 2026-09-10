import json
import hydra
import wandb
from omegaconf import DictConfig
from evals.shared import (
    resolve_train_cfg, apply_checkpoint_model_cfg, init_wandb, run_eval_worker, run_metrics_pipeline,
)

@hydra.main(config_path="configs", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig
    from utils import setup_gcs_credentials
    # MUST precede resolve_train_cfg: it reads the checkpoint's .hydra/config.yaml from GCS, which
    # needs the user's adc.json (GCS_USER_EMAIL). Without this, the read falls back to the box's
    # compute service account and 403s on memory-layers-training. (rag_eval.py already does this.)
    setup_gcs_credentials()
    train_cfg, ckpt_dir, step = resolve_train_cfg(cfg)
    # Checkpoint's saved model config is authoritative for architecture (else the eval default
    # silently clobbers it — e.g. mem_layers=[9,14,20,27] -> [14]). Explicit model.* overrides win.
    cfg = apply_checkpoint_model_cfg(cfg, train_cfg, HydraConfig.get().overrides.task)
    init_wandb(cfg, train_cfg)
    out_dir = HydraConfig.get().runtime.output_dir
    manifest = run_eval_worker(cfg, train_cfg, ckpt_dir, step, out_dir)
    all_metrics = run_metrics_pipeline(manifest, cfg, ckpt_dir, step, out_dir)
    print("All Evaluation Results:")
    print(json.dumps(all_metrics, indent=2))
    if wandb.run: wandb.log(all_metrics)

if __name__ == "__main__":
    main()
