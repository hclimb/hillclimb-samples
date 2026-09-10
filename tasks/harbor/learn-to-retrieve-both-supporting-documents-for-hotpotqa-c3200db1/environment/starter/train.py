import hydra
import jax
import wandb
from omegaconf import DictConfig, OmegaConf
from data import get_dataset
from models import get_model
from utils import (setup_checkpointing, get_run_name, setup_optimizer, setup_gcs_credentials,
                   resolve_wandb_run_id, run_dir_name, init_jax_distributed,
                   promote_trainable_to_fp32)
from trainer.trainer import Trainer
from trainer.grad_accum_trainer import GradAccumTrainer

from dotenv import load_dotenv
load_dotenv()
setup_gcs_credentials()


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg) + "\n")

    init_jax_distributed()
    
    # Load model
    print(f"Initializing model...")
    model = get_model(cfg.model, cfg.trainer.tp_devices)

    # fp32 master weights for every trainable leaf across ALL stages. Runs BEFORE
    # setup_optimizer so optimizer.init() sees fp32 params and allocates fp32 mu/nu.
    # See utils.py::promote_trainable_to_fp32 for the full rationale (Bug 2 fix).
    model.weights = promote_trainable_to_fp32(model.weights, cfg.trainer.get("training_stages"))

    # Setup Optimizer
    print(f"Setting up optimizer...")
    optimizer, model, lr_schedule = setup_optimizer(cfg, model)

    # Setup Dataset
    print(f"Loading dataset '{cfg.dataset.name}'...")
    data = get_dataset(cfg.dataset, model)

    # Setup Checkpointing
    print(f"Initializing checkpoint manager...")
    checkpoint_manager, resume_step, resume_from_dir = setup_checkpointing(cfg)

    # Initialize wandb
    if cfg.trainer.get("use_wandb", False) and jax.process_index() == 0:
        run_name = get_run_name(cfg)
        # The run-dir is this launch's identity: the SAME string names the GCS folder we
        # checkpoint into and (via wandb_run_id_from_run_dir) this wandb run, so an eval box
        # pointed at the folder can derive the id and cannot mix runs. run_dir_name() is pure,
        # so asking for it here doesn't re-run setup_checkpointing's GCS side effects.
        run_dir = run_dir_name(cfg)
        run_id = resolve_wandb_run_id(cfg, run_dir)
        init_kwargs = {}
        if run_id is not None:
            # Deterministic id => the eval box can attach to THIS run and log eval/* next to
            # train/*. shared mode is what makes two processes writing one run safe
            # (wandb>=0.19); we are the primary writer, so our finish() is the one that ends the
            # run. NOTE a preemption-resume mints a new run-dir => new id => a SECOND wandb run;
            # pass an explicit trainer.wandb_run_id to stay on one curve.
            # See wiki/evaluation/eval-boxes.md.
            init_kwargs = dict(
                id=run_id,
                resume="allow",
                settings=wandb.Settings(mode="shared", x_primary=True, x_label="train"),
            )
            print(f"W&B shared-mode primary writer, run id: {run_id}  (run_dir: {run_dir})")
        print(f"Starting W&B run: {run_name}")
        wandb.init(
            project=cfg.trainer.get("wandb_project", "memory-layers"),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            **init_kwargs,
        )
        # Shared mode silently ignores wandb.log(..., step=N) — the SDK auto-
        # increments a per-process _step instead. Define an explicit step metric so
        # every train/* series uses our training step as the x-axis, and resumes
        # (which set step to the loaded checkpoint's step) plot at the right x.
        # Applies to any run_id; no-op when run_id is None (non-shared mode).
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("stage", step_metric="train/step")
        wandb.define_metric("stage_ce_weight", step_metric="train/step")
        wandb.define_metric("stage_trainable_params", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        # Weight-monitor metrics use the same wandb.log call as train/* but need
        # their own step_metric declaration or they fall through to wandb's
        # per-process _step counter. On resume, _step keeps climbing from where
        # the pre-preempt process left it — so weight metrics plot at wrong x
        # values (empirically observed: losses at step 60k, weights at step 73k
        # after a preempt-resume from ckpt-60000). Pin them to train/step too.
        wandb.define_metric("weight/*", step_metric="train/step")

    # Create trainer and start training
    print(f"Starting training for {cfg.trainer.steps} steps...")
    TrainerClass = GradAccumTrainer if cfg.trainer.get("grad_accum_steps", 1) > 1 else Trainer
    trainer = TrainerClass(cfg, model, data, optimizer, checkpoint_manager, resume_step=resume_step, resume_from_dir=resume_from_dir, lr_schedule=lr_schedule)
    trainer.train()
    print("Training successfully completed.")


if __name__ == "__main__":
    main()
