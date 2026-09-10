"""
JAX inference worker — runs all evaluators and saves raw samples to disk, then exits.

Called by eval.py as a subprocess. The parent process (no JAX) runs metrics
(e.g. vLLM judge) after this process exits and releases the TPU.

Usage:
    python -m evals.eval_worker \
        --eval-cfg /tmp/eval_cfg.json \
        --train-cfg /tmp/train_cfg.json \
        --output-dir /path/to/hydra/output \
        --manifest-out /tmp/manifest.json \
        [--checkpoint-dir /path/to/ckpt/step] \
        [--step 100]
"""

import argparse
import gc
import json
import os
from utils import setup_gcs_credentials
from dotenv import load_dotenv


def _output_file_path(output_dir, step, eval_key, filename):
    if step is not None:
        return os.path.join(output_dir, "eval_results", f"step_{step}", eval_key, filename)
    return os.path.join(output_dir, "eval_results", eval_key, filename)


def _collect_deferred_evals(cfg, args):
    """
    Pre-JAX pass: pull out any generation_base evals into deferred manifest entries.
    Returns (deferred_manifest, remaining_evals) where remaining_evals is the subset
    of cfg.evals that still need JAX.
    """
    from omegaconf import OmegaConf

    deferred = {}
    remaining = {}

    evals = cfg.get("evals", {})
    for eval_key, eval_cfg in evals.items():
        actual_eval_cfg = eval_cfg.get("eval", eval_cfg)
        eval_type = actual_eval_cfg.get("type", actual_eval_cfg.get("name", ""))

        if eval_type == "generation_base":
            eval_cfg_dict = OmegaConf.to_container(actual_eval_cfg, resolve=True)
            metrics_cfg = eval_cfg_dict.pop("metrics", None)
            dataset_cfg_dict = OmegaConf.to_container(eval_cfg.dataset, resolve=True) if eval_cfg.get("dataset") else {}
            output_file = None
            if actual_eval_cfg.get("output_file"):
                output_file = _output_file_path(args.output_dir, args.step, eval_key, actual_eval_cfg.output_file)
            deferred[eval_key] = {
                "output_file": output_file,
                "metrics_cfg": metrics_cfg,
                "inference_metrics": None,
                "deferred_type": "generation_base",
                "deferred_eval_cfg": eval_cfg_dict,
                "deferred_dataset_cfg": dataset_cfg_dict,
            }
            print(f"[eval_worker] Deferring eval '{eval_key}' (generation_base) to parent process.")
        else:
            remaining[eval_key] = eval_cfg

    return deferred, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-cfg", required=True)
    parser.add_argument("--train-cfg", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    os.environ["EVAL_OUTPUT_DIR"] = args.output_dir
    load_dotenv()
    setup_gcs_credentials()

    # Load configs before JAX so we can route generation_base evals early
    from omegaconf import OmegaConf
    with open(args.eval_cfg) as f:
        cfg = OmegaConf.create(json.load(f))
    with open(args.train_cfg) as f:
        train_cfg = OmegaConf.create(json.load(f))

    manifest, remaining_evals = _collect_deferred_evals(cfg, args)

    # Only initialize JAX if there are evals that actually need it
    if not remaining_evals and not cfg.get("eval"):
        print("[eval_worker] All evals deferred — skipping JAX init.")
        os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[eval_worker] manifest written to {args.manifest_out}")
        return

    import jax
    from utils import init_jax_distributed; init_jax_distributed()

    import orbax.checkpoint as ocp
    from models import get_model
    from data import get_dataset
    from evals import get_evaluator
    from utils import load_inference_checkpoint

    # Sort evals by tp_devices so all evals with the same tp_devices run together,
    # minimising model reloads to at most one per unique tp_devices value.
    def _eval_tp_devices(item):
        _, eval_cfg = item
        return eval_cfg.get("tp_devices", cfg.tp_devices)

    sorted_evals = sorted(remaining_evals.items(), key=_eval_tp_devices, reverse=True)

    # cfg.model is already CHECKPOINT-AUTHORITATIVE here: eval.py/rag_eval.py run
    # apply_checkpoint_model_cfg() first, which rebuilds cfg.model from the checkpoint's saved config
    # (plus explicit model.* CLI overrides) whenever a checkpoint is loaded. So this merge is a no-op
    # for those paths (cfg.model already ⊇ train_cfg.model). It stays only as a safety net for any
    # caller that hands the worker a raw eval-default cfg.model without resolving it first.
    effective_model_cfg = OmegaConf.merge(train_cfg.model, cfg.model) if cfg.get("model") else train_cfg.model

    current_tp_devices = _eval_tp_devices(sorted_evals[0]) if sorted_evals else cfg.tp_devices
    model = get_model(effective_model_cfg, current_tp_devices)

    checkpoint_manager = None
    checkpoint_step = args.step
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        if not checkpoint_dir.startswith("gs://"):
            checkpoint_dir = os.path.abspath(checkpoint_dir)
        parts = checkpoint_dir.rstrip("/").split("/")
        checkpoint_step = int(parts[-1])
        model_ckpt_dir = "/".join(parts[:-1])
        options = ocp.CheckpointManagerOptions(max_to_keep=2)
        checkpoint_manager = ocp.CheckpointManager(
            model_ckpt_dir, ocp.PyTreeCheckpointer(), options=options
        )
        load_inference_checkpoint(checkpoint_manager, model, step=checkpoint_step)

    aux_loss_config = None
    from utils import freeze_dict
    if hasattr(train_cfg, "trainer") and train_cfg.trainer.get("aux_losses"):
        aux_loss_config = freeze_dict(
            {k: dict(v) for k, v in train_cfg.trainer.aux_losses.items()}
        )
    if cfg.get("aux_losses"):
        from utils import unfreeze_dict
        override = {k: dict(v) for k, v in cfg.aux_losses.items()}
        merged = unfreeze_dict(aux_loss_config) if aux_loss_config is not None else {}
        for k, v in override.items():
            merged[k] = {**(merged.get(k) or {}), **v}
        aux_loss_config = freeze_dict(merged)

    if cfg.get("evals"):
        for eval_key, eval_cfg in sorted_evals:
            actual_eval_cfg = eval_cfg.get("eval", eval_cfg)

            eval_tp_devices = eval_cfg.get("tp_devices", cfg.tp_devices)
            if eval_tp_devices != current_tp_devices:
                if jax.process_index() == 0:
                    print(f"[eval_worker] tp_devices changed {current_tp_devices}→{eval_tp_devices}, reloading model...")
                del model
                gc.collect()
                jax.effects_barrier()
                current_tp_devices = eval_tp_devices
                model = get_model(effective_model_cfg, current_tp_devices)
                if checkpoint_manager is not None:
                    load_inference_checkpoint(checkpoint_manager, model, step=checkpoint_step)

            dataset = None
            if eval_cfg.get("dataset"):
                if jax.process_index() == 0:
                    print(f"Loading dataset '{eval_cfg.dataset.name}' for eval '{eval_key}'...")
                dataset = get_dataset(eval_cfg.dataset, model)

            eval_cfg_dict = OmegaConf.to_container(actual_eval_cfg, resolve=True)
            metrics_cfg = eval_cfg_dict.pop("metrics", None)

            evaluator = get_evaluator(OmegaConf.create(eval_cfg_dict), key=eval_key)

            doc_dataset = None
            if actual_eval_cfg.get("doc_dataset"):
                if jax.process_index() == 0:
                    print(f"Loading doc_dataset '{actual_eval_cfg.doc_dataset.name}' for eval '{eval_key}'...")
                doc_dataset = get_dataset(actual_eval_cfg.doc_dataset, model)

            inference_metrics = evaluator.evaluate(
                model, dataset,
                step=args.step,
                doc_dataset=doc_dataset,
                aux_loss_config=aux_loss_config,
            )

            output_file = None
            if actual_eval_cfg.get("output_file"):
                output_file = _output_file_path(args.output_dir, args.step, eval_key, actual_eval_cfg.output_file)

            manifest[eval_key] = {
                "output_file": output_file,
                "metrics_cfg": metrics_cfg,
                "inference_metrics": inference_metrics,
            }

    else:
        # Single eval mode
        actual_eval_cfg = cfg.eval
        eval_key = actual_eval_cfg.get("type", actual_eval_cfg.get("name"))

        if jax.process_index() == 0:
            print(f"Loading dataset: {cfg.dataset.name}")

        dataset = get_dataset(cfg.dataset, model)

        eval_cfg_dict = OmegaConf.to_container(actual_eval_cfg, resolve=True)
        metrics_cfg = eval_cfg_dict.pop("metrics", None)

        evaluator = get_evaluator(OmegaConf.create(eval_cfg_dict), key=eval_key)

        doc_dataset = None
        if actual_eval_cfg.get("doc_dataset"):
            doc_dataset = get_dataset(actual_eval_cfg.doc_dataset, model)

        inference_metrics = evaluator.evaluate(
            model, dataset,
            step=args.step,
            doc_dataset=doc_dataset,
            aux_loss_config=aux_loss_config,
        )

        output_file = None
        if actual_eval_cfg.get("output_file"):
            output_file = _output_file_path(args.output_dir, args.step, eval_key, actual_eval_cfg.output_file)

        manifest[eval_key] = {
            "output_file": output_file,
            "metrics_cfg": metrics_cfg,
            "inference_metrics": inference_metrics,
        }

    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[eval_worker] manifest written to {args.manifest_out}")


if __name__ == "__main__":
    main()
