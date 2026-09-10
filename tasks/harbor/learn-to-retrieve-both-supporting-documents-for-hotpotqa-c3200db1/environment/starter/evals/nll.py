import jax
import jax.numpy as jnp
import optax
import numpy as np
import json
import wandb
from tqdm import tqdm
from functools import partial
from .base import Evaluator
from utils import process_train_pairs, unfreeze_dict
from losses.registry import compute_aux_losses
from losses.doc_access_acc import compute_doc_access_acc

class NLLEvaluator(Evaluator):
    def evaluate(self, model, dataset, step=None, aux_loss_config=None, **kwargs):
        # Trainer config aux_losses take priority; fall back to aux_losses in eval config
        if aux_loss_config is None and self.cfg.get('aux_losses'):
            from utils import freeze_dict
            aux_loss_config = freeze_dict({k: dict(v) for k, v in self.cfg.aux_losses.items()})
        if jax.process_index() == 0:
            print("Starting NLL Evaluation...")

        aux_loss_cfg_dict = unfreeze_dict(aux_loss_config) if aux_loss_config is not None else {}
        compute_acc = bool(aux_loss_cfg_dict.get('doc_access_acc', {}).get('enabled', False))
        # Read-channel telemetry (mem_pos_weight_mass etc.) on the NLL batches. Gated on the same
        # aux config the trainer uses, so a checkpoint trained with the weight-0 telemetry block
        # gets it at eval for free (eval_worker seeds aux_loss_config from train_cfg.trainer);
        # enable by hand with '+aux_losses.mem_pos_weight_mass.enabled=true'.
        compute_tel = bool(aux_loss_cfg_dict.get('mem_pos_weight_mass', {}).get('enabled', False))

        @partial(jax.jit, static_argnames=("forward", "aux_loss_config", "collect_aux"))
        def loss_fn(forward, weights, inputs, targets, input_masks, loss_masks, aux_loss_config=None, collect_aux=False):
            pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)

            # collect_aux is gated on whether doc_access_acc is enabled; eval has no
            # backward pass so jax.remat is a no-op and there is no memory risk.
            output = forward(inputs, weights, pad_mask=pad_mask, collect_aux=collect_aux)
            logits = output.logits

            preds = jnp.argmax(logits, axis=-1)

            one_hot = jax.nn.one_hot(targets, logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits, one_hot)
            batch_loss = (loss * loss_masks).sum() / (loss_masks.sum() + 1e-9)
            per_sample_loss = (loss * loss_masks).sum(axis=-1) / (loss_masks.sum(axis=-1) + 1e-9)

            return batch_loss, preds, per_sample_loss, loss_masks, output.aux

        nll_scores = []
        acc_scores = []
        telemetry_batches = []
        samples = []

        iterator = dataset.generator(num_epochs=1)

        pbar = tqdm(desc="Evaluating NLL")
        
        eval_steps = self.cfg.get('eval_steps', None)
        step_count = 0

        lookup_chunk_size = self.cfg.get("lookup_chunk_size", None)
        reset_chunk_size = False
        if lookup_chunk_size is not None:
            if jax.process_index() == 0:
                print(f"  Enabling chunked retrieval (lookup_chunk_size={lookup_chunk_size})")
            if model.cfg.get("mem_lookup_chunk_size", None) is None:
                reset_chunk_size = True
            model.cfg['main_model']['mem_lookup_chunk_size'] = lookup_chunk_size

        for tokens, masks in iterator:
            if eval_steps is not None and step_count >= eval_steps:
                break

            inputs, targets, input_masks, loss_masks, _ce_enable = process_train_pairs(tokens, masks)

            batch_loss, preds, per_sample_loss, loss_masks_out, aux_data = loss_fn(
                model.forward,
                model.weights,
                inputs,
                targets,
                input_masks,
                loss_masks,
                collect_aux=compute_acc or compute_tel,
            )
            nll_scores.append(float(batch_loss))

            if compute_acc and aux_data is not None and isinstance(inputs, dict) and "docs" in inputs:
                acc = float(compute_doc_access_acc(aux_data, loss_masks, input_masks, inputs))
                acc_scores.append(acc)

            if compute_tel and aux_data is not None:
                try:
                    from losses.mem_telemetry import collect_eval_telemetry
                    tel = collect_eval_telemetry(aux_data, loss_masks, input_masks)
                    if tel:
                        telemetry_batches.append(tel)
                except Exception as e:
                    if jax.process_index() == 0 and not telemetry_batches:
                        print(f"  mem telemetry warn: {e}")

            # Convert to numpy for per-sample processing
            batch_preds = np.array(jax.experimental.multihost_utils.process_allgather(preds, tiled=True))
            batch_targets = np.array(jax.experimental.multihost_utils.process_allgather(targets, tiled=True))
            batch_loss_masks = np.array(jax.experimental.multihost_utils.process_allgather(loss_masks_out, tiled=True))
            per_sample_nlls = np.array(jax.experimental.multihost_utils.process_allgather(per_sample_loss, tiled=True))

            if isinstance(inputs, dict):
                batch_inputs = np.array(jax.experimental.multihost_utils.process_allgather(inputs["batch"], tiled=True))
            else:
                batch_inputs = np.array(jax.experimental.multihost_utils.process_allgather(inputs, tiled=True))

            # Extract docs if present
            raw_docs = np.array(jax.experimental.multihost_utils.process_allgather(tokens["docs"], tiled=True)) if isinstance(tokens, dict) and "docs" in tokens else None
            docs_mask = np.array(jax.experimental.multihost_utils.process_allgather(masks["docs_mask"], tiled=True)) if "docs_mask" in masks else None
            pos_doc_mask = np.array(jax.experimental.multihost_utils.process_allgather(masks["pos_doc_mask"], tiled=True)) if "pos_doc_mask" in masks else None

            for b in range(batch_preds.shape[0]):
                valid_indices = np.where(batch_loss_masks[b] > 0)[0]
                if len(valid_indices) == 0:
                    continue

                entry = {
                    "nll": float(per_sample_nlls[b]),
                    "prompt": model.tokenizer.decode(batch_inputs[b]),
                    "generated": model.tokenizer.decode(batch_preds[b, valid_indices]),
                    "ground_truth": model.tokenizer.decode(batch_targets[b, valid_indices]),
                }

                if raw_docs is not None:
                    if pos_doc_mask is not None:
                        # Multi-doc (streaming_qa): reshape (B*M, D) → (B, M, D)
                        M = pos_doc_mask.shape[1]
                        raw_docs_3d = raw_docs.reshape(batch_preds.shape[0], M, -1)
                        dm_3d = docs_mask.reshape(batch_preds.shape[0], M, -1)
                        doc_texts = []
                        for d in range(M):
                            if pos_doc_mask[b, d]:
                                idx = np.where(dm_3d[b, d] > 0)[0]
                                doc_texts.append(model.tokenizer.decode(raw_docs_3d[b, d, idx]))
                        entry["doc"] = " || ".join(doc_texts) if doc_texts else ""
                        entry["num_doc_chunks"] = int(pos_doc_mask[b].sum())
                    else:
                        # Single-doc (BioR): existing logic
                        doc_indices = np.where(docs_mask[b] > 0)[0] if docs_mask is not None else slice(None)
                        entry["doc"] = model.tokenizer.decode(raw_docs[b, doc_indices])

                samples.append(entry)

            step_count += 1
            pbar.update(1)
            pbar.set_postfix(nll=f"{nll_scores[-1]:.4f}")

        pbar.close()

        if reset_chunk_size:
            model.cfg['main_model']['mem_lookup_chunk_size'] = None

        avg_nll = float(np.mean(nll_scores)) if nll_scores else 0.0
        avg_acc = float(np.mean(acc_scores)) if acc_scores else None
        # Batch-mean of each telemetry key (mem_pos_weight_mass, mem_topk_entropy, ...).
        avg_tel = {}
        if telemetry_batches:
            for tk in set().union(*telemetry_batches):
                vals = [b[tk] for b in telemetry_batches if tk in b]
                if vals:
                    avg_tel[tk] = float(np.mean(vals))
        # NLL eval does not accumulate per-aux-loss values; keep the output schema
        # stable without referencing an undefined name.
        avg_aux_losses = {"doc_access_acc": avg_acc} if avg_acc is not None else {}
        avg_aux_losses.update(avg_tel)
        if jax.process_index() == 0:
            acc_str = f", doc_access_acc: {avg_acc:.4f}" if avg_acc is not None else ""
            mass = avg_tel.get("mem_pos_weight_mass")
            mass_str = f", mem_pos_weight_mass: {mass:.4f}" if mass is not None else ""
            print(f"NLL Evaluation Complete. Average NLL: {avg_nll:.4f}{acc_str}{mass_str}")

        # Log to wandb
        if jax.process_index() == 0 and wandb.run is not None:
            log_dict = {f"eval/{self.key}/nll": avg_nll}
            if avg_acc is not None:
                log_dict[f"eval/{self.key}/doc_access_acc"] = avg_acc
            for tk, tv in avg_tel.items():
                log_dict[f"eval/{self.key}/{tk}"] = tv
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

        if self.cfg.get("output_file"):
            if jax.process_index() == 0:
                output = {
                    "stats": {
                        "num_samples": len(samples),
                        "avg_nll": avg_nll,
                        **avg_aux_losses,
                    },
                    "samples": samples,
                }
                output_path = self._get_output_path(step, self.cfg.output_file)
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
                print(f"Saved NLL results to {output_path}")
                if wandb.run is not None:
                    artifact_name = f"{wandb.run.id}-eval-{self.key}-step-{step}-results" if step is not None else f"{wandb.run.id}-eval-{self.key}-results"
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="evaluation_results",
                    )
                    artifact.add_file(output_path)
                    wandb.log_artifact(artifact)

        inference_metrics = {"nll": avg_nll}
        if avg_acc is not None:
            inference_metrics["doc_access_acc"] = avg_acc
        inference_metrics.update(avg_tel)
        return inference_metrics
