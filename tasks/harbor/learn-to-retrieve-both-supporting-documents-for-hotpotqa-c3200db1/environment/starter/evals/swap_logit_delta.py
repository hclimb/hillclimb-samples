import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import tqdm
from functools import partial
from jax.sharding import PartitionSpec as P

from .base import Evaluator
from utils import process_train_pairs


def _build_swapped_docs_jax(docs_reshaped, docs_mask_reshaped, pos_doc_mask):
    """Replace each example's positive doc chunks with a distractor from the same example.

    docs_reshaped:      (B, M, D) — batch's doc chunks
    docs_mask_reshaped: (B, M, D)
    pos_doc_mask:       (B, M) — 1 where chunk is a positive doc for that query

    Uses the first non-positive chunk per row as the distractor. Vectorised so
    the sharding of `docs_reshaped` on the batch axis carries through (only
    per-shard fancy indexing along the chunk axis is needed).

    Edge case: if an example has zero non-positive chunks, argmax falls back to
    index 0 (a positive chunk) and its per-example delta contribution is ~0.
    Typical qa configs (num_chunks_per_doc=4, ~2 neg per example) avoid this.
    """
    B, M, D = docs_reshaped.shape
    pos_mask = pos_doc_mask.astype(jnp.bool_)
    neg_idx = jnp.argmax((~pos_mask).astype(jnp.int32), axis=1)  # (B,)
    b_range = jnp.arange(B)
    neg_doc = docs_reshaped[b_range, neg_idx]                    # (B, D)
    neg_doc_mask = docs_mask_reshaped[b_range, neg_idx]          # (B, D)
    docs_swapped = jnp.where(
        pos_mask[:, :, None], neg_doc[:, None, :], docs_reshaped
    )
    docs_mask_swapped = jnp.where(
        pos_mask[:, :, None], neg_doc_mask[:, None, :], docs_mask_reshaped
    )
    return docs_swapped, docs_mask_swapped


class SwapLogitDeltaEvaluator(Evaluator):
    """Per-batch swap_logit_delta evaluator (variant A: per-batch pool).

    For each teacher-forced qa batch:
      Pass A: forward with the batch's own docs  → per-token log P(gt).
      Pass C: forward with each query's positive doc chunks overwritten by a
              distractor from the same query's neg pool → per-token log P(gt).
      Per-example delta = mean(logp_A - logp_C) over loss-masked positions.
      Batch metric = mean over the batch. Eval metric = mean over batches.

    Compared with the corpus-based L3C in evals/gen_large_mem.py, this operates
    on the batch's own doc pool (no persistent corpus / doc_dataset) and
    matches the nll eval task-config pattern so it can drop into the trainer's
    eval_interval loop later.
    """

    def evaluate(self, model, dataset, step=None, aux_loss_config=None, **kwargs):
        if jax.process_index() == 0:
            print("Starting Swap Logit Delta Evaluation (variant A: per-batch pool)...")

        @partial(jax.jit, static_argnames=("forward",))
        def _per_sample_logp_fn(forward, weights, inputs, targets, input_masks, loss_masks):
            pad_mask = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
            output = forward(inputs, weights, pad_mask=pad_mask)
            logits = output.logits                               # (B, T, V)
            ls = jax.nn.log_softmax(logits, axis=-1)             # bf16 kept; see gen_large_mem.py:618
            B_ax = jnp.arange(targets.shape[0])[:, None]
            T_ax = jnp.arange(targets.shape[1])[None, :]
            # Explicit out_sharding on the fancy-indexed gather along the model-
            # sharded vocab axis, matching _prefill_aux_l2_fn (gen_large_mem.py:628).
            logp_gt = ls.at[B_ax, T_ax, targets].get(
                out_sharding=P('data', None)
            ).astype(jnp.float32)                                # (B, T)
            # NaN/Inf can leak into logp_gt at padded positions (rare — seen when
            # the prompt tail is all-endoftext and pass-A retrieval collapses).
            # Multiplying NaN by a 0 loss-mask still yields NaN, which then poisons
            # the reduction. Gate positions on `finite & loss_mask` and use the
            # valid-count as the denominator instead of loss_masks.sum().
            valid = jnp.isfinite(logp_gt) & (loss_masks > 0)
            valid_f = valid.astype(jnp.float32)
            logp_gt_safe = jnp.where(valid, logp_gt, 0.0)
            per_sample_logp = logp_gt_safe.sum(axis=-1) / jnp.maximum(valid_f.sum(axis=-1), 1.0)
            per_sample_valid_count = valid_f.sum(axis=-1)        # (B,) — 0 means example dropped
            return per_sample_logp, per_sample_valid_count

        deltas = []
        wall_times = []
        samples = []

        eval_steps = self.cfg.get('eval_steps', None)
        step_count = 0
        pbar = tqdm(desc="Evaluating swap_logit_delta")

        for tokens, masks in dataset.generator(num_epochs=1):
            if eval_steps is not None and step_count >= eval_steps:
                break
            if not (isinstance(tokens, dict) and "docs" in tokens):
                raise ValueError(
                    "swap_logit_delta requires a batch with 'docs'; use a qa "
                    "dataset (configs/dataset/qa.yaml)."
                )
            if "pos_doc_mask" not in masks:
                raise ValueError(
                    "swap_logit_delta requires 'pos_doc_mask' in batch masks; the "
                    "qa dataset must be configured with provide_docs=true."
                )

            inputs, targets, input_masks, loss_masks, _ = process_train_pairs(tokens, masks)

            # Build swapped inputs in JAX (sharding-preserving).
            docs_flat = inputs["docs"]                            # (B*M, D)
            docs_mask_flat = input_masks["docs_mask"]             # (B*M, D)
            pos_doc_mask = input_masks["pos_doc_mask"]            # (B, M)
            B, M = pos_doc_mask.shape
            D = docs_flat.shape[-1]
            docs_swapped, docs_mask_swapped = _build_swapped_docs_jax(
                docs_flat.reshape(B, M, D),
                docs_mask_flat.reshape(B, M, D),
                pos_doc_mask,
            )
            inputs_swapped = {**inputs, "docs": docs_swapped.reshape(B * M, D)}
            input_masks_swapped = {
                **input_masks,
                "docs_mask": docs_mask_swapped.reshape(B * M, D),
                # No positives left after swap. The model uses pos_doc_mask's shape
                # (not values) to auto-build pos_slot_indices, so this is
                # semantically accurate without changing compute.
                "pos_doc_mask": jnp.zeros_like(pos_doc_mask),
            }

            t0 = time.perf_counter()
            per_sample_logp_A, valid_A = _per_sample_logp_fn(
                model.forward, model.weights, inputs, targets, input_masks, loss_masks
            )
            per_sample_logp_C, valid_C = _per_sample_logp_fn(
                model.forward, model.weights, inputs_swapped, targets, input_masks_swapped, loss_masks
            )
            jax.block_until_ready(per_sample_logp_A)
            jax.block_until_ready(per_sample_logp_C)
            wall_times.append(time.perf_counter() - t0)

            logp_A_np = np.array(jax.experimental.multihost_utils.process_allgather(
                per_sample_logp_A, tiled=True))
            logp_C_np = np.array(jax.experimental.multihost_utils.process_allgather(
                per_sample_logp_C, tiled=True))
            valid_A_np = np.array(jax.experimental.multihost_utils.process_allgather(
                valid_A, tiled=True))
            valid_C_np = np.array(jax.experimental.multihost_utils.process_allgather(
                valid_C, tiled=True))

            # An example counts only when BOTH passes had at least one valid
            # (finite, loss-masked) answer position. Sidesteps the NaN-poisoning
            # we saw with padded-tail prompts on pass A.
            has_answer = (valid_A_np > 0) & (valid_C_np > 0)

            delta_per_sample = logp_A_np - logp_C_np
            if has_answer.any():
                # nanmean as belt-and-suspenders — the NaN mask above should already
                # be enough, but nanmean guards against any residual non-finite.
                deltas.append(float(np.nanmean(delta_per_sample[has_answer])))

            if self.cfg.get("output_file") and jax.process_index() == 0:
                batch_np = np.array(jax.experimental.multihost_utils.process_allgather(
                    inputs["batch"], tiled=True))
                targets_np = np.array(jax.experimental.multihost_utils.process_allgather(
                    targets, tiled=True))
                loss_masks_np = np.array(jax.experimental.multihost_utils.process_allgather(
                    loss_masks, tiled=True))
                for b in range(batch_np.shape[0]):
                    if not has_answer[b]:
                        continue
                    valid = np.where(loss_masks_np[b] > 0)[0]
                    samples.append({
                        "swap_logit_delta": float(delta_per_sample[b]),
                        "logp_orig":        float(logp_A_np[b]),
                        "logp_swap":        float(logp_C_np[b]),
                        "prompt":           model.tokenizer.decode(batch_np[b]),
                        "ground_truth":     model.tokenizer.decode(targets_np[b, valid]),
                    })

            step_count += 1
            pbar.update(1)
            if deltas:
                pbar.set_postfix(delta=f"{deltas[-1]:+.4f}", batch_s=f"{wall_times[-1]:.2f}")

        pbar.close()

        avg_delta = float(np.nanmean(deltas)) if deltas else 0.0
        # Skip first batch's wall time — it includes one-time JIT compile of the forward.
        steady_wall_times = wall_times[1:] if len(wall_times) > 1 else wall_times
        avg_wall_time = float(np.mean(steady_wall_times)) if steady_wall_times else 0.0
        if jax.process_index() == 0:
            print(f"Swap Logit Delta Evaluation Complete. "
                  f"delta={avg_delta:+.4f} nats over {len(deltas)} batches, "
                  f"wall_time_per_batch={avg_wall_time:.2f}s (steady state, excludes first-batch compile)")

        if jax.process_index() == 0 and wandb.run is not None:
            log_dict = {
                f"eval/{self.key}/swap_logit_delta":     avg_delta,
                f"eval/{self.key}/wall_time_per_batch_s": avg_wall_time,
            }
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

        if self.cfg.get("output_file") and jax.process_index() == 0:
            output = {
                "stats": {
                    "num_batches":             len(deltas),
                    "num_samples":             len(samples),
                    "swap_logit_delta":        avg_delta,
                    "wall_time_per_batch_s":   avg_wall_time,
                },
                "samples": samples,
            }
            output_path = self._get_output_path(step, self.cfg.output_file)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Saved swap_logit_delta results to {output_path}")
            if wandb.run is not None:
                artifact_name = (
                    f"{wandb.run.id}-eval-{self.key}-step-{step}-results"
                    if step is not None
                    else f"{wandb.run.id}-eval-{self.key}-results"
                )
                artifact = wandb.Artifact(name=artifact_name, type="evaluation_results")
                artifact.add_file(output_path)
                wandb.log_artifact(artifact)

        return {
            "swap_logit_delta":      avg_delta,
            "wall_time_per_batch_s": avg_wall_time,
        }
