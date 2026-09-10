"""
GradAccumTrainer: gradient accumulation with a shared memory bank.

Decouples the number of documents embedded (memory bank size) from the
mini-batch size for the main model forward/backward:

  1. embed_forward(all docs)  → mem_k, mem_v          (full memory bank, built once)
  2. for each query mini-batch:
       main_forward(mini_batch, mem_k, mem_v) → loss   (smaller activation footprint)
       accumulate ∂L/∂mem_k, ∂L/∂main_weights
  3. embed_backward(∂L/∂mem_k)                         (single backward through embed)
  4. optimizer.update(merged_grads)

The two-pass gather temp that caused OOM scales with B_per_device (per mini-batch),
not with the total batch.  With n_accum=4 and batch_size=128 the gather temp drops
from ~8 GB (B_per_device=16) to ~2 GB (B_per_device=4).

No changes needed to Trainer or train.py beyond selecting this class.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial

from utils import unfreeze_dict
from losses import compute_aux_losses
from models.qwen3_mem_embed import embed_forward, main_forward
from models.utils import split_weights, merge_weights

from .trainer import Trainer


class GradAccumTrainer(Trainer):
    """Trainer with gradient accumulation over query mini-batches.

    Drop-in replacement for Trainer.  Set ``trainer.grad_accum_steps`` in
    config to the number of mini-batches to split each batch into.
    ``batch_size`` must be divisible by ``grad_accum_steps``.
    """

    def __init__(self, cfg, model, data, optimizer, checkpoint_manager, **kwargs):
        super().__init__(cfg, model, data, optimizer, checkpoint_manager, **kwargs)
        self._n_accum = int(cfg.trainer.get("grad_accum_steps", 4))

    # ------------------------------------------------------------------
    # Override _train_step as an instance method so we can inject n_accum.
    # Trainer.train() calls self._train_step(...), which Python resolves to
    # this method via normal MRO — no changes to trainer.py needed.
    # ------------------------------------------------------------------
    def _train_step(self, forward, optimizer, weights, opt_state,
                    inputs, targets, input_masks, loss_masks,
                    ce_weight, aux_loss_config=None, ce_enable=None):
        # ce_enable (per-row CE gate) is not yet wired through grad accumulation;
        # it is unused here. Grad-accum runs (grad_accum_steps>1) don't use the
        # CE-masked similarity variant, so this is a no-op for them.
        return GradAccumTrainer._grad_accum_impl(
            forward, optimizer, weights, opt_state,
            inputs, targets, input_masks, loss_masks,
            ce_weight, aux_loss_config,
            n_accum=self._n_accum,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("forward", "optimizer", "aux_loss_config", "n_accum"))
    def _grad_accum_impl(forward, optimizer, weights, opt_state,
                         inputs, targets, input_masks, loss_masks,
                         ce_weight, aux_loss_config=None, n_accum=4):
        """JIT-compiled gradient accumulation step."""

        aux_cfg = unfreeze_dict(aux_loss_config)
        collect_aux = aux_cfg is not None and len(aux_cfg) > 0

        # Cast all masks to bool, matching original _train_step behaviour.
        input_masks = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
        loss_masks = loss_masks.astype(jnp.bool_)

        # forward is partial(qwen3_mem_embed.forward, cfg); cfg is static at JIT time.
        cfg = forward.args[0]
        main_cfg = cfg["main_model"]
        embed_cfg = cfg["embed_model"]

        main_weights, embed_weights = split_weights(weights, ["main_model", "embed_model"])

        # ----------------------------------------------------------------
        # Step 1: embed all docs once; save vjp function for later backward.
        # ----------------------------------------------------------------
        def _embed(ew):
            mk, mv, mm, _ = embed_forward(embed_cfg, inputs["docs"], ew,
                                          input_masks["docs_mask"])
            return mk, mv, mm

        (mem_k, mem_v, mem_mask), embed_vjp_fn = jax.vjp(_embed, embed_weights)

        # main_weights["mem_k"/"mem_v"/"mem_mask"] are (0,)-shaped placeholders in the
        # optimizer state.  We must NOT pass freshly-computed (131072,1024) grads for
        # those keys or optax will see a shape mismatch.
        # Strategy: differentiate w.r.t. (main params without mem placeholders) AND
        # (mem_k, mem_v) separately via argnums=(0,1).  Accumulated mem_k/mem_v grads
        # go to the embed backward; the optimizer always sees zeros for the placeholders.
        mem_placeholder_keys = {k for k in ("mem_k", "mem_v", "mem_mask") if k in main_weights}
        main_only = {k: v for k, v in main_weights.items() if k not in mem_placeholder_keys}

        # ----------------------------------------------------------------
        # Step 2: accumulate gradients over n_accum query mini-batches.
        # ----------------------------------------------------------------
        batch_size = inputs["batch"].shape[0]
        mini_bs = batch_size // n_accum

        def _mini_loss(main_only, mem_kv, x_m, tgt_m, mask_m, lmask_m, mini_im,
                       pos_slot_indices=None):
            mk, mv = mem_kv
            mw = {**main_only, "mem_k": mk, "mem_v": mv, "mem_mask": mem_mask}
            logits, _, aux_data = main_forward(
                main_cfg, x_m, mw, mask_m, collect_aux=collect_aux,
                pos_slot_indices=pos_slot_indices,
            )
            one_hot = jax.nn.one_hot(tgt_m, logits.shape[-1])
            ce = optax.softmax_cross_entropy(logits, one_hot)
            ml = (ce * lmask_m).sum() / (lmask_m.sum() + 1e-9)
            ar = compute_aux_losses(aux_data, lmask_m, mini_im, x_m, aux_cfg)
            return (ml * ce_weight + ar["total"]) / n_accum, (ml, ar)

        # Use lax.scan so the mini-batch body is compiled ONCE (not unrolled n_accum times).
        # Unrolling 4× inflates the XLA binary by ~4× and causes OOM at program-load time.
        def _mini_step(carry, i):
            acc_grad_main, acc_gmk, acc_gmv, total_ce_loss = carry
            x_m   = jax.lax.dynamic_slice_in_dim(inputs["batch"],            i * mini_bs, mini_bs, 0)
            tgt_m = jax.lax.dynamic_slice_in_dim(targets,                    i * mini_bs, mini_bs, 0)
            msk_m = jax.lax.dynamic_slice_in_dim(input_masks["batch_mask"],  i * mini_bs, mini_bs, 0)
            lm_m  = jax.lax.dynamic_slice_in_dim(loss_masks,                 i * mini_bs, mini_bs, 0)

            # Slice pos_doc_mask so loss functions see [mini_bs, docs_per_query] not [full_B, docs_per_query].
            # Pass i as mini_batch_offset so losses can compute global query indices.
            mini_im = dict(input_masks)
            mini_im["batch_mask"] = msk_m
            pos_slot_indices = None
            if "pos_doc_mask" in input_masks:
                mini_im["pos_doc_mask"] = jax.lax.dynamic_slice_in_dim(
                    input_masks["pos_doc_mask"], i * mini_bs, mini_bs, 0)
                mini_im["mini_batch_offset"] = i

                # Build flat memory slot indices for all docs of this mini-batch so that
                # mem_lookup_two_pass can compute positive-doc logits in pass-2.
                num_total_docs = input_masks["docs_mask"].shape[0]
                effective_doc_len = mem_k.shape[0] // num_total_docs
                docs_per_query = num_total_docs // batch_size
                global_query_idx = i * mini_bs + jnp.arange(mini_bs)  # [mini_bs]
                global_doc_idx = (
                    global_query_idx[:, None] * docs_per_query
                    + jnp.arange(docs_per_query)[None, :]
                )  # [mini_bs, docs_per_query]
                pos_slot_indices = (
                    global_doc_idx[:, :, None] * effective_doc_len
                    + jnp.arange(effective_doc_len)[None, None, :]
                ).reshape(mini_bs, -1)  # [mini_bs, docs_per_query * effective_doc_len]

            (_, (ce_l, ar)), (g_main, (gmk, gmv)) = jax.value_and_grad(
                _mini_loss, argnums=(0, 1), has_aux=True
            )(main_only, (mem_k, mem_v), x_m, tgt_m, msk_m, lm_m, mini_im, pos_slot_indices)

            acc_grad_main = jax.tree_util.tree_map(jnp.add, acc_grad_main, g_main)
            acc_gmk = acc_gmk + gmk
            acc_gmv = acc_gmv + gmv
            total_ce_loss = total_ce_loss + ce_l / n_accum
            return (acc_grad_main, acc_gmk, acc_gmv, total_ce_loss), ar

        init_carry = (
            jax.tree_util.tree_map(jnp.zeros_like, main_only),
            jnp.zeros_like(mem_k),
            jnp.zeros_like(mem_v),
            jnp.zeros(()),
        )
        (acc_grad_main, acc_gmk, acc_gmv, total_ce_loss), stacked_aux = jax.lax.scan(
            _mini_step, init_carry, jnp.arange(n_accum)
        )
        # scan stacks outputs along axis 0; take the last iteration's aux for logging.
        last_aux = jax.tree_util.tree_map(lambda x: x[-1], stacked_aux)

        # ----------------------------------------------------------------
        # Step 3: backward through embed using accumulated mem gradients.
        # ----------------------------------------------------------------
        (grad_embed_weights,) = embed_vjp_fn(
            (acc_gmk, acc_gmv, jnp.zeros_like(mem_mask))
        )

        # Restore full main_weights structure: placeholders get zero grads so optimizer
        # state shapes (0,) are preserved.
        grad_main = {
            **acc_grad_main,
            **{k: jnp.zeros_like(main_weights[k]) for k in mem_placeholder_keys},
        }

        full_grads = merge_weights(["main_model", "embed_model"],
                                   [grad_main, grad_embed_weights])

        updates, opt_state = optimizer.update(full_grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        return weights, opt_state, total_ce_loss, last_aux
