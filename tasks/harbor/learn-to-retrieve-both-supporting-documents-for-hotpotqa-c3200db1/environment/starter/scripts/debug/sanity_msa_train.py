"""Sanity: MSA training forward + Eq-5 routing loss + one grad step.

Builds a stock-Qwen3-4B MSA (fresh routers), runs train_forward on a synthetic
QA batch, checks logits/route_scores shapes, that the loss is finite, and that a
single optimizer step succeeds. Run on a TPU VM:
    cd memory-layers && set -a && source .env && set +a && \
    .venv/bin/python scripts/embed/sanity_msa_train.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from models import get_model
from losses import compute_msa_route_loss, compute_msa_route_acc

cfg = OmegaConf.create({
    "name": "qwen3_msa",
    "trainable_params": [".*router_.*"],
    "main_model": {"model_id": "Qwen/Qwen3-4B", "load_weights": True},
    "msa": {"router_layers": "half", "pooling_kernel_size": 64, "top_k_docs": 16,
            "infonce_loss_temp": 0.1, "router_init": "copy", "force_pos_docs": True},
})

print("Loading stock Qwen3-4B + fresh routers ...", flush=True)
model = get_model(cfg, tp_devices=1)
L = model.cfg["num_hidden_layers"]
RL = model.cfg["msa"]["router_layers"]
print("router_layers:", RL, flush=True)
print("fresh router_q_proj L%d present:" % RL[0],
      f"layers.{RL[0]}.router_q_proj" in model.weights, flush=True)

# synthetic batch: B queries, M doc-chunk slots each.
# B must be divisible by the data-mesh size (device_count // tp_devices).
B, T, M, Ld = jax.device_count(), 64, 4, 128
V = model.cfg["vocab_size"]
rng = np.random.default_rng(0)
q_ids = rng.integers(0, V, size=(B, T), dtype=np.int32)
docs = rng.integers(0, V, size=(B * M, Ld), dtype=np.int32)
batch_mask = np.ones((B, T), np.float32)
docs_mask = np.ones((B * M, Ld), np.float32)
pos_doc_mask = np.zeros((B, M), np.float32)
pos_doc_mask[:, 0] = 1.0   # first slot of each query is positive
# Exercise the real padding condition: last slot of each query is fully padded
# -> its pooled K̄/V̄/K̄ᴿ are exact-zero vectors (the NaN-grad trigger). Safe-normalize
# must keep grads finite. docs row index = b*M + m.
docs_mask = docs_mask.reshape(B, M, Ld)
docs_mask[:, M - 1, :] = 0.0
docs_mask = docs_mask.reshape(B * M, Ld)

x = {"batch": jnp.array(q_ids[:, :-1]), "docs": jnp.array(docs)}
targets = jnp.array(q_ids[:, 1:])
pad_mask = {"batch_mask": jnp.array(batch_mask[:, :-1]).astype(bool),
            "docs_mask": jnp.array(docs_mask).astype(bool),
            "pos_doc_mask": jnp.array(pos_doc_mask).astype(bool)}
loss_masks = pad_mask["batch_mask"].astype(jnp.float32)

print("Running train_forward ...", flush=True)
out = model.forward(x, model.weights, pad_mask=pad_mask, collect_aux=True)
print("logits:", out.logits.shape, "expected:", (B, T - 1, V), flush=True)
print("route_scores layers:", len(out.aux["route_scores"]),
      "each:", out.aux["route_scores"][0].shape, "expected:", (B, M), flush=True)
assert out.logits.shape == (B, T - 1, V)
assert len(out.aux["route_scores"]) == len(RL)
assert out.aux["route_scores"][0].shape == (B, M)

rl = compute_msa_route_loss(out.aux, loss_masks, pad_mask, x, temperature=0.1)
ra = compute_msa_route_acc(out.aux, loss_masks, pad_mask, x)
print("route_loss:", float(rl), "route_acc:", float(ra), flush=True)
assert jnp.isfinite(rl)

# one grad step over routers only (full grad, finite check)
def loss_fn(w):
    o = model.forward(x, w, pad_mask=pad_mask, collect_aux=True)
    one_hot = jax.nn.one_hot(targets, o.logits.shape[-1])
    ce = (optax.softmax_cross_entropy(o.logits, one_hot) * loss_masks).sum() / (loss_masks.sum() + 1e-9)
    aux = compute_msa_route_loss(o.aux, loss_masks, pad_mask, x, temperature=0.1)
    return 0.1 * ce + aux

print("Computing grads ...", flush=True)
# jit to mirror the trainer's _train_step (explicit-axis mesh needs a jit context).
loss, grads = jax.jit(jax.value_and_grad(loss_fn))(model.weights)
gnorm = optax.global_norm(grads)
print("loss:", float(loss), "grad_norm:", float(gnorm), flush=True)
assert jnp.isfinite(loss) and jnp.isfinite(gnorm)
print("SANITY_MSA_TRAIN_DONE", flush=True)
