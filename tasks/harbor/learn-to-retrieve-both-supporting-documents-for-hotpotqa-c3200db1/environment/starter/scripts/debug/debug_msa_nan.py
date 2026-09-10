"""Debug: find NaN-grad batches on REAL data and localize the source.

Iterates N real batches from qa_hard_neg_think_sft4b through the MSA train step
(jit value_and_grad), counts non-finite grads, and on the first bad batch dumps
the suspect intermediates (doc_scores range, invalid-slot counts, route_loss).
Run on TPU:
  cd memory-layers && source scripts/infrastructure/setup_shell.sh && PYTHONPATH=. .venv/bin/python scripts/embed/debug_msa_nan.py
"""
import jax, jax.numpy as jnp, numpy as np, optax
from omegaconf import OmegaConf
from models import get_model
from data import get_dataset
from losses import compute_msa_route_loss
from utils import process_train_pairs

N_BATCHES = 150

mcfg = OmegaConf.create({
    "name": "qwen3_msa", "trainable_params": [".*"],
    "main_model": {"model_id": "Qwen/Qwen3-4B", "load_weights": True},
    "msa": {"router_layers": "half", "pooling_kernel_size": 64, "top_k_docs": 16,
            "infonce_loss_temp": 0.1, "router_init": "copy", "force_pos_docs": True},
})
print("loading model...", flush=True)
model = get_model(mcfg, tp_devices=1)

# Use the REAL training dataset (all 4 interleaved sources) so positives exist and
# the route-loss path is actually exercised.
from hydra import compose, initialize
with initialize(config_path="../../configs", version_base=None):
    fullcfg = compose(config_name="train",
                      overrides=["dataset=qa_hard_neg_think_sft4b", "dataset.num_workers=0",
                                 "trainer=staged_msa",
                                 "eval_set@trainer.evals=msa_hard_neg_think_nll"])
dcfg = fullcfg.dataset
data = get_dataset(dcfg, model)

def _ce(w, inputs, targets, lm, pm):
    o = model.forward(inputs, w, pad_mask=pm, collect_aux=False)
    oh = jax.nn.one_hot(targets, o.logits.shape[-1])
    return (optax.softmax_cross_entropy(o.logits, oh) * lm).sum() / (lm.sum() + 1e-9)

def _route(w, inputs, targets, lm, pm):
    o = model.forward(inputs, w, pad_mask=pm, collect_aux=True)
    return compute_msa_route_loss(o.aux, lm, pm, inputs, temperature=0.1)

def _scores_sum(w, inputs, targets, lm, pm):
    # grad of just the raw route_scores (isolates _route_doc_scores / cosine-normalize
    # from the InfoNCE loss math). Mask invalid slots to 0 so only valid scores contribute.
    o = model.forward(inputs, w, pad_mask=pm, collect_aux=True)
    sv = o.aux['slot_valid'].astype(jnp.float32)
    return sum((s * sv).sum() for s in o.aux['route_scores'])

@jax.jit
def grad_route(w, inputs, targets, input_masks, loss_masks):
    pm = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
    rl, grl = jax.value_and_grad(_route)(w, inputs, targets, loss_masks, pm)
    return rl, optax.global_norm(grl)

@jax.jit
def grad_scores(w, inputs, targets, input_masks, loss_masks):
    pm = jax.tree_util.tree_map(lambda x: x.astype(jnp.bool_), input_masks)
    ss, gss = jax.value_and_grad(_scores_sum)(w, inputs, targets, loss_masks, pm)
    return ss, optax.global_norm(gss)

bad_rl = bad_ss = 0
gen = data.generator(num_epochs=1)
for i in range(N_BATCHES):
    tokens, masks = next(gen)
    inp, tgt, im, lm = process_train_pairs(tokens, masks)
    rl, gnrl = grad_route(model.weights, inp, tgt, im, lm)
    ss, gnss = grad_scores(model.weights, inp, tgt, im, lm)
    frl, fss = bool(jnp.isfinite(gnrl)), bool(jnp.isfinite(gnss))
    if not (frl and fss):
        if not frl: bad_rl += 1
        if not fss: bad_ss += 1
        print(f"[BAD {i}] gn_route={float(gnrl)} gn_scores={float(gnss)} (route_loss={float(rl):.3f})", flush=True)
    else:
        print(f"[ok {i}] route={float(gnrl):.1f} scores={float(gnss):.1f}", flush=True)

print(f"\nNON-FINITE: route_grad {bad_rl}/{N_BATCHES}, scores_grad {bad_ss}/{N_BATCHES}", flush=True)
print("DEBUG_DONE", flush=True)
