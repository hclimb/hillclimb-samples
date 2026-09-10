"""Pinpoint the exact op producing NaN grad in the route-score path.

Runs the scores-sum gradient with jax_debug_nans=True so JAX raises at the first
NaN-producing primitive, with a traceback to the source line. Uses a synthetic
batch crafted to include fully-padded doc slots + partial slots (the trigger).
"""
import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp, numpy as np
from omegaconf import OmegaConf
from models import get_model

mcfg = OmegaConf.create({
    "name": "qwen3_msa", "trainable_params": [".*"],
    "main_model": {"model_id": "Qwen/Qwen3-4B", "load_weights": True},
    "msa": {"router_layers": "half", "pooling_kernel_size": 64, "top_k_docs": 16,
            "infonce_loss_temp": 0.1, "router_init": "copy", "force_pos_docs": True},
})
print("loading...", flush=True)
model = get_model(mcfg, tp_devices=1)

from data import get_dataset
from utils import process_train_pairs
from hydra import compose, initialize
with initialize(config_path="../../configs", version_base=None):
    fc = compose(config_name="train", overrides=["dataset=qa_hard_neg_think_sft4b",
                 "dataset.num_workers=0", "trainer=staged_msa",
                 "eval_set@trainer.evals=msa_hard_neg_think_nll"])
data = get_dataset(fc.dataset, model)

def scores_sum(w, x, pm):
    o = model.forward(x, w, pad_mask=pm, collect_aux=True)
    sv = o.aux['slot_valid'].astype(jnp.float32)
    return sum((s * sv).sum() for s in o.aux['route_scores'])

gen = data.generator(num_epochs=1)
for i in range(40):
    tokens, masks = next(gen)
    inp, tgt, im, lm = process_train_pairs(tokens, masks)
    pm = jax.tree_util.tree_map(lambda a: a.astype(jnp.bool_), im)
    try:
        g = jax.jit(jax.grad(lambda w: scores_sum(w, inp, pm)))(model.weights)
        gn = float(jnp.sqrt(sum((v.astype(jnp.float32)**2).sum() for v in jax.tree_util.tree_leaves(g))))
        print(f"[{i}] finite gn={gn:.1f}", flush=True)
    except FloatingPointError as e:
        print(f"=== [{i}] NaN RAISED ===", flush=True)
        print(str(e)[:2500], flush=True)
        break
print("PINPOINT_DONE", flush=True)
