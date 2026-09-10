import jax
import jax.numpy as jnp
import re

def merge_weights(names, weights):
    
    if len(names) != len(weights):
        raise ValueError("names and weights must have the same length")
    
    # make a new jax tree, loop through each of the weights in each weights and add them in format {name.{weight}}
    new_weights = {}
    for name, weight_dict in zip(names, weights):
        new_weights.update({f"{name}.{k}": v for k, v in weight_dict.items()})
    return new_weights

def merge_configs(names, configs):
    
    if len(names) != len(configs):
        raise ValueError("names and configs must have the same length")
    
    new_configs = {name: config for name, config in zip(names, configs)}
    return new_configs

def split_weights(weights, names):
    
    new_weights = []
    for name in names:
        new_weights.append({k.replace(f"{name}.", ""): v for k, v in weights.items() if k.startswith(f"{name}.")})
    return tuple(new_weights)

def init_lora(weights, lora_cfg, sharding_fn):
    rank = lora_cfg['rank']
    patterns = [re.compile(p) for p in lora_cfg['params']]
    lora_weights = {}
    rng_key = jax.random.PRNGKey(42)
    
    for key, val in weights.items():
        if any(p.search(key) for p in patterns):
            d1, d2 = val.shape
            
            # a_proj: [R, d2]
            a_key = key.replace('_proj', '_a_proj')
            rng_key, subkey = jax.random.split(rng_key)
            lora_weights[a_key] = jax.device_put(
                jax.nn.initializers.normal(stddev=1/rank)(subkey, (rank, d2), jnp.bfloat16),
                sharding_fn(a_key)
            )
            
            # b_proj: [d1, R]
            b_key = key.replace('_proj', '_b_proj')
            lora_weights[b_key] = jax.device_put(
                jnp.zeros((d1, rank), dtype=jnp.bfloat16),
                sharding_fn(b_key)
            )
    weights.update(lora_weights)
    return weights