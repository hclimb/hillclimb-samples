import jax
import jax.numpy as jnp
from models.qwen3 import apply_rope

def test_nan_propagation():
    # Simulate a sequence where the second token is fully masked
    # In attention, if all masks are False for a row, dot_product_attention returns NaN
    B, T, N, H = 1, 2, 1, 64
    q = jnp.ones((B, T, N, H), dtype=jnp.bfloat16)
    k = jnp.ones((B, T, N, H), dtype=jnp.bfloat16)
    v = jnp.ones((B, T, N, H), dtype=jnp.bfloat16)
    
    # Mask: row 0 is True, row 1 is False (all masked)
    mask = jnp.array([[[[True, True], [False, False]]]], dtype=bool)
    
    out = jax.nn.dot_product_attention(q, k, v, mask=mask)
    print(f"Attention output with all-False mask row:\n{out}")

    # Now simulate the LoRA branch
    # x_norm has NaN at index 1
    x_norm = out[..., 0, :] # [B, T, H]
    w_a = jnp.ones((16, H), dtype=jnp.bfloat16)
    w_b = jnp.zeros((H, 16), dtype=jnp.bfloat16)
    
    lora_branch = jnp.einsum('btd,rd->btr', x_norm, w_a)
    lora_branch = jnp.einsum('btr,dr->btd', lora_branch, w_b)
    
    print(f"LoRA branch with NaN in input (but zero weights):\n{lora_branch}")
    
    gate_base = jnp.ones((B, T, H), dtype=jnp.bfloat16)
    gate_base += lora_branch
    print(f"Final activations (base + lora):\n{gate_base}")

if __name__ == "__main__":
    test_nan_propagation()
