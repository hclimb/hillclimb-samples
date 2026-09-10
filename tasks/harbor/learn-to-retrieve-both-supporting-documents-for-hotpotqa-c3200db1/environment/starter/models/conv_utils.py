"""
Convolution utilities for embedding sequence compression.
"""
import jax.numpy as jnp
import jax.lax as lax


def apply_conv1d(x, weight, bias, stride):
    """Apply 1D conv along the sequence dimension.
    
    Args:
        x: (batch, seq_len, channels) — input tensor
        weight: (out_channels, in_channels, kernel_size) — conv weight
        bias: (out_channels,) — conv bias
        stride: int — conv stride
    
    Returns:
        (batch, new_seq_len, out_channels)
    """
    # Transpose to channel-first: (batch, channels, seq_len)
    x_t = jnp.transpose(x, (0, 2, 1))
    # conv_general_dilated (unlike jnp.einsum elsewhere in this codebase) requires exact dtype
    # match and does NOT do implicit type promotion — cast the weight to x's dtype so a
    # fp32-promoted conv weight (utils.py::promote_trainable_to_fp32, trained via
    # `.*embed_proj_conv.*`) still works against bf16 activations, per that function's own
    # contract ("cast to bf16 only inside the forward pass").
    weight = weight.astype(x_t.dtype)
    # conv_general_dilated expects: lhs=(batch, in_channels, spatial), rhs=(out_channels, in_channels, kernel)
    out = lax.conv_general_dilated(
        x_t, weight,
        window_strides=(stride,),
        padding='VALID',
        dimension_numbers=('NCH', 'IOH', 'NCH'),
    )
    # Add bias: (out_channels,) -> (1, out_channels, 1)
    out = out + bias[None, :, None].astype(out.dtype)
    # Transpose back to (batch, new_seq_len, channels)
    return jnp.transpose(out, (0, 2, 1))


def pool_pad_mask(pad_mask, kernel_size, stride):
    """Recompute pad_mask after conv by max-pooling with the same kernel/stride.
    
    Any valid token (1) in the kernel window makes the output position valid.
    
    Args:
        pad_mask: (batch, seq_len) — binary mask
        kernel_size: int
        stride: int
    
    Returns:
        (batch, new_seq_len) — binary mask after pooling
    """
    # Expand to (batch, 1, seq_len) for reduce_window
    pm = pad_mask[:, None, :].astype(jnp.float32)
    pooled = lax.reduce_window(
        pm,
        init_value=0.0,
        computation=lax.max,
        window_dimensions=(1, 1, kernel_size),
        window_strides=(1, 1, stride),
        padding='VALID',
    )
    return pooled[:, 0, :].astype(pad_mask.dtype)
