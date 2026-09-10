import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from .registry import register_aux_loss


@register_aux_loss("distillation_loss")
def compute_distillation_loss(aux_data, mask, input_mask, inputs, top_k=20, temperature=1.0, **kwargs):
    """
    Forward KL divergence (mean-seeking) between teacher and student logits.
    Uses argsort to statically align the correct answer tokens dynamically extracted via distill_masks.

    Args:
        aux_data: Dict with 'teacher_logits' and 'student_logits' (unaligned)
        mask: Loss mask [B, T] (unused for main logic, relying on distill masks)
        input_mask: Dict with 'student_distill_mask' and 'teacher_distill_mask'
        inputs: Input tokens dict (unused)
        top_k: Number of top tokens to compute KL over (currently computes over all vocab)
        temperature: Temperature for softmax scaling

    Returns:
        Scalar KL divergence loss
    """
    teacher_logits = aux_data.get("teacher_logits")
    student_logits = aux_data.get("student_logits")
    
    teacher_mask = input_mask.get("teacher_distill_mask")
    student_mask = input_mask.get("student_distill_mask")

    if teacher_logits is None or student_logits is None or teacher_mask is None or student_mask is None:
        return 0.0

    # Scale by temperature; cast to float32 to avoid bfloat16 sum/exp overflow
    teacher_scaled = teacher_logits.astype(jnp.float32) / temperature
    student_scaled = student_logits.astype(jnp.float32) / temperature

    B, T_student = student_mask.shape
    _, T_teacher = teacher_mask.shape

    # Stable sort to bring 1s to the front, maintaining order for exact alignment
    student_sort_keys = -student_mask * 1e11 + jnp.arange(T_student)
    teacher_sort_keys = -teacher_mask * 1e11 + jnp.arange(T_teacher)

    student_indices = jnp.argsort(student_sort_keys, axis=1) # [B, T_student]
    teacher_indices = jnp.argsort(teacher_sort_keys, axis=1) # [B, T_teacher]

    # Gather the logits and masks along the sequence axis
    student_indices_expanded = jnp.expand_dims(student_indices, -1)
    aligned_student_logits = jnp.take_along_axis(student_scaled, student_indices_expanded, axis=1)
    aligned_student_mask = jnp.take_along_axis(student_mask, student_indices, axis=1)

    teacher_indices_expanded = jnp.expand_dims(teacher_indices, -1)
    aligned_teacher_logits = jnp.take_along_axis(teacher_scaled, teacher_indices_expanded, axis=1)
    
    # Both have answer tokens placed symmetrically at indices 0 to N-1
    # Truncate aligned teacher to match aligned student shape. The 1s will safely fit in T_student.
    aligned_teacher_logits = aligned_teacher_logits[:, :T_student, :]

    # Forward KL: KL(Teacher || Student) — mean-seeking
    # Formula: P(teacher) * (log P(teacher) - log P(student))
    student_log_probs = jax.nn.log_softmax(aligned_student_logits, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(aligned_teacher_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)

    # Use jnp.where to avoid 0 * -inf = NaN when student_probs is near zero or valid entries are none
    kl_per_token = jnp.where(
        teacher_probs > 0,
        teacher_probs * (teacher_log_probs - student_log_probs),
        0.0
    )
    kl = kl_per_token.sum(axis=-1)  # [B, T_student]

    # Apply loss mask (answer tokens only exactly at front) and average
    masked_kl = kl * aligned_student_mask
    loss = masked_kl.sum() / (aligned_student_mask.sum() + 1e-9)

    return loss * (temperature ** 2)
