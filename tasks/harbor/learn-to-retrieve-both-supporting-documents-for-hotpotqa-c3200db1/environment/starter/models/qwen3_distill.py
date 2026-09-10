"""
Two-component distillation model: frozen teacher (base qwen3) + trainable student (qwen3_mem_embed).

The teacher sees [doc | question | answer] as a flat sequence.
The student sees [question | answer] with docs routed through the embedding model into memory.
KL divergence on answer-token logits drives the student to learn from the teacher.
"""
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass

from .qwen3 import Model, forward as qwen3_forward, create_mask, load as load_qwen3
from .qwen3_mem_embed import forward as mem_embed_forward, load as load_mem_embed
from .utils import merge_weights, split_weights, merge_configs
from .output import ModelOutput


def forward(cfg, x, weights, pad_mask=None, kv=None, pos=0, collect_aux=False):
    """
    Forward pass for distillation model.

    Args:
        cfg: Merged config with 'teacher', 'student' keys
        x: Dict with 'batch', 'docs', 'teacher_batch'
        weights: Merged weights with 'teacher.' and 'student.' prefixed keys
        pad_mask: Dict with 'batch_mask', 'docs_mask', 'teacher_mask', 'pos_doc_mask'
        kv: KV cache (unused during training)
        pos: Position offset
        collect_aux: Whether to collect auxiliary data

    Returns:
        ModelOutput with student logits and teacher/student logits in aux
    """
    teacher_weights, student_weights = split_weights(weights, ['teacher', 'student'])
    teacher_cfg = cfg['teacher']
    student_cfg = cfg['student']

    has_teacher_batch = isinstance(x, dict) and "teacher_batch" in x

    # Student forward (qwen3_mem_embed)
    student_x = {"batch": x["batch"], "docs": x["docs"]}
    student_pad_mask = {
        "batch_mask": pad_mask["batch_mask"],
        "docs_mask": pad_mask["docs_mask"],
    }
    if "pos_doc_mask" in pad_mask:
        student_pad_mask["pos_doc_mask"] = pad_mask["pos_doc_mask"]

    student_output = mem_embed_forward(student_cfg, student_x, student_weights, pad_mask=student_pad_mask, collect_aux=collect_aux)
    student_logits = student_output.logits  # [B, seq_len, V]

    aux_data = student_output.aux if student_output.aux is not None else {}

    # Teacher forward (base qwen3, frozen) — only when teacher_batch is provided
    if has_teacher_batch:
        teacher_batch = x["teacher_batch"]
        teacher_mask = pad_mask["teacher_mask"].astype(jnp.bool_)
        teacher_output = qwen3_forward(teacher_cfg, teacher_batch, teacher_weights, pad_mask=teacher_mask)
        teacher_logits = jax.lax.stop_gradient(teacher_output.logits)  # [B, teacher_seq_len, V]

        # Pass full teacher logits; alignment handled in distillation_loss using distillation masks
        aux_data["teacher_logits"] = teacher_logits
        aux_data["student_logits"] = student_logits

    return ModelOutput(
        logits=student_logits,
        kv=None,
        aux=aux_data,
    )


def load(cfg, tp_devices=1, hf_ckpt_dir='~/weights/huggingface'):
    """
    Load distillation model with frozen teacher and trainable student.

    Args:
        cfg: Config with teacher_model, main_model, memory, embed_model sections
        tp_devices: Tensor parallel device count
        hf_ckpt_dir: HuggingFace checkpoint directory
    """
    # Load teacher (base qwen3, will be frozen via trainable_params config)
    teacher_model_id = cfg.teacher_model.model_id
    teacher_model = load_qwen3(
        teacher_model_id,
        tp_devices=tp_devices,
        load_weights=cfg.teacher_model.load_weights,
        hf_ckpt_dir=hf_ckpt_dir,
        mask_type=cfg.teacher_model.get("mask_type", "causal"),
    )

    # Load student (qwen3_mem_embed)
    from .qwen3_mem_embed import load as load_mem_embed_model
    student_model = load_mem_embed_model(cfg, tp_devices=tp_devices, hf_ckpt_dir=hf_ckpt_dir)

    # Merge weights under 'teacher' and 'student' prefixes
    model_weights = merge_weights(
        ["teacher", "student"],
        [teacher_model.weights, student_model.weights]
    )

    # Merge configs
    model_cfg = merge_configs(
        ["teacher", "student"],
        [teacher_model.cfg, student_model.cfg]
    )

    model_forward = partial(forward, model_cfg)

    return Model(
        weights=model_weights,
        forward=model_forward,
        init_kv=student_model.init_kv,
        tokenizer=student_model.tokenizer,
        cfg=model_cfg,
    )


def init(cfg, tp_devices):
    model = load(cfg, tp_devices=tp_devices)
    return model
