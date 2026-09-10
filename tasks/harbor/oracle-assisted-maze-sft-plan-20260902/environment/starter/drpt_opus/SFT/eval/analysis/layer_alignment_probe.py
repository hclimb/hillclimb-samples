#!/usr/bin/env python
"""Q2: is a per-layer-group weight w_l derivable from target gradients alone?

One forward+backward sweep at the BASE checkpoint — no training, no fine-tuned
weights. A shared base means any difference between settings comes only from the
target set, which is exactly the claim under test: does "IF uses these layers,
math uses those" actually separate?

Per layer group g it reports four normalised profiles plus two diagnostics:

  energy_raw     ||grad_g L_target||^2                      / sum over groups
  energy_opt     ||P^(1/2) grad_g L_target||^2              / sum over groups
  align_raw      mean_i |<grad_g L_i, grad_g L_target>|     / sum over groups
  align_opt      mean_i |<P grad_g L_i, grad_g L_target>|   / sum over groups
  cosine         mean_i cos(grad_g L_i, grad_g L_target)    (scale-free)
  energy_per_param  energy_raw divided by the group's parameter count

P is AdamW's diagonal preconditioner 1/(sqrt(vhat) + eps). A base model has no
optimizer state, so vhat is estimated as the mean of squared *batch* gradients
over the first `--windows` logical windows of the pinned candidate order — the
stationary value AdamW's exp_avg_sq converges to under that data.

Qwen3 ties the embedding and the LM head to one tensor. The probe unties them
(an exact-value clone) so the two call sites get separate gradients and the
requested Embedding / LM head rows are genuinely separate. Untying changes no
forward value and nothing is ever stepped.

Usage:
  python -m SFT.eval.analysis.layer_alignment_probe --setting inst_if
  python -m SFT.eval.analysis.layer_alignment_probe --all-settings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from SFT.data.dolci32k.artifacts import resolve_current_build, validate_build
from SFT.data.dolci32k.profile import (
    DEFAULT_SIZES,
    MAX_SEQ_LEN,
    MODEL_PROFILES,
    SETTINGS,
    TOKENIZER_USE_FAST,
    artifact_relative_path,
)
from SFT.eval.analysis.campaign_io import DEFAULT_CAMPAIGN, results_root

DEPTH_BANDS = ("early", "mid", "late")


def group_of(name: str, num_layers: int) -> str:
    """Map a parameter name to its layer group.

    Decoder blocks are split into depth thirds so the profile can answer
    "which *part* of the stack", not merely "attention or MLP".
    """
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "embedding"
    parts = name.split(".")
    index = None
    for position, token in enumerate(parts):
        if token == "layers" and position + 1 < len(parts):
            index = int(parts[position + 1])
            break
    if index is None:
        return "norm_other"
    band = DEPTH_BANDS[min(int(index * len(DEPTH_BANDS) / num_layers), len(DEPTH_BANDS) - 1)]
    if ".self_attn." in name:
        return f"attn_{band}"
    if ".mlp." in name:
        return f"mlp_{band}"
    return "norm_other"


def group_order(num_layers: int) -> List[str]:
    order = ["embedding"]
    for band in DEPTH_BANDS:
        order.append(f"attn_{band}")
    for band in DEPTH_BANDS:
        order.append(f"mlp_{band}")
    order += ["norm_other", "lm_head"]
    return order


def load_model_and_tokenizer(model_profile: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_PROFILES[model_profile]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_name"],
        revision=spec["tokenizer_revision"],
        use_fast=TOKENIZER_USE_FAST,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_name_or_path"],
        revision=spec["model_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Untie so the embedding and the LM head accumulate separate gradients.
    # Values are identical, so every forward is unchanged.
    if getattr(model.config, "tie_word_embeddings", False):
        import torch.nn as nn

        source = model.get_input_embeddings().weight
        model.lm_head.weight = nn.Parameter(source.detach().clone())
        model.config.tie_word_embeddings = False
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, tokenizer


def load_split(path: Path, tokenizer, max_length: int, expected: Optional[int] = None):
    from SFT.data.get_val_dataset import get_messages_file_dataset

    return get_messages_file_dataset(
        str(path), tokenizer, max_length, expected_count=expected
    )


def collate(rows, pad_id: int, device: torch.device):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids, labels, mask = [], [], []
    for row in rows:
        ids = list(row["input_ids"])
        lab = list(row["labels"])
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(mask, device=device),
    }


def supervised_tokens(labels: torch.Tensor) -> torch.Tensor:
    """Causal LM predicts labels[:, 1:], so count those positions only."""
    return (labels[:, 1:] != -100).sum()


def zero_grads(model) -> None:
    model.zero_grad(set_to_none=True)


def flat_grads(model) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def accumulate_square(model, buffer: Dict[str, torch.Tensor]) -> None:
    for name, grad in flat_grads(model).items():
        value = grad.detach().to(torch.float32)
        if name in buffer:
            buffer[name].add_(value.square())
        else:
            buffer[name] = value.square()


def token_weighted_backward(model, batch, weight: float) -> float:
    outputs = model(**batch)
    loss = outputs.loss * weight
    loss.backward()
    return float(outputs.loss.detach().float().cpu())


def estimate_vhat(model, rows, pad_id, device, window: int, windows: int) -> Dict[str, torch.Tensor]:
    """Mean squared *batch* gradient over `windows` logical windows.

    AdamW's exp_avg_sq tracks the second moment of the gradients it is actually
    fed, which are batch gradients over one logical window — not per-example
    gradients. Estimating it any other way would precondition with the wrong
    scale.
    """
    buffer: Dict[str, torch.Tensor] = {}
    used = 0
    for index in range(windows):
        chunk = rows[index * window:(index + 1) * window]
        if len(chunk) < window:
            break
        zero_grads(model)
        total = sum(
            int(supervised_tokens(torch.tensor([row["labels"]])))
            for row in chunk
        )
        if total <= 0:
            continue
        for start in range(0, len(chunk), 2):
            micro = collate(chunk[start:start + 2], pad_id, device)
            share = float(supervised_tokens(micro["labels"])) / total
            token_weighted_backward(model, micro, share)
        accumulate_square(model, buffer)
        used += 1
    if not used:
        raise RuntimeError("vhat estimation consumed no complete window")
    for tensor in buffer.values():
        tensor.div_(used)
    zero_grads(model)
    return buffer


def target_gradient(model, rows, pad_id, device, microbatch: int) -> Tuple[Dict[str, torch.Tensor], float]:
    """Gradient of the token-mean loss over the whole target split."""
    zero_grads(model)
    total = sum(
        int(supervised_tokens(torch.tensor([row["labels"]]))) for row in rows
    )
    if total <= 0:
        raise RuntimeError("target split has no supervised tokens")
    loss_sum = 0.0
    for start in range(0, len(rows), microbatch):
        batch = collate(rows[start:start + microbatch], pad_id, device)
        share = float(supervised_tokens(batch["labels"])) / total
        loss_sum += token_weighted_backward(model, batch, share) * share
    grads = {
        name: grad.detach().to(torch.float32).clone()
        for name, grad in flat_grads(model).items()
    }
    zero_grads(model)
    return grads, loss_sum


def probe_setting(
    setting: str,
    build: Path,
    model_profile: str,
    device: torch.device,
    windows: int,
    max_length: int,
    split: str,
) -> dict:
    spec = SETTINGS[setting]
    pool, target = str(spec["general_pool"]), str(spec["target"])

    model, tokenizer = load_model_and_tokenizer(model_profile, device)
    num_layers = int(model.config.num_hidden_layers)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    target_rows = list(
        load_split(
            build / artifact_relative_path("targets", f"{target}/{split}"),
            tokenizer,
            max_length,
            expected=(
                DEFAULT_SIZES.target_val if split == "val" else DEFAULT_SIZES.target_grad
            ),
        )
    )
    train_path = build / artifact_relative_path("general", f"{pool}/train")
    pool_rows = load_split(
        train_path, tokenizer, max_length, expected=DEFAULT_SIZES.general_train
    )
    # The candidate order stores stable IDs, not row numbers, so resolve them
    # through the pool's own file order exactly as train.py does. Keep the
    # domain labels alongside, so per-candidate alignment can be aggregated by
    # domain without going back through lift ratios.
    row_by_id: Dict[str, int] = {}
    row_meta: List[dict] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            row_by_id[str(record["id"])] = index
            row_meta.append({
                "id": str(record["id"]),
                "domain": str(record.get("domain", "unknown")),
                "source_dataset": str(record.get("source_dataset", "unknown")),
            })
    order_path = build / artifact_relative_path("candidate_orders", pool)
    candidate_indices = []
    with order_path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position >= windows * 16:
                break
            record = json.loads(line)
            if record.get("position") != position:
                raise RuntimeError(
                    f"candidate-order position mismatch at row {position}"
                )
            candidate_indices.append(row_by_id[str(record["id"])])
    candidate_rows = [pool_rows[i] for i in candidate_indices]
    candidate_meta = [row_meta[i] for i in candidate_indices]

    started = time.time()
    vhat = estimate_vhat(model, candidate_rows, pad_id, device, 16, windows)
    eps = 1e-8
    precond = {name: value.sqrt().add_(eps).reciprocal_() for name, value in vhat.items()}
    del vhat
    torch.cuda.empty_cache()

    target_grad, target_loss = target_gradient(model, target_rows, pad_id, device, 2)

    groups = group_order(num_layers)
    param_counts = {name: 0 for name in groups}
    for name, parameter in model.named_parameters():
        param_counts[group_of(name, num_layers)] += parameter.numel()

    energy_raw = {name: 0.0 for name in groups}
    energy_opt = {name: 0.0 for name in groups}
    target_norm_sq = {name: 0.0 for name in groups}
    for name, grad in target_grad.items():
        key = group_of(name, num_layers)
        square = grad.square()
        energy_raw[key] += float(square.sum())
        target_norm_sq[key] += float(square.sum())
        if name in precond:
            energy_opt[key] += float((square * precond[name]).sum())

    align_raw = {name: 0.0 for name in groups}
    align_opt = {name: 0.0 for name in groups}
    cosine_sum = {name: 0.0 for name in groups}
    cosine_n = {name: 0 for name in groups}
    # Whole-model target norms, for the scale-free per-candidate cosines.
    target_norm_total = sum(target_norm_sq.values())
    target_norm_total_p = sum(energy_opt.values())
    candidates: List[dict] = []

    for row, meta in zip(candidate_rows, candidate_meta):
        zero_grads(model)
        batch = collate([row], pad_id, device)
        # Token-SUM loss: proportional to this example's contribution to the
        # window gradient. A single global constant differs from the trainer's
        # 1/T_window, which cancels when groups are normalised into shares.
        tokens = float(supervised_tokens(batch["labels"]))
        if tokens <= 0:
            continue
        outputs = model(**batch)
        example_loss = float(outputs.loss.detach().float().cpu())
        (outputs.loss * tokens).backward()

        per_group_dot = {name: 0.0 for name in groups}
        per_group_dot_p = {name: 0.0 for name in groups}
        per_group_norm = {name: 0.0 for name in groups}
        per_group_norm_p = {name: 0.0 for name in groups}
        for name, grad in flat_grads(model).items():
            if name not in target_grad:
                continue
            key = group_of(name, num_layers)
            value = grad.detach().to(torch.float32)
            reference = target_grad[name]
            square = value.square()
            per_group_dot[key] += float((value * reference).sum())
            per_group_norm[key] += float(square.sum())
            if name in precond:
                scale = precond[name]
                per_group_dot_p[key] += float((value * scale * reference).sum())
                per_group_norm_p[key] += float((square * scale).sum())
        for key in groups:
            align_raw[key] += abs(per_group_dot[key])
            align_opt[key] += abs(per_group_dot_p[key])
            denominator = (per_group_norm[key] * target_norm_sq[key]) ** 0.5
            if denominator > 0:
                cosine_sum[key] += per_group_dot[key] / denominator
                cosine_n[key] += 1

        # Whole-model, scale-free alignment. cos_raw is the natural absolute
        # reading of "how close is this example's gradient to the target
        # direction"; cos_opt is the same in AdamW's geometry, i.e. the
        # scale-free version of the OptA score the selector actually ranks on.
        dot = sum(per_group_dot.values())
        dot_p = sum(per_group_dot_p.values())
        norm = sum(per_group_norm.values())
        norm_p = sum(per_group_norm_p.values())
        candidates.append({
            "id": meta["id"],
            "domain": meta["domain"],
            "source_dataset": meta["source_dataset"],
            "tokens": tokens,
            "loss": example_loss,
            "grad_norm": norm ** 0.5,
            "dot_raw": dot,
            "dot_opt": dot_p,
            "cos_raw": dot / ((norm * target_norm_total) ** 0.5)
            if norm > 0 and target_norm_total > 0 else float("nan"),
            "cos_opt": dot_p / ((norm_p * target_norm_total_p) ** 0.5)
            if norm_p > 0 and target_norm_total_p > 0 else float("nan"),
        })

    count = max(len(candidate_rows), 1)
    payload = {
        "setting": setting,
        "target": target,
        "general_pool": pool,
        "model_profile": model_profile,
        "split": split,
        "target_examples": len(target_rows),
        "candidate_examples": len(candidate_rows),
        "windows": windows,
        "num_layers": num_layers,
        "target_loss": target_loss,
        "elapsed_seconds": time.time() - started,
        "groups": groups,
        "param_counts": param_counts,
        "energy_raw": energy_raw,
        "energy_opt": energy_opt,
        "align_raw": {k: v / count for k, v in align_raw.items()},
        "align_opt": {k: v / count for k, v in align_opt.items()},
        "cosine": {
            k: (cosine_sum[k] / cosine_n[k]) if cosine_n[k] else float("nan")
            for k in groups
        },
        "candidates": candidates,
    }

    del model, target_grad, precond
    torch.cuda.empty_cache()
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--setting", default=None, help="one setting id")
    parser.add_argument("--all-settings", action="store_true")
    parser.add_argument("--model-profile", default="qwen3_1_7b")
    parser.add_argument("--data-dir", default="SFT/data")
    parser.add_argument("--artifact-build-id", default=None)
    parser.add_argument(
        "--split",
        default="val",
        choices=("val", "grad"),
        help="target split: val = target_val (128), grad = target_grad (64)",
    )
    parser.add_argument("--windows", type=int, default=8, help="logical windows of 16")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not args.all_settings and not args.setting:
        parser.error("pass --setting ID or --all-settings")
    settings = list(SETTINGS) if args.all_settings else [args.setting]
    for setting in settings:
        if setting not in SETTINGS:
            parser.error(f"unknown setting {setting!r}; expected one of {tuple(SETTINGS)}")

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA device")
    device = torch.device("cuda")

    build = resolve_current_build(args.data_dir, args.artifact_build_id)
    validate_build(build)

    out_dir = (
        results_root(args.campaign, Path(args.out) if args.out else None)
        / "layer_alignment"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for setting in settings:
        print(f"[probe] {setting} split={args.split} windows={args.windows}", flush=True)
        payload = probe_setting(
            setting,
            build,
            args.model_profile,
            device,
            args.windows,
            args.max_length,
            args.split,
        )
        destination = out_dir / f"{setting}_{args.split}_probe.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[probe] {setting} done in {payload['elapsed_seconds']:.0f}s -> {destination}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
