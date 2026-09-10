#!/usr/bin/env python
"""
Empirical weak-submodularity checks for aggregate-then-polarize Muon selection.

The set function tested here is

    F(S) = <G_target, rho * Polar_eps(Q_S)>
    Q_S = mu^2 * M_t + (1 - mu^2) * mean_{i in S} G_i,

where G_i are per-sample gradient matrices for one Muon-applied parameter
group/layer.  The script can run synthetic toy experiments or load saved
gradient tensors.

This is an empirical diagnostic, not a proof.  In fact, the set function is
not submodular in full generality because it uses a cardinality-normalized
average and a nonlinear polar map.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class ViolationStats:
    trials: int
    violations: int
    violation_rate: float
    mean_slack: float
    min_slack: float
    p05_slack: float
    p50_slack: float
    worst_violation: Optional[Dict[str, Any]]


@dataclass
class RatioStats:
    trials: int
    valid: int
    skipped_nonpositive_denominator: int
    gamma_min: Optional[float]
    gamma_p01: Optional[float]
    gamma_p05: Optional[float]
    gamma_p50: Optional[float]
    gamma_mean: Optional[float]
    gamma_clipped_min: Optional[float]


@dataclass
class MonotonicityStats:
    trials: int
    negative_marginals: int
    negative_rate: float
    mean_marginal: float
    min_marginal: float
    p05_marginal: float
    p50_marginal: float


@dataclass
class GreedyStats:
    k: int
    greedy_set: List[int]
    greedy_value: float
    greedy_gain: float
    optimum_set: Optional[List[int]]
    optimum_value: Optional[float]
    optimum_gain: Optional[float]
    gain_ratio: Optional[float]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_tensor_from_file(path: str, key: Optional[str] = None) -> Tensor:
    """Load a tensor/array from .pt/.pth/.npy/.npz files."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = _torch_load(p)
        if key is not None:
            if not isinstance(obj, dict):
                raise ValueError(f"{path} is not a dict, so --*-key cannot be used")
            obj = obj[key]
    elif suffix == ".npy":
        obj = np.load(p, allow_pickle=False)
    elif suffix == ".npz":
        npz = np.load(p, allow_pickle=False)
        if key is None:
            if len(npz.files) != 1:
                raise ValueError(
                    f"{path} has keys {npz.files}; pass the desired --*-key"
                )
            key = npz.files[0]
        obj = npz[key]
    else:
        raise ValueError(f"Unsupported file suffix for {path}")

    if isinstance(obj, Tensor):
        return obj.detach().cpu()
    return torch.as_tensor(obj)


def ensure_gradient_matrices(grads: Tensor, matrix_shape: Optional[Tuple[int, int]]) -> Tensor:
    """Return gradients as [n, rows, cols]."""
    grads = grads.detach().to(torch.float32)
    if grads.ndim == 3:
        return grads
    if grads.ndim == 2 and matrix_shape is not None:
        rows, cols = matrix_shape
        if grads.shape[1] != rows * cols:
            raise ValueError(
                f"Flat gradient dimension {grads.shape[1]} does not match "
                f"--matrix-shape {rows} {cols}"
            )
        return grads.reshape(grads.shape[0], rows, cols)
    raise ValueError(
        "Gradient tensor must have shape [n, rows, cols], or [n, rows*cols] "
        "with --matrix-shape rows cols."
    )


def crop_matrices(
    gradients: Tensor,
    target: Tensor,
    momentum: Optional[Tensor],
    crop_shape: Optional[Tuple[int, int]],
    seed: int,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict[str, Any]]]:
    if crop_shape is None:
        return gradients, target, momentum, None
    crop_rows, crop_cols = crop_shape
    rows, cols = gradients.shape[-2:]
    if crop_rows <= 0 or crop_cols <= 0:
        raise ValueError("--matrix-crop dimensions must be positive")
    if crop_rows > rows or crop_cols > cols:
        raise ValueError(
            f"--matrix-crop {crop_rows} {crop_cols} exceeds matrix shape {rows}x{cols}"
        )
    rng = random.Random(seed + 911)
    row_start = rng.randint(0, rows - crop_rows)
    col_start = rng.randint(0, cols - crop_cols)
    row_slice = slice(row_start, row_start + crop_rows)
    col_slice = slice(col_start, col_start + crop_cols)
    cropped_momentum = None
    if momentum is not None:
        cropped_momentum = momentum[row_slice, col_slice]
    meta = {
        "matrix_crop": [crop_rows, crop_cols],
        "matrix_crop_row_start": row_start,
        "matrix_crop_col_start": col_start,
        "original_matrix_shape": [rows, cols],
    }
    return (
        gradients[:, row_slice, col_slice],
        target[row_slice, col_slice],
        cropped_momentum,
        meta,
    )


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def move_batch_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def keep_model_input_columns(dataset):
    """Drop raw metadata columns before passing examples to a HF data collator."""
    keep = {"input_ids", "attention_mask", "labels"}
    if not hasattr(dataset, "column_names"):
        return dataset
    missing = keep.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required token columns: {sorted(missing)}")
    remove = [name for name in dataset.column_names if name not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)
    return dataset


def select_real_sft_parameter(
    model: torch.nn.Module,
    layer_name: Optional[str],
) -> Tuple[str, torch.nn.Parameter]:
    named_params = dict(model.named_parameters())
    if layer_name is not None:
        candidates = [layer_name]
        if not layer_name.endswith(".weight"):
            candidates.append(f"{layer_name}.weight")
        for name in candidates:
            param = named_params.get(name)
            if param is not None:
                if param.ndim != 2:
                    raise ValueError(f"Selected parameter {name} is not a matrix")
                return name, param
        raise ValueError(f"Could not find --real-layer-name '{layer_name}' in model parameters")

    skip = ("embed", "embedding", "lm_head", "norm", "ln_", "bias")
    prefer = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "self_attn", "attn", "mlp",
    )
    fallback: Optional[Tuple[str, torch.nn.Parameter]] = None
    for name, param in model.named_parameters():
        lowered = name.lower()
        if param.ndim != 2 or not name.endswith(".weight"):
            continue
        if any(marker in lowered for marker in skip):
            continue
        if fallback is None:
            fallback = (name, param)
        if any(marker in lowered for marker in prefer):
            return name, param
    if fallback is not None:
        return fallback
    raise ValueError("Could not auto-select a 2D Muon-style matrix parameter")


def collect_real_sft_gradients(args: argparse.Namespace) -> Tuple[Tensor, Tensor, Optional[Tensor], Dict[str, Any]]:
    """Collect per-sample gradients from a small real SFT dataset subset."""
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

    from SFT.data.get_train_dataset import (
        _get_default_train_files,
        encode_data,
        get_train_files_for_dataset,
        load_raw_dataset,
    )
    from SFT.data.get_val_dataset import ensure_chat_template, get_dataset
    from SFT.train.model_arguments import add_padding_to_tokenizer

    if args.grad_file is not None:
        raise ValueError("--real-sft and --grad-file are mutually exclusive")
    if args.model_name_or_path is None:
        raise ValueError("--real-sft requires --model-name-or-path")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = None
    if args.real_torch_dtype == "float32":
        dtype = torch.float32
    elif args.real_torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.real_torch_dtype == "float16":
        dtype = torch.float16
    elif args.real_torch_dtype == "auto" and device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[real-sft] Loading tokenizer: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.hf_cache_dir,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_chat_template(tokenizer)
    add_padding_to_tokenizer(tokenizer)

    train_dataset_names = parse_csv_list(args.train_dataset_names)
    if train_dataset_names is not None:
        train_files: List[str] = []
        for name in train_dataset_names:
            train_files.extend(get_train_files_for_dataset(args.data_dir, name))
    else:
        train_files = _get_default_train_files(args.data_dir, args.analysis_dataset)

    print(
        f"[real-sft] Loading train_n={args.train_n} from {train_files}",
        flush=True,
    )
    raw_train = load_raw_dataset(
        train_files,
        sample_size=args.train_n,
        seed=args.seed,
    )
    train_dataset = encode_data(
        raw_train,
        tokenizer,
        args.max_seq_length,
        processing_num_workers=(
            None if args.preprocessing_num_workers <= 1 else args.preprocessing_num_workers
        ),
        overwrite_cache=args.overwrite_cache,
    )
    train_dataset = keep_model_input_columns(train_dataset)

    print(
        f"[real-sft] Loading val_n={args.val_n} task={args.analysis_dataset} split={args.eval_split}",
        flush=True,
    )
    val_dataset = get_dataset(
        args.analysis_dataset,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        split=args.eval_split,
        k=args.val_n,
        seed=args.seed + 1,
    )
    val_dataset = keep_model_input_columns(val_dataset)

    print(f"[real-sft] Loading model: {args.model_name_or_path}", flush=True)
    model_kwargs: Dict[str, Any] = {
        "cache_dir": args.hf_cache_dir,
        "local_files_only": not args.allow_download,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()

    param_name, target_param = select_real_sft_parameter(model, args.real_layer_name)
    for param in model.parameters():
        param.requires_grad_(False)
    target_param.requires_grad_(True)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")

    def grad_for_examples(dataset, count: int, label: str) -> Tensor:
        loader = DataLoader(dataset.select(range(count)), batch_size=1, collate_fn=collator)
        grads: List[Tensor] = []
        for idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Non-finite loss while collecting {label} gradient {idx}")
            loss.backward()
            if target_param.grad is None:
                raise RuntimeError(f"No gradient produced for {param_name}")
            grads.append(target_param.grad.detach().to(torch.float32).cpu().clone())
            if (idx + 1) % max(1, args.real_log_every) == 0 or idx + 1 == count:
                print(f"[real-sft] collected {label} gradients {idx + 1}/{count}", flush=True)
        return torch.stack(grads, dim=0)

    train_grads = grad_for_examples(train_dataset, args.train_n, "train")
    val_grads = grad_for_examples(val_dataset, args.val_n, "val")
    target = val_grads.mean(dim=0)
    momentum = None

    meta = {
        "source": "real_sft",
        "model_name_or_path": args.model_name_or_path,
        "data_dir": args.data_dir,
        "analysis_dataset": args.analysis_dataset,
        "train_dataset_names": train_dataset_names,
        "eval_split": args.eval_split,
        "train_n": args.train_n,
        "val_n": args.val_n,
        "max_seq_length": args.max_seq_length,
        "param_name": param_name,
        "param_shape": list(train_grads.shape[1:]),
        "target": "mean_validation_gradient",
        "momentum": "zero" if args.momentum_file is None else "provided",
    }

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return train_grads, target, momentum, meta


def polar_svd_eps(matrix: Tensor, eps: float = 1e-7) -> Tensor:
    """Regularized rectangular polar factor via inverse square root."""
    x = matrix.to(torch.float32)
    rows, cols = x.shape
    if rows >= cols:
        gram = x.transpose(0, 1) @ x
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(eps))) @ evecs.transpose(0, 1)
        return x @ inv_sqrt

    gram = x @ x.transpose(0, 1)
    evals, evecs = torch.linalg.eigh(gram)
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(eps))) @ evecs.transpose(0, 1)
    return inv_sqrt @ x


def polar_muon_ns(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Muon-style Newton-Schulz zeroth-power matrix normalization.

    This mirrors drpt.optimizer.zeropower_via_newton_schulz so the empirical
    test matches the local Muon update geometry.
    """
    if steps <= 0:
        return matrix

    orig_dtype = matrix.dtype
    x = matrix.to(torch.float32)
    norm = x.norm()
    if norm <= eps:
        return torch.zeros_like(matrix)
    x = x / norm.clamp_min(eps)

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.transpose(0, 1)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(orig_dtype)


def mask_to_indices(mask: int) -> List[int]:
    indices: List[int] = []
    while mask:
        bit = mask & -mask
        indices.append(bit.bit_length() - 1)
        mask ^= bit
    return indices


def indices_to_mask(indices: Iterable[int]) -> int:
    mask = 0
    for idx in indices:
        mask |= 1 << int(idx)
    return mask


def iter_submasks(mask: int) -> Iterable[int]:
    sub = mask
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q))


class AggregatePolarSetFunction:
    def __init__(
        self,
        gradients: Tensor,
        target: Tensor,
        momentum: Optional[Tensor] = None,
        mu: float = 0.95,
        rho: float = 1.0,
        polar: str = "ns",
        ns_steps: int = 5,
        eps: float = 1e-7,
        normalize_target: bool = True,
        device: str = "cpu",
    ) -> None:
        if gradients.ndim != 3:
            raise ValueError("gradients must have shape [n, rows, cols]")
        self.gradients = gradients.to(device=device, dtype=torch.float32)
        self.n, self.rows, self.cols = self.gradients.shape
        self.target = target.to(device=device, dtype=torch.float32)
        if self.target.shape != (self.rows, self.cols):
            raise ValueError(
                f"target shape {tuple(self.target.shape)} does not match "
                f"gradient matrices {(self.rows, self.cols)}"
            )
        if normalize_target:
            self.target = self.target / self.target.norm().clamp_min(eps)

        if momentum is None:
            self.momentum = torch.zeros((self.rows, self.cols), device=device)
        else:
            self.momentum = momentum.to(device=device, dtype=torch.float32)
            if self.momentum.shape != (self.rows, self.cols):
                raise ValueError(
                    f"momentum shape {tuple(self.momentum.shape)} does not match "
                    f"gradient matrices {(self.rows, self.cols)}"
                )

        self.mu2 = float(mu) ** 2
        self.rho = float(rho)
        self.polar = polar
        self.ns_steps = int(ns_steps)
        self.eps = float(eps)
        self._cache: Dict[int, float] = {}

    def aggregate_q(self, mask: int) -> Tensor:
        if mask == 0:
            selected_mean = torch.zeros_like(self.momentum)
        else:
            idx = mask_to_indices(mask)
            selected_mean = self.gradients[idx].mean(dim=0)
        return self.mu2 * self.momentum + (1.0 - self.mu2) * selected_mean

    def polarize(self, matrix: Tensor) -> Tensor:
        if self.polar == "ns":
            return polar_muon_ns(matrix, steps=self.ns_steps, eps=self.eps)
        if self.polar == "svd":
            return polar_svd_eps(matrix, eps=self.eps)
        raise ValueError("--polar must be one of: ns, svd")

    def value(self, mask: int) -> float:
        cached = self._cache.get(mask)
        if cached is not None:
            return cached
        q = self.aggregate_q(mask)
        update = self.rho * self.polarize(q)
        val = float(torch.sum(self.target * update).detach().cpu().item())
        self._cache[mask] = val
        return val

    def marginal(self, mask: int, item: int) -> float:
        bit = 1 << item
        if mask & bit:
            raise ValueError("item is already in the set")
        return self.value(mask | bit) - self.value(mask)


def make_target(
    gradients: Tensor,
    momentum: Optional[Tensor],
    mu: float,
    mode: str,
    eps: float,
    polar: str,
    ns_steps: int,
    seed: int,
) -> Tensor:
    rows, cols = gradients.shape[1:]
    if mode == "mean":
        return gradients.mean(dim=0)
    if mode == "random":
        gen = torch.Generator().manual_seed(seed + 100_003)
        return torch.randn((rows, cols), generator=gen)
    if mode == "first":
        return gradients[0].clone()
    if mode != "polar_full":
        raise ValueError("--target-mode must be one of: polar_full, mean, random, first")

    m = torch.zeros((rows, cols), dtype=torch.float32) if momentum is None else momentum
    q_full = (mu ** 2) * m + (1.0 - mu ** 2) * gradients.mean(dim=0)
    if polar == "ns":
        return polar_muon_ns(q_full, steps=ns_steps, eps=eps)
    return polar_svd_eps(q_full, eps=eps)


def toy_gradients(
    toy: str,
    n: int,
    rows: int,
    cols: int,
    seed: int,
    noise: float,
    signal: float,
    rank: int,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Return (gradients, target_override, momentum)."""
    gen = torch.Generator().manual_seed(seed)

    if toy == "gaussian":
        base = signal * torch.randn((rows, cols), generator=gen)
        grads = base.unsqueeze(0) + noise * torch.randn((n, rows, cols), generator=gen)
        return grads, None, None

    if toy == "low_rank":
        rank = max(1, min(rank, rows, cols))
        u = torch.randn((rank, rows), generator=gen)
        v = torch.randn((rank, cols), generator=gen)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        coeff_center = signal * torch.linspace(1.0, 0.25, rank)
        coeffs = coeff_center.unsqueeze(0) + 0.25 * torch.randn((n, rank), generator=gen)
        grads = torch.einsum("nr,ro,ri->noi", coeffs, u, v)
        grads = grads + noise * torch.randn((n, rows, cols), generator=gen)
        return grads, None, None

    if toy == "clustered":
        n_clusters = min(4, max(2, n // 4))
        centers = signal * torch.randn((n_clusters, rows, cols), generator=gen)
        assignments = torch.randint(0, n_clusters, (n,), generator=gen)
        grads = centers[assignments] + noise * torch.randn((n, rows, cols), generator=gen)
        return grads, None, None

    if toy == "sign_counterexample":
        if n < 3:
            raise ValueError("sign_counterexample requires --n >= 3")
        dim = min(rows, cols)
        eye = torch.zeros((rows, cols))
        eye[:dim, :dim] = torch.eye(dim)
        values = [-100.0, 90.0, 20.0]
        if n > 3:
            values.extend([0.0 for _ in range(n - 3)])
        grads = torch.stack([v * eye for v in values], dim=0)
        target = eye.clone()
        return grads, target, None

    raise ValueError("--toy must be one of: gaussian, low_rank, clustered, sign_counterexample")


def sample_nested_pair(n: int, rng: random.Random, max_b_size: Optional[int]) -> Tuple[int, int, int]:
    i = rng.randrange(n)
    pool = [j for j in range(n) if j != i]
    upper = len(pool) if max_b_size is None else min(max_b_size, len(pool))
    b_size = rng.randint(0, upper)
    b_indices = rng.sample(pool, b_size)
    a_size = rng.randint(0, b_size)
    a_indices = rng.sample(b_indices, a_size)
    return indices_to_mask(a_indices), indices_to_mask(b_indices), i


def sample_violation_stats(
    f: AggregatePolarSetFunction,
    trials: int,
    seed: int,
    tol: float,
    max_b_size: Optional[int],
) -> ViolationStats:
    rng = random.Random(seed)
    slacks: List[float] = []
    violations = 0
    worst: Optional[Dict[str, Any]] = None

    for _ in range(trials):
        a_mask, b_mask, i = sample_nested_pair(f.n, rng, max_b_size)
        delta_a = f.marginal(a_mask, i)
        delta_b = f.marginal(b_mask, i)
        slack = delta_a - delta_b
        slacks.append(slack)
        if slack < -tol:
            violations += 1
            severity = delta_b - delta_a
            if worst is None or severity > worst["severity"]:
                worst = {
                    "A": mask_to_indices(a_mask),
                    "B": mask_to_indices(b_mask),
                    "i": i,
                    "delta_A": delta_a,
                    "delta_B": delta_b,
                    "slack": slack,
                    "severity": severity,
                }

    return ViolationStats(
        trials=trials,
        violations=violations,
        violation_rate=violations / max(1, trials),
        mean_slack=float(np.mean(slacks)) if slacks else float("nan"),
        min_slack=float(np.min(slacks)) if slacks else float("nan"),
        p05_slack=quantile(slacks, 0.05),
        p50_slack=quantile(slacks, 0.50),
        worst_violation=worst,
    )


def exact_violation_stats(
    f: AggregatePolarSetFunction,
    tol: float,
    max_b_size: Optional[int],
) -> ViolationStats:
    slacks: List[float] = []
    violations = 0
    worst: Optional[Dict[str, Any]] = None
    full = (1 << f.n) - 1

    for b_mask in range(1 << f.n):
        if max_b_size is not None and b_mask.bit_count() > max_b_size:
            continue
        available = full ^ b_mask
        item_bits = available
        while item_bits:
            item_bit = item_bits & -item_bits
            i = item_bit.bit_length() - 1
            item_bits ^= item_bit
            delta_b = f.value(b_mask | item_bit) - f.value(b_mask)
            for a_mask in iter_submasks(b_mask):
                delta_a = f.value(a_mask | item_bit) - f.value(a_mask)
                slack = delta_a - delta_b
                slacks.append(slack)
                if slack < -tol:
                    violations += 1
                    severity = delta_b - delta_a
                    if worst is None or severity > worst["severity"]:
                        worst = {
                            "A": mask_to_indices(a_mask),
                            "B": mask_to_indices(b_mask),
                            "i": i,
                            "delta_A": delta_a,
                            "delta_B": delta_b,
                            "slack": slack,
                            "severity": severity,
                        }

    trials = len(slacks)
    return ViolationStats(
        trials=trials,
        violations=violations,
        violation_rate=violations / max(1, trials),
        mean_slack=float(np.mean(slacks)) if slacks else float("nan"),
        min_slack=float(np.min(slacks)) if slacks else float("nan"),
        p05_slack=quantile(slacks, 0.05),
        p50_slack=quantile(slacks, 0.50),
        worst_violation=worst,
    )


def sample_ratio_stats(
    f: AggregatePolarSetFunction,
    trials: int,
    seed: int,
    denominator_tol: float,
    max_a_size: Optional[int],
    max_l_size: int,
) -> RatioStats:
    rng = random.Random(seed + 17)
    ratios: List[float] = []
    skipped = 0

    for _ in range(trials):
        a_upper = f.n - 1 if max_a_size is None else min(max_a_size, f.n - 1)
        a_size = rng.randint(0, a_upper)
        a_indices = rng.sample(range(f.n), a_size)
        a_mask = indices_to_mask(a_indices)
        remaining = [j for j in range(f.n) if not (a_mask & (1 << j))]
        if not remaining:
            skipped += 1
            continue
        l_size = rng.randint(1, min(max_l_size, len(remaining)))
        l_indices = rng.sample(remaining, l_size)
        l_mask = indices_to_mask(l_indices)

        f_a = f.value(a_mask)
        denom = f.value(a_mask | l_mask) - f_a
        if denom <= denominator_tol:
            skipped += 1
            continue
        numerator = sum(f.value(a_mask | (1 << i)) - f_a for i in l_indices)
        ratios.append(numerator / denom)

    return _ratio_stats_from_values(trials, ratios, skipped)


def exact_ratio_stats(
    f: AggregatePolarSetFunction,
    denominator_tol: float,
    max_a_size: Optional[int],
    max_l_size: int,
) -> RatioStats:
    ratios: List[float] = []
    skipped = 0
    full = (1 << f.n) - 1

    for a_mask in range(1 << f.n):
        if max_a_size is not None and a_mask.bit_count() > max_a_size:
            continue
        remaining = full ^ a_mask
        for l_mask in iter_submasks(remaining):
            if l_mask == 0:
                continue
            if l_mask.bit_count() > max_l_size:
                continue
            f_a = f.value(a_mask)
            denom = f.value(a_mask | l_mask) - f_a
            if denom <= denominator_tol:
                skipped += 1
                continue
            numerator = 0.0
            bits = l_mask
            while bits:
                bit = bits & -bits
                numerator += f.value(a_mask | bit) - f_a
                bits ^= bit
            ratios.append(numerator / denom)

    return _ratio_stats_from_values(len(ratios) + skipped, ratios, skipped)


def _ratio_stats_from_values(trials: int, ratios: Sequence[float], skipped: int) -> RatioStats:
    if not ratios:
        return RatioStats(
            trials=trials,
            valid=0,
            skipped_nonpositive_denominator=skipped,
            gamma_min=None,
            gamma_p01=None,
            gamma_p05=None,
            gamma_p50=None,
            gamma_mean=None,
            gamma_clipped_min=None,
        )

    arr = np.asarray(ratios, dtype=np.float64)
    return RatioStats(
        trials=trials,
        valid=len(ratios),
        skipped_nonpositive_denominator=skipped,
        gamma_min=float(np.min(arr)),
        gamma_p01=float(np.quantile(arr, 0.01)),
        gamma_p05=float(np.quantile(arr, 0.05)),
        gamma_p50=float(np.quantile(arr, 0.50)),
        gamma_mean=float(np.mean(arr)),
        gamma_clipped_min=float(np.clip(np.min(arr), 0.0, 1.0)),
    )


def monotonicity_stats(
    f: AggregatePolarSetFunction,
    trials: int,
    seed: int,
    tol: float,
    max_set_size: Optional[int],
) -> MonotonicityStats:
    rng = random.Random(seed + 29)
    marginals: List[float] = []
    negative = 0
    for _ in range(trials):
        i = rng.randrange(f.n)
        pool = [j for j in range(f.n) if j != i]
        upper = len(pool) if max_set_size is None else min(max_set_size, len(pool))
        size = rng.randint(0, upper)
        mask = indices_to_mask(rng.sample(pool, size))
        marginal = f.marginal(mask, i)
        marginals.append(marginal)
        if marginal < -tol:
            negative += 1

    return MonotonicityStats(
        trials=trials,
        negative_marginals=negative,
        negative_rate=negative / max(1, trials),
        mean_marginal=float(np.mean(marginals)) if marginals else float("nan"),
        min_marginal=float(np.min(marginals)) if marginals else float("nan"),
        p05_marginal=quantile(marginals, 0.05),
        p50_marginal=quantile(marginals, 0.50),
    )


def greedy_and_exact_optimum(
    f: AggregatePolarSetFunction,
    k: int,
    max_opt_combinations: int,
) -> GreedyStats:
    if k < 0 or k > f.n:
        raise ValueError("--k must be between 0 and n")

    base = f.value(0)
    greedy_mask = 0
    for _ in range(k):
        best_item = None
        best_gain = -float("inf")
        for i in range(f.n):
            bit = 1 << i
            if greedy_mask & bit:
                continue
            gain = f.value(greedy_mask | bit) - f.value(greedy_mask)
            if gain > best_gain:
                best_gain = gain
                best_item = i
        if best_item is None:
            break
        greedy_mask |= 1 << best_item

    greedy_value = f.value(greedy_mask)
    greedy_gain = greedy_value - base

    opt_set: Optional[List[int]] = None
    opt_value: Optional[float] = None
    opt_gain: Optional[float] = None
    gain_ratio: Optional[float] = None

    num_combinations = math.comb(f.n, k)
    if num_combinations <= max_opt_combinations:
        best_mask = 0
        best_value = -float("inf")
        for combo in itertools.combinations(range(f.n), k):
            mask = indices_to_mask(combo)
            value = f.value(mask)
            if value > best_value:
                best_value = value
                best_mask = mask
        opt_set = mask_to_indices(best_mask)
        opt_value = best_value
        opt_gain = opt_value - base
        if opt_gain is not None and opt_gain > 1e-12:
            gain_ratio = greedy_gain / opt_gain

    return GreedyStats(
        k=k,
        greedy_set=mask_to_indices(greedy_mask),
        greedy_value=greedy_value,
        greedy_gain=greedy_gain,
        optimum_set=opt_set,
        optimum_value=opt_value,
        optimum_gain=opt_gain,
        gain_ratio=gain_ratio,
    )


def build_experiment(args: argparse.Namespace) -> Tuple[AggregatePolarSetFunction, Dict[str, Any]]:
    matrix_shape = tuple(args.matrix_shape) if args.matrix_shape is not None else None

    target_override: Optional[Tensor] = None
    real_meta: Dict[str, Any] = {}
    if args.real_sft:
        gradients, target_override, momentum, real_meta = collect_real_sft_gradients(args)
        if args.momentum_file is not None:
            momentum = load_tensor_from_file(args.momentum_file, args.momentum_key).to(torch.float32)
    elif args.grad_file is not None:
        gradients = ensure_gradient_matrices(
            load_tensor_from_file(args.grad_file, args.grad_key),
            matrix_shape,
        )
        momentum = (
            load_tensor_from_file(args.momentum_file, args.momentum_key).to(torch.float32)
            if args.momentum_file is not None
            else None
        )
        if args.target_file is not None:
            target_override = load_tensor_from_file(args.target_file, args.target_key).to(torch.float32)
    else:
        gradients, target_override, momentum = toy_gradients(
            toy=args.toy,
            n=args.n,
            rows=args.rows,
            cols=args.cols,
            seed=args.seed,
            noise=args.noise,
            signal=args.signal,
            rank=args.rank,
        )

    if args.limit_n is not None:
        gradients = gradients[: args.limit_n]

    if target_override is not None:
        target = target_override
    else:
        target = make_target(
            gradients=gradients,
            momentum=momentum,
            mu=args.mu,
            mode=args.target_mode,
            eps=args.eps,
            polar=args.polar,
            ns_steps=args.ns_steps,
            seed=args.seed,
        )

    crop_shape = tuple(args.matrix_crop) if args.matrix_crop is not None else None
    gradients, target, momentum, crop_meta = crop_matrices(
        gradients,
        target,
        momentum,
        crop_shape,
        args.seed,
    )
    if args.save_real_gradients is not None and args.real_sft:
        save_path = Path(args.save_real_gradients)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_meta = dict(real_meta)
        if crop_meta is not None:
            save_meta.update(crop_meta)
        torch.save(
            {
                "gradients": gradients.detach().cpu(),
                "target": target.detach().cpu(),
                "momentum": None if momentum is None else momentum.detach().cpu(),
                "metadata": save_meta,
            },
            save_path,
        )
        print(f"[real-sft] Saved analysis gradients to {save_path}", flush=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    f = AggregatePolarSetFunction(
        gradients=gradients,
        target=target,
        momentum=momentum,
        mu=args.mu,
        rho=args.rho,
        polar=args.polar,
        ns_steps=args.ns_steps,
        eps=args.eps,
        normalize_target=not args.no_normalize_target,
        device=device,
    )
    meta = {
        "n": f.n,
        "rows": f.rows,
        "cols": f.cols,
        "toy": None if (args.grad_file or args.real_sft) else args.toy,
        "grad_file": args.grad_file,
        "source": "real_sft" if args.real_sft else ("file" if args.grad_file else "toy"),
        "target_mode": (
            real_meta.get("target", "provided")
            if args.real_sft
            else ("provided" if target_override is not None else args.target_mode)
        ),
        "polar": args.polar,
        "ns_steps": args.ns_steps,
        "mu": args.mu,
        "rho": args.rho,
        "device": device,
    }
    meta.update(real_meta)
    if crop_meta is not None:
        meta.update(crop_meta)
    return f, meta


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    f, meta = build_experiment(args)
    max_b_size = args.max_b_size
    max_a_size = args.max_a_size
    max_monotone_size = args.max_monotone_size

    if args.exact or f.n <= args.exact_n:
        if f.n > args.exact_n and args.exact:
            raise ValueError(
                f"--exact requested with n={f.n}; increase --exact-n if intended"
            )
        violation = exact_violation_stats(f, args.tol, max_b_size)
        ratio = exact_ratio_stats(f, args.denominator_tol, max_a_size, args.max_l_size)
        exact_used = True
    else:
        violation = sample_violation_stats(
            f,
            trials=args.trials,
            seed=args.seed,
            tol=args.tol,
            max_b_size=max_b_size,
        )
        ratio = sample_ratio_stats(
            f,
            trials=args.trials,
            seed=args.seed,
            denominator_tol=args.denominator_tol,
            max_a_size=max_a_size,
            max_l_size=args.max_l_size,
        )
        exact_used = False

    mono = monotonicity_stats(
        f,
        trials=args.trials,
        seed=args.seed,
        tol=args.tol,
        max_set_size=max_monotone_size,
    )
    greedy = greedy_and_exact_optimum(f, args.k, args.max_opt_combinations)

    return {
        "meta": meta,
        "exact_violation_and_ratio": exact_used,
        "violation": asdict(violation),
        "submodularity_ratio": asdict(ratio),
        "monotonicity": asdict(mono),
        "greedy": asdict(greedy),
        "cache_size": len(f._cache),
    }


def print_report(result: Dict[str, Any]) -> None:
    meta = result["meta"]
    violation = result["violation"]
    ratio = result["submodularity_ratio"]
    mono = result["monotonicity"]
    greedy = result["greedy"]

    title = meta["toy"] if meta["toy"] is not None else meta["grad_file"]
    print("=" * 80)
    print(f"Experiment: {title}")
    print(
        f"n={meta['n']} shape={meta['rows']}x{meta['cols']} "
        f"polar={meta['polar']} ns_steps={meta['ns_steps']} "
        f"mu={meta['mu']} target={meta['target_mode']} device={meta['device']}"
    )
    print(f"exact_violation_and_ratio={result['exact_violation_and_ratio']}")
    print("-" * 80)
    print(
        "Diminishing-return violations: "
        f"{violation['violations']}/{violation['trials']} "
        f"rate={violation['violation_rate']:.6f}, "
        f"mean_slack={violation['mean_slack']:.6g}, "
        f"p05_slack={violation['p05_slack']:.6g}, "
        f"min_slack={violation['min_slack']:.6g}"
    )
    if violation["worst_violation"] is not None:
        worst = violation["worst_violation"]
        print(
            "Worst violation: "
            f"A={worst['A']} B={worst['B']} i={worst['i']} "
            f"delta_A={worst['delta_A']:.6g} "
            f"delta_B={worst['delta_B']:.6g} "
            f"severity={worst['severity']:.6g}"
        )

    print(
        "Submodularity ratio samples: "
        f"valid={ratio['valid']}/{ratio['trials']} "
        f"skipped_nonpositive_denominator={ratio['skipped_nonpositive_denominator']}"
    )
    if ratio["gamma_min"] is not None:
        print(
            f"gamma_min={ratio['gamma_min']:.6g}, "
            f"gamma_p01={ratio['gamma_p01']:.6g}, "
            f"gamma_p05={ratio['gamma_p05']:.6g}, "
            f"gamma_median={ratio['gamma_p50']:.6g}, "
            f"gamma_mean={ratio['gamma_mean']:.6g}, "
            f"clipped_min={ratio['gamma_clipped_min']:.6g}"
        )
    else:
        print("gamma unavailable: no positive joint gains were observed.")

    print(
        "Monotonicity: "
        f"negative_marginals={mono['negative_marginals']}/{mono['trials']} "
        f"rate={mono['negative_rate']:.6f}, "
        f"mean={mono['mean_marginal']:.6g}, "
        f"p05={mono['p05_marginal']:.6g}, "
        f"min={mono['min_marginal']:.6g}"
    )
    print(
        "Greedy@k: "
        f"k={greedy['k']} set={greedy['greedy_set']} "
        f"value={greedy['greedy_value']:.6g} gain={greedy['greedy_gain']:.6g}"
    )
    if greedy["optimum_value"] is not None:
        print(
            "Exact optimum@k: "
            f"set={greedy['optimum_set']} value={greedy['optimum_value']:.6g} "
            f"gain={greedy['optimum_gain']:.6g} "
            f"greedy_gain_ratio={greedy['gain_ratio']}"
        )
    else:
        print("Exact optimum@k skipped because n choose k exceeded --max-opt-combinations.")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real-sft", action="store_true", help="Collect real per-sample SFT gradients before analysis.")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="HF model id or local path for --real-sft.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get(
            "DRPT_DATA_DIR", str(Path(__file__).resolve().parent / "SFT" / "data")
        ),
    )
    parser.add_argument("--analysis-dataset", type=str, default="samsum", choices=["samsum", "tydiqa", "nq_open", "squad", "triviaqa"])
    parser.add_argument("--train-dataset-names", type=str, default=None, help="Comma-separated train dataset names, e.g. alpaca or less.")
    parser.add_argument("--eval-split", type=str, default="validation", choices=["validation", "test", "lr"])
    parser.add_argument("--train-n", type=int, default=16, help="Number of real train examples to collect gradients for.")
    parser.add_argument("--val-n", type=int, default=4, help="Number of real validation examples for target gradient.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--real-layer-name", type=str, default=None, help="Parameter/module name for real gradient collection; auto-selects a Muon-style matrix if omitted.")
    parser.add_argument("--real-torch-dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=os.environ.get("HF_HUB_CACHE"),
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow HF downloads; by default only local cache/files are used.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--real-log-every", type=int, default=1)
    parser.add_argument("--save-real-gradients", type=str, default=None, help="Optional .pt path to save collected real gradients.")

    parser.add_argument("--grad-file", type=str, default=None, help="Optional saved gradients file.")
    parser.add_argument("--grad-key", type=str, default=None, help="Key inside .pt/.npz grad file.")
    parser.add_argument("--target-file", type=str, default=None, help="Optional target matrix file.")
    parser.add_argument("--target-key", type=str, default=None, help="Key inside target file.")
    parser.add_argument("--momentum-file", type=str, default=None, help="Optional Muon momentum matrix file.")
    parser.add_argument("--momentum-key", type=str, default=None, help="Key inside momentum file.")
    parser.add_argument("--matrix-shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    parser.add_argument("--matrix-crop", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"), help="Analyze a seeded contiguous submatrix crop to keep polar computations tractable.")
    parser.add_argument("--limit-n", type=int, default=None, help="Use only the first n gradients.")

    parser.add_argument("--toy", type=str, default="low_rank", choices=["gaussian", "low_rank", "clustered", "sign_counterexample"])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--signal", type=float, default=1.0)

    parser.add_argument("--target-mode", type=str, default="polar_full", choices=["polar_full", "mean", "random", "first"])
    parser.add_argument("--no-normalize-target", action="store_true")
    parser.add_argument("--mu", type=float, default=0.95)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--polar", type=str, default="ns", choices=["ns", "svd"])
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-7)

    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--exact-n", type=int, default=10, help="Use exact checks automatically when n <= this.")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--denominator-tol", type=float, default=1e-10)
    parser.add_argument("--max-b-size", type=int, default=None, help="Cap |B| in A subset B violation checks.")
    parser.add_argument("--max-a-size", type=int, default=None, help="Cap |A| in ratio checks.")
    parser.add_argument("--max-l-size", type=int, default=4, help="Cap |L| in ratio checks.")
    parser.add_argument("--max-monotone-size", type=int, default=None, help="Cap |A| in monotonicity checks.")

    parser.add_argument("--k", type=int, default=4, help="Cardinality for greedy-vs-optimum check.")
    parser.add_argument("--max-opt-combinations", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json-out", type=str, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical weak-submodularity diagnostics for Muon aggregate-then-polarize F(S)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--toy-suite",
        action="store_true",
        help="Run gaussian, low_rank, clustered, and sign_counterexample toy experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_sft and args.toy_suite:
        raise ValueError("--real-sft and --toy-suite are mutually exclusive")
    if args.real_sft and args.k > (args.limit_n or args.train_n):
        raise ValueError("--k cannot exceed the number of real train gradients")
    if args.k > (args.limit_n or args.n) and args.grad_file is None and not args.real_sft:
        raise ValueError("--k cannot exceed --n for toy experiments")

    if args.toy_suite and args.grad_file is None:
        results = []
        for toy in ["gaussian", "low_rank", "clustered", "sign_counterexample"]:
            toy_args = argparse.Namespace(**vars(args))
            toy_args.toy = toy
            if toy == "sign_counterexample":
                toy_args.n = max(3, min(args.n, 8))
                toy_args.rows = max(2, args.rows)
                toy_args.cols = max(2, args.cols)
                toy_args.target_mode = "polar_full"
            result = run_one(toy_args)
            results.append(result)
            print_report(result)
        if args.json_out is not None:
            Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        return

    result = run_one(args)
    print_report(result)
    if args.json_out is not None:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
