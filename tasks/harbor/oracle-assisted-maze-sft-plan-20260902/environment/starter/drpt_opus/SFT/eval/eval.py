#!/usr/bin/env python
"""
Unified evaluation script for SFT experiments.

Supports:
- SamSUM: Dialogue summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- TyDiQA: Multilingual QA (F1, EM)
- NQ-open: Closed-book factoid QA (EM, F1)
- SQuAD: Closed-book reading-comprehension QA, no context (EM, F1)
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from SFT.eval.task_registry import TASK_SPECS, default_max_new_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device():
    """Get the appropriate CUDA device (respects CUDA_VISIBLE_DEVICES)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model_and_tokenizer(model_path: str, base_model: Optional[str] = None):
    """Load trained model and tokenizer."""
    device = get_device()
    logger.info(f"Loading model from {model_path} to {device}")

    # Check if tokenizer exists in model_path
    tokenizer_path = model_path
    if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            tokenizer_path = adapter_config.get("base_model_name_or_path", model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    from SFT.data.get_val_dataset import ensure_chat_template
    ensure_chat_template(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Check if this is a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        logger.info("Detected LoRA adapter")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        if base_model is None:
            base_model = adapter_config.get("base_model_name_or_path")
            if base_model is None:
                raise ValueError("Could not determine base model. Please specify --base_model")

        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        ).to(device)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)

    model.eval()

    # Check for NaN/Inf in model weights
    nan_inf_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_inf_params.append(name)
    if nan_inf_params:
        logger.warning(f"Model contains NaN/Inf values in {len(nan_inf_params)} parameters")
        if len(nan_inf_params) > 10:
            # Model is severely corrupted - raise error to skip
            raise ValueError(f"Model is corrupted: NaN/Inf in {len(nan_inf_params)} parameters (training diverged)")

    logger.info("Model loaded successfully")
    return model, tokenizer


# Output dir naming convention:
#   current: {train}_{task}-{method_label}-{optimizer}-p{pct}-lr{lr}-b{bs}-v{nv}-s{seed}-{model}
#   legacy:  {train}_{task}-{model}-{curation}-{finetuning}-{optimizer?}-p{pct}-lr{lr}-b{bs}-v{nv}-s{seed}
# Both {model} (e.g. "Llama-3.2-1B") and method names may contain hyphens, so
# positional split-on-"-" parsing is wrong. Anchor on fixed suffix tokens.
_METHOD_LABELS = {
    "FullTraining": ("FullTraining", "Full", ""),
    "GlobalRaw": ("GlobalSubset", "Full", ""),
    "LayerwiseRaw": ("LayerWiseSubset", "Full", ""),
    "OptGroupRaw": ("OptimizerGroupWise", "Full", ""),
    "GlobalOptA": ("OptimizerAwareGlobalSubset", "Full", "opta"),
    "LayerwiseOptA": ("LayerWiseOptimizerAwareSubset", "Full", "opta"),
    "GlobalOptANorm": ("OptimizerAwareGlobalSubset", "Full", "opta"),
    "LayerwiseOptANorm": ("LayerWiseOptimizerAwareSubset", "Full", "opta"),
    "OptGroupOptA": ("OptimizerAwareGroupWise", "Full", "opta"),
    "GlobalOptB": ("OptimizerAwareGlobalSubset", "Full", "optb"),
    "LayerwiseOptB": ("LayerWiseOptimizerAwareSubset", "Full", "optb"),
    "OptGroupOptB": ("OptimizerAwareGroupWise", "Full", "optb"),
    "GlobalRandom": ("GlobalRandomSubset", "Full", ""),
    "LayerwiseRandom": ("LayerWiseRandomSubset", "Full", ""),
    "GlobalSoft": ("GlobalSoftWeighting", "Full", ""),
    "LayerwiseSoft": ("LayerWiseSoftWeighting", "Full", ""),
    "LayerwiseSoftP": ("LayerWiseSoftProbability", "Full", ""),
    "GlobalHybridMuonSur": ("GlobalMuonSpectral", "Full", ""),
    "LayerwiseHybridMuonSur": ("LayerWiseMuonSpectral", "Full", ""),
    "GlobalHybridMuonMatrixSur": ("GlobalMuonMatrixSpectral", "Full", ""),
    "LayerwiseHybridMuonMatrixSur": ("LayerWiseMuonMatrixSpectral", "Full", ""),
    "GlobalMuonSur": ("GlobalMuonMatrixSpectral", "Full", ""),
    "LayerwiseMuonSur": ("LayerWiseMuonMatrixSpectral", "Full", ""),
    "LayerwiseMuonPSur": ("LayerWiseMuonMatrixSpectralP", "Full", ""),
    "LayerwiseMuonSatSur": ("LayerWiseMuonMatrixSpectralSat", "Full", ""),
    "LayerwiseMuonSatPSur": ("LayerWiseMuonMatrixSpectralSatP", "Full", ""),
}

_METHOD_TO_LABEL = {
    "FullTraining-Full": "FullTraining",
    "FullTraining": "FullTraining",
    "Full-Training": "FullTraining",
    "GlobalSubset-Full": "GlobalRaw",
    "GlobalRaw": "GlobalRaw",
    "Global-Raw": "GlobalRaw",
    "LayerWiseSubset-Full": "LayerwiseRaw",
    "LayerwiseRaw": "LayerwiseRaw",
    "LayerWiseRaw": "LayerwiseRaw",
    "Layerwise-Raw": "LayerwiseRaw",
    "LayerWise-Raw": "LayerwiseRaw",
    "OptimizerGroupWise-Full": "OptGroupRaw",
    "OptGroupRaw": "OptGroupRaw",
    "OptGroup-Raw": "OptGroupRaw",
    "OptimizerAwareGlobalSubset-Full": "GlobalOptA",
    "OptimizerAwareGlobalSubset-OptA-Full": "GlobalOptA",
    "GlobalOptA": "GlobalOptA",
    "Global-OptA": "GlobalOptA",
    "OptimizerAwareGlobalSubset-OptANorm-Full": "GlobalOptANorm",
    "GlobalOptANorm": "GlobalOptANorm",
    "Global-OptANorm": "GlobalOptANorm",
    "LayerWiseOptimizerAwareSubset-Full": "LayerwiseOptA",
    "LayerWiseOptimizerAwareSubset-OptA-Full": "LayerwiseOptA",
    "LayerwiseOptA": "LayerwiseOptA",
    "LayerWiseOptA": "LayerwiseOptA",
    "Layerwise-OptA": "LayerwiseOptA",
    "LayerWise-OptA": "LayerwiseOptA",
    "LayerWiseOptimizerAwareSubset-OptANorm-Full": "LayerwiseOptANorm",
    "LayerwiseOptANorm": "LayerwiseOptANorm",
    "LayerWiseOptANorm": "LayerwiseOptANorm",
    "Layerwise-OptANorm": "LayerwiseOptANorm",
    "LayerWise-OptANorm": "LayerwiseOptANorm",
    "OptimizerAwareGroupWise-Full": "OptGroupOptA",
    "OptimizerAwareGroupWise-OptA-Full": "OptGroupOptA",
    "OptGroupOptA": "OptGroupOptA",
    "OptGroup-OptA": "OptGroupOptA",
    "OptimizerAwareGlobalSubset-OptB-Full": "GlobalOptB",
    "GlobalOptB": "GlobalOptB",
    "Global-OptB": "GlobalOptB",
    "LayerWiseOptimizerAwareSubset-OptB-Full": "LayerwiseOptB",
    "LayerwiseOptB": "LayerwiseOptB",
    "LayerWiseOptB": "LayerwiseOptB",
    "Layerwise-OptB": "LayerwiseOptB",
    "LayerWise-OptB": "LayerwiseOptB",
    "OptimizerAwareGroupWise-OptB-Full": "OptGroupOptB",
    "OptGroupOptB": "OptGroupOptB",
    "OptGroup-OptB": "OptGroupOptB",
    "GlobalRandomSubset-Full": "GlobalRandom",
    "GlobalRandomSubset": "GlobalRandom",
    "GlobalRandom": "GlobalRandom",
    "LayerWiseRandomSubset-Full": "LayerwiseRandom",
    "LayerWiseRandomSubset": "LayerwiseRandom",
    "LayerwiseRandom": "LayerwiseRandom",
    "LayerWiseRandom": "LayerwiseRandom",
    "GlobalSoftWeighting-Full": "GlobalSoft",
    "GlobalSoftWeighting": "GlobalSoft",
    "GlobalSoft": "GlobalSoft",
    "LayerWiseSoftWeighting-Full": "LayerwiseSoft",
    "LayerWiseSoftWeighting": "LayerwiseSoft",
    "LayerwiseSoft": "LayerwiseSoft",
    "LayerWiseSoft": "LayerwiseSoft",
    "LayerWiseSoftProbability-Full": "LayerwiseSoftP",
    "LayerWiseSoftProbability": "LayerwiseSoftP",
    "LayerwiseSoftP": "LayerwiseSoftP",
    "LayerWiseSoftP": "LayerwiseSoftP",
    "GlobalMuonSpectral-Full": "GlobalHybridMuonSur",
    "GlobalMuonSpectral": "GlobalHybridMuonSur",
    "GlobalHybridMuonSur": "GlobalHybridMuonSur",
    "GlobalHybridMuonMatrixSur": "GlobalHybridMuonMatrixSur",
    "GlobalMuonSur": "GlobalMuonSur",
    "LayerWiseMuonSpectral-Full": "LayerwiseHybridMuonSur",
    "LayerWiseMuonSpectral": "LayerwiseHybridMuonSur",
    "LayerwiseHybridMuonSur": "LayerwiseHybridMuonSur",
    "LayerWiseHybridMuonSur": "LayerwiseHybridMuonSur",
    "LayerwiseHybridMuonMatrixSur": "LayerwiseHybridMuonMatrixSur",
    "LayerWiseHybridMuonMatrixSur": "LayerwiseHybridMuonMatrixSur",
    "LayerwiseMuonSur": "LayerwiseMuonSur",
    "LayerWiseMuonSur": "LayerwiseMuonSur",
    "GlobalMuonMatrixSpectral-Full": "GlobalMuonSur",
    "GlobalMuonMatrixSpectral": "GlobalMuonSur",
    "GlobalMuonMatrixSur": "GlobalMuonSur",
    "GlobalMuonOnlySur": "GlobalMuonSur",
    "LayerWiseMuonMatrixSpectral-Full": "LayerwiseMuonSur",
    "LayerWiseMuonMatrixSpectral": "LayerwiseMuonSur",
    "LayerwiseMuonMatrixSur": "LayerwiseMuonSur",
    "LayerWiseMuonMatrixSur": "LayerwiseMuonSur",
    "LayerwiseMuonOnlySur": "LayerwiseMuonSur",
    "LayerWiseMuonOnlySur": "LayerwiseMuonSur",
    "LayerWiseMuonMatrixSpectralP-Full": "LayerwiseMuonPSur",
    "LayerWiseMuonMatrixSpectralP": "LayerwiseMuonPSur",
    "LayerwiseMuonMatrixPSur": "LayerwiseMuonPSur",
    "LayerWiseMuonMatrixPSur": "LayerwiseMuonPSur",
    "LayerwiseMuonOnlyPSur": "LayerwiseMuonPSur",
    "LayerWiseMuonOnlyPSur": "LayerwiseMuonPSur",
    "LayerwiseMuonPSur": "LayerwiseMuonPSur",
    "LayerWiseMuonPSur": "LayerwiseMuonPSur",
    "LayerWiseMuonMatrixSpectralSat-Full": "LayerwiseMuonSatSur",
    "LayerWiseMuonMatrixSpectralSat": "LayerwiseMuonSatSur",
    "LayerwiseMuonMatrixSatSur": "LayerwiseMuonSatSur",
    "LayerWiseMuonMatrixSatSur": "LayerwiseMuonSatSur",
    "LayerwiseMuonOnlySatSur": "LayerwiseMuonSatSur",
    "LayerWiseMuonOnlySatSur": "LayerwiseMuonSatSur",
    "LayerwiseMuonSatSur": "LayerwiseMuonSatSur",
    "LayerWiseMuonSatSur": "LayerwiseMuonSatSur",
    "LayerWiseMuonMatrixSpectralSatP-Full": "LayerwiseMuonSatPSur",
    "LayerWiseMuonMatrixSpectralSatP": "LayerwiseMuonSatPSur",
    "LayerwiseMuonMatrixSatPSur": "LayerwiseMuonSatPSur",
    "LayerWiseMuonMatrixSatPSur": "LayerwiseMuonSatPSur",
    "LayerwiseMuonOnlySatPSur": "LayerwiseMuonSatPSur",
    "LayerWiseMuonOnlySatPSur": "LayerwiseMuonSatPSur",
    "LayerwiseMuonSatPSur": "LayerwiseMuonSatPSur",
    "LayerWiseMuonSatPSur": "LayerwiseMuonSatPSur",
}

_NEW_NAME_RE = re.compile(
    r"^"
    r"(?P<prefix>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    r"-(?P<method_label>FullTraining|GlobalRaw|LayerwiseRaw|OptGroupRaw|"
    r"GlobalOptA|LayerwiseOptA|GlobalOptANorm|LayerwiseOptANorm|OptGroupOptA|"
    r"GlobalOptB|LayerwiseOptB|OptGroupOptB|"
    r"GlobalRandom|LayerwiseRandom|GlobalSoft|LayerwiseSoft|LayerwiseSoftP|"
    r"GlobalHybridMuonSur|LayerwiseHybridMuonSur|"
    r"GlobalHybridMuonMatrixSur|LayerwiseHybridMuonMatrixSur|"
    r"GlobalMuonSur|LayerwiseMuonSur|"
    r"LayerwiseMuonPSur|LayerwiseMuonSatSur|LayerwiseMuonSatPSur|"
    r"GlobalMuonMatrixSur|LayerwiseMuonMatrixSur|LayerwiseMuonOnlyPSur|"
    r"LayerwiseMuonOnlySatSur|LayerwiseMuonOnlySatPSur)"
    r"-(?P<optimizer_type>adamw|muon|hybrid)"
    r"-(?:p(?P<percentage>[\d.]+)|ms(?P<max_steps>\d+))"
    r"-lr(?P<learning_rate>[\d.]+e-?\d+)"
    r"-b(?P<batch_size>\d+)"
    r"-v(?P<n_val>\d+)"
    r"-s(?P<seed>\d+)"
    r"-(?P<model>.+)"
    r"$"
)

_NAME_RE = re.compile(
    r"^"
    r"(?P<prefix>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    r"-(?P<model>.+?)"
    r"-(?P<curation>FullTraining|GlobalSubset|LayerWiseSubset|LayerWiseOptimizerAwareSubset|"
    r"OptimizerGroupWise|OptimizerAwareGlobalSubset|OptimizerAwareGroupWise|"
    r"GlobalRandomSubset|LayerWiseRandomSubset|GlobalSoftWeighting|LayerWiseSoftWeighting|"
    r"LayerWiseSoftProbability|"
    r"GlobalMuonSpectral|LayerWiseMuonSpectral|GlobalMuonMatrixSpectral|"
    r"LayerWiseMuonMatrixSpectral|LayerWiseMuonMatrixSpectralP|"
    r"LayerWiseMuonMatrixSpectralSat|LayerWiseMuonMatrixSpectralSatP|Standard)"
    r"(?:-(?P<target_mode>OptA|OptANorm|OptB))?"
    r"-(?P<finetuning>MeSO-LoRA|Full|LoRA|MeSO)"
    r"(?:-(?P<optimizer_type>optadamw|opthybrid|adamw-only|adamw|muon|hybrid))?"
    r"-(?:p(?P<percentage>[\d.]+)|ms(?P<max_steps>\d+))"
    r"-lr(?P<learning_rate>[\d.]+e-?\d+)"
    r"-b(?P<batch_size>\d+)"
    r"-v(?P<n_val>\d+)"
    r"-s(?P<seed>\d+)"
    r"$"
)


# Ordered longest-name-first so `_parse_prefix` and `get_task_from_model_name`
# match `mbpp_plus` / `nq_open` before any shorter name that is a suffix of them.
VALID_TASKS = sorted(TASK_SPECS, key=len, reverse=True)


def _normalize_optimizer_label(optimizer_type: str) -> str:
    if optimizer_type in ("optadamw", "adamw-only", "adamw_only"):
        return "adamw"
    if optimizer_type == "opthybrid":
        return "hybrid"
    return optimizer_type or ""


def _canonical_method_label(method_label: str, optimizer_type: str) -> str:
    """Normalize historical run labels without losing their scorer semantics."""
    legacy_aliases = {
        "GlobalMuonMatrixSur": "GlobalMuonSur",
        "LayerwiseMuonMatrixSur": "LayerwiseMuonSur",
        "LayerwiseMuonOnlyPSur": "LayerwiseMuonPSur",
        "LayerwiseMuonOnlySatSur": "LayerwiseMuonSatSur",
        "LayerwiseMuonOnlySatPSur": "LayerwiseMuonSatPSur",
    }
    if optimizer_type == "hybrid":
        if method_label == "GlobalMuonSur":
            return "GlobalHybridMuonSur"
        if method_label == "LayerwiseMuonSur":
            return "LayerwiseHybridMuonSur"
        if method_label == "GlobalMuonMatrixSur":
            return "GlobalHybridMuonMatrixSur"
        if method_label == "LayerwiseMuonMatrixSur":
            return "LayerwiseHybridMuonMatrixSur"
    return legacy_aliases.get(method_label, method_label)


def _parse_prefix(prefix: str) -> Dict[str, str]:
    """Split a ``<train>_<task>`` run-dir prefix into its two parts.

    Both halves can contain underscores (``nq_open``, ``reasoning_32k``),
    so this matches the longest known task suffix rather than splitting on ``_``.
    """
    for task in VALID_TASKS:
        if prefix == f"{task}_val_{task}":
            return {"train_dataset": f"{task}_val", "eval_task": task}
    for task in VALID_TASKS:
        suffix = "_" + task
        if prefix.endswith(suffix) and len(prefix) > len(suffix):
            return {"train_dataset": prefix[: -len(suffix)], "eval_task": task}
    prefix_parts = prefix.split("_")
    if len(prefix_parts) == 3 and prefix_parts[1] == "val":
        return {"train_dataset": f"{prefix_parts[0]}_val", "eval_task": prefix_parts[2]}
    if len(prefix_parts) == 2:
        return {"train_dataset": prefix_parts[0], "eval_task": prefix_parts[1]}
    return {"train_dataset": prefix, "eval_task": ""}


def _method_filter_patterns(method: str) -> List[str]:
    label = _METHOD_TO_LABEL.get(method, method)
    patterns = [f"-{method}-", f"-{label}-"]
    historical_run_labels = {
        "GlobalHybridMuonSur": ("GlobalMuonSur",),
        "LayerwiseHybridMuonSur": ("LayerwiseMuonSur",),
        "GlobalHybridMuonMatrixSur": ("GlobalMuonMatrixSur",),
        "LayerwiseHybridMuonMatrixSur": ("LayerwiseMuonMatrixSur",),
        "GlobalMuonSur": ("GlobalMuonMatrixSur",),
        "LayerwiseMuonSur": ("LayerwiseMuonMatrixSur", "LayerwiseMuonOnlySur"),
        "LayerwiseMuonPSur": ("LayerwiseMuonOnlyPSur", "LayerwiseMuonMatrixPSur"),
        "LayerwiseMuonSatSur": (
            "LayerwiseMuonOnlySatSur",
            "LayerwiseMuonMatrixSatSur",
        ),
        "LayerwiseMuonSatPSur": (
            "LayerwiseMuonOnlySatPSur",
            "LayerwiseMuonMatrixSatPSur",
        ),
    }
    patterns.extend(
        f"-{historical_label}-"
        for historical_label in historical_run_labels.get(label, ())
    )
    normalized_legacy = {
        "GlobalOptANorm": "OptimizerAwareGlobalSubset-OptANorm-Full",
        "LayerwiseOptANorm": "LayerWiseOptimizerAwareSubset-OptANorm-Full",
    }
    if label in normalized_legacy:
        patterns.append(f"-{normalized_legacy[label]}-")
        return sorted(set(patterns))
    if label in _METHOD_LABELS:
        selection, finetuning, target_mode = _METHOD_LABELS[label]
        if target_mode:
            patterns.append(
                f"-{selection}-{'OptA' if target_mode == 'opta' else 'OptB'}-{finetuning}-"
            )
        patterns.append(f"-{selection}-{finetuning}-")
    return sorted(set(patterns))


def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse a model output-dir name into experiment fields. Cosmetic — used
    only for the printed eval summary and the master log row.
    """
    config = {
        "model_name": model_name,
        "train_dataset": "", "eval_task": "",
        "model": "",
        "selection": "", "training_type": "",
        "optimizer_type": "", "optimizer_aware_target_mode": "",
        "percentage": "", "max_steps": "",
        "learning_rate": "", "batch_size": "", "n_val": "", "seed": "",
    }
    new_m = _NEW_NAME_RE.match(model_name)
    if new_m:
        g = new_m.groupdict()
        config.update(_parse_prefix(g["prefix"]))
        optimizer_type = _normalize_optimizer_label(g["optimizer_type"])
        method_label = _canonical_method_label(g["method_label"], optimizer_type)
        selection, finetuning, target_mode = _METHOD_LABELS[method_label]
        config["model"] = g["model"]
        config["selection"] = selection
        config["training_type"] = finetuning
        config["optimizer_type"] = optimizer_type
        config["optimizer_aware_target_mode"] = target_mode
        config["percentage"] = g["percentage"] or ""
        config["max_steps"] = g["max_steps"] or ""
        config["learning_rate"] = g["learning_rate"]
        config["batch_size"] = g["batch_size"]
        config["n_val"] = g["n_val"]
        config["seed"] = g["seed"]
        return config

    m = _NAME_RE.match(model_name)
    if not m:
        return config
    g = m.groupdict()

    config.update(_parse_prefix(g["prefix"]))

    config["model"] = g["model"]
    config["selection"] = g["curation"]
    config["training_type"] = g["finetuning"]
    config["optimizer_type"] = _normalize_optimizer_label(g["optimizer_type"] or "")
    legacy_target_mode = (g["target_mode"] or "").lower()
    config["optimizer_aware_target_mode"] = (
        "opta" if legacy_target_mode == "optanorm" else legacy_target_mode
    )
    config["percentage"] = g["percentage"] or ""
    config["max_steps"] = g["max_steps"] or ""
    config["learning_rate"] = g["learning_rate"]
    config["batch_size"] = g["batch_size"]
    config["n_val"] = g["n_val"]
    config["seed"] = g["seed"]
    return config


def find_models(
    models_dir: str,
    train_dataset: Optional[str] = None,
    method: Optional[str] = None,
    optimizer_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Find all model directories, optionally filtering by prefix pattern and method.

    Args:
        models_dir: Directory containing model directories
        train_dataset: Filter prefix (e.g., "alpaca_samsum")
        method: Method filter (e.g., "FullTraining-MeSO", "LayerWiseSubset-Full")
        optimizer_type: Optimizer filter ("adamw", "muon", or "hybrid"), matched against
            the training run-name suffix "-opt{optimizer_type}-".
        seed: Exact training seed parsed from the run name.
    """
    model_paths = []
    for entry in os.listdir(models_dir):
        entry_path = os.path.join(models_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        has_model = (
            os.path.exists(os.path.join(entry_path, "config.json")) or
            os.path.exists(os.path.join(entry_path, "adapter_config.json"))
        )
        if has_model:
            # Check train_dataset filter
            if train_dataset is not None:
                if not (entry.startswith(train_dataset + "-") or entry.startswith(train_dataset + "_")):
                    continue

            # Check method filter (e.g., "FullTraining-MeSO" matches "-FullTraining-MeSO-")
            if method is not None:
                # Current names use short labels (e.g. "-GlobalOptB-").
                # Legacy names use internal config names (e.g.
                # "-OptimizerAwareGlobalSubset-OptB-Full-").
                if not any(pattern in entry for pattern in _method_filter_patterns(method)):
                    continue

            if optimizer_type is not None:
                opt_patterns = [f"-opt{optimizer_type}-", f"-{optimizer_type}-"]
                if optimizer_type == "adamw":
                    opt_patterns.append("-adamw-only-")
                elif optimizer_type == "hybrid":
                    opt_patterns.append("-hybrid-")
                if not any(pattern in entry for pattern in opt_patterns):
                    continue

            # Do not use a substring such as ``-s42`` here: that would also
            # match seeds like 420. Both supported naming schemes expose the
            # seed through parse_model_name, so compare the parsed field.
            if seed is not None:
                parsed_seed = parse_model_name(entry).get("seed", "")
                if parsed_seed != str(seed):
                    continue

            model_paths.append(entry_path)
    return sorted(model_paths)


_RESULT_FILENAMES = {
    "samsum": "samsum_results.json",
    "tydiqa": "tydiqa_results.json",
    "nq_open": "nq_open_results.json",
    "squad": "squad_results.json",
    "triviaqa": "triviaqa_results.json",
    "ifeval": "ifeval_results.json",
    "ifbench": "ifbench_results.json",
    "math500": "math500_results.json",
    "mbpp_plus": "mbpp_plus_results.json",
}

_PRIMARY_RESULT_METRICS = {
    "samsum": "rougeL",
    "tydiqa": "f1_score",
    "nq_open": "f1",
    "squad": "f1",
    "triviaqa": "f1",
    "ifeval": "prompt_level_strict_acc",
    "ifbench": "prompt_level_loose_acc",
    "math500": "accuracy",
    "mbpp_plus": "base_plus_extra_pass_at_1",
}

# Tasks whose primary metric may legitimately be absent. MT-Bench generation
# always succeeds, but judge scoring needs OPENAI_API_KEY; without it the run is
# complete and its answers are on disk, so a null score is a state, not a failure.
_OPTIONAL_METRIC_TASKS: set[str] = set()


def _validate_required_result(model_path: str, task: str) -> Optional[str]:
    """Return an error message unless the task result JSON is usable."""
    filename = _RESULT_FILENAMES.get(task)
    if filename is None:
        return f"No required result JSON is defined for task {task!r}"

    result_path = os.path.join(model_path, filename)
    if not os.path.isfile(result_path):
        return f"Required result JSON was not written: {result_path}"

    try:
        with open(result_path, "r") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        return f"Required result JSON is unreadable: {result_path}: {exc}"

    if not isinstance(payload, dict):
        return f"Required result JSON is not an object: {result_path}"
    if payload.get("task") != task:
        return (
            f"Required result JSON has task {payload.get('task')!r}, "
            f"expected {task!r}: {result_path}"
        )
    primary_metric = _PRIMARY_RESULT_METRICS[task]
    metric_value = payload.get(primary_metric)
    if metric_value is None and task in _OPTIONAL_METRIC_TASKS:
        status = payload.get("status")
        if status == "generated_unscored":
            return None
        return (
            f"Required result JSON has a null primary metric {primary_metric!r} "
            f"without an explanatory status (got {status!r}): {result_path}"
        )
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (int, float))
        or not math.isfinite(float(metric_value))
    ):
        return (
            f"Required result JSON has non-finite or non-numeric primary metric "
            f"{primary_metric!r}: {result_path}"
        )
    return None


def evaluate_samsum(args, model, tokenizer) -> dict:
    """Run SamSUM evaluation."""
    from .tasks.samsum import compute_accuracy

    logger.info("Evaluating on SamSUM")
    scores = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    out = {"task": "samsum"}
    out.update(scores)
    return out


def evaluate_tydiqa(args, model, tokenizer) -> dict:
    """Run TyDiQA evaluation."""
    from .tasks.tydiqa import compute_accuracy

    logger.info("Evaluating on TyDiQA")
    results = compute_accuracy(args=args, model=model, tokenizer=tokenizer)
    out = {"task": "tydiqa"}
    out.update(results)
    return out


def evaluate_nq_open(args, model, tokenizer) -> dict:
    """Run NQ-open closed-book QA evaluation (EM/F1)."""
    from .tasks.nq_open import compute_accuracy
    logger.info("Evaluating on NQ-open")
    out = {"task": "nq_open"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def evaluate_squad(args, model, tokenizer) -> dict:
    """Run SQuAD closed-book (no context) evaluation (EM/F1)."""
    from .tasks.squad import compute_accuracy
    logger.info("Evaluating on SQuAD (closed-book)")
    out = {"task": "squad"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def evaluate_triviaqa(args, model, tokenizer) -> dict:
    """Run TriviaQA closed-book evaluation (EM/F1)."""
    from .tasks.triviaqa import compute_accuracy
    logger.info("Evaluating on TriviaQA (closed-book)")
    out = {"task": "triviaqa"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def evaluate_ifeval(args, model, tokenizer) -> dict:
    """Run IFEval (official strict/loose, prompt- and instruction-level)."""
    from .tasks.ifeval import compute_accuracy
    logger.info("Evaluating on IFEval")
    out = {"task": "ifeval"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    ))
    return out


def evaluate_ifbench(args, model, tokenizer) -> dict:
    """Run the pinned official IFBench strict/loose verifier."""
    from .tasks.ifbench import compute_accuracy

    logger.info("Evaluating on IFBench")
    out = {"task": "ifbench"}
    out.update(compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    ))
    return out


def evaluate_math500(args, model, tokenizer) -> dict:
    """Run MATH-500 with pinned math-verify equivalence scoring."""
    from .tasks.math500 import compute_accuracy

    logger.info("Evaluating on MATH-500")
    out = {"task": "math500"}
    out.update(compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    ))
    return out


def evaluate_mbpp_plus(args, model, tokenizer) -> dict:
    """Run full MBPP+ through the pinned network-isolated EvalPlus image."""
    from .tasks.mbpp_plus import compute_accuracy

    logger.info("Evaluating on MBPP+")
    out = {"task": "mbpp_plus"}
    out.update(compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    ))
    return out


def get_task_from_model_name(model_name: str) -> Optional[str]:
    """Extract the evaluation task from a run-dir name. Two layouts:

    Main runs:        ``<train>_<task>-<model>-...``      (e.g. ``alpaca_samsum-...``)
    Target-only runs: ``<task>_val_<task>-<model>-...``    (e.g. ``samsum_val_samsum-...``)
    """
    head = model_name.split("-", 1)[0]
    # Target-only: <task>_val_<task>
    for t in VALID_TASKS:
        if head == f"{t}_val_{t}":
            return t
    # Main: <train>_<task> — task is the suffix
    for t in VALID_TASKS:
        if head.endswith("_" + t):
            return t
    return None


def evaluate_model(
    model_path: str,
    data_dir: str,
    n_test: int = -1,
    batch_size: int = 1,
    max_new_tokens: Optional[int] = None,
    base_model: Optional[str] = None,
    task_override: Optional[str] = None,
    subject: Optional[str] = None,
) -> Dict:
    """Evaluate a single model. Task is auto-detected from model name."""
    model_name = os.path.basename(model_path)
    results = parse_model_name(model_name)
    results["model_path"] = model_path

    # Auto-detect task from model name, or use override
    task = task_override or get_task_from_model_name(model_name)
    if task is None:
        logger.error(f"Could not detect task from model name: {model_name}")
        results["error"] = "Could not detect task from model name"
        return results

    logger.info(f"Auto-detected task: {task}")
    resolved_max_new_tokens = (
        default_max_new_tokens(task)
        if max_new_tokens is None or int(max_new_tokens) <= 0
        else int(max_new_tokens)
    )

    try:
        model, tokenizer = load_model_and_tokenizer(model_path, base_model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        results["error"] = str(e)
        return results

    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.n_test = n_test
    args.batch_size = batch_size
    args.max_new_tokens = resolved_max_new_tokens
    args.subject = subject  # legacy arg; unused in current scope
    # Extended-benchmark evaluators write per-example generations next to the
    # run so failures can be inspected without re-running the model.
    args.output_dir = model_path

    try:
        if task == "samsum":
            logger.info(f"Evaluating {model_name} on SamSUM...")
            samsum_results = evaluate_samsum(args, model, tokenizer)
            results["samsum_rouge1"] = samsum_results["rouge1"]
            results["samsum_rouge2"] = samsum_results["rouge2"]
            results["samsum_rougeL"] = samsum_results["rougeL"]

            with open(os.path.join(model_path, "samsum_results.json"), "w") as f:
                json.dump(samsum_results, f, indent=2)

        elif task == "tydiqa":
            logger.info(f"Evaluating {model_name} on TyDiQA...")
            tydiqa_results = evaluate_tydiqa(args, model, tokenizer)
            results["tydiqa_f1"] = tydiqa_results["f1_score"]
            results["tydiqa_em"] = tydiqa_results["exact_match"]

            with open(os.path.join(model_path, "tydiqa_results.json"), "w") as f:
                json.dump(tydiqa_results, f, indent=2)

        elif task == "nq_open":
            logger.info(f"Evaluating {model_name} on NQ-open...")
            nq_results = evaluate_nq_open(args, model, tokenizer)
            results["nq_open_em"] = nq_results["em"]
            results["nq_open_f1"] = nq_results["f1"]
            with open(os.path.join(model_path, "nq_open_results.json"), "w") as f:
                json.dump(nq_results, f, indent=2)

        elif task == "squad":
            logger.info(f"Evaluating {model_name} on SQuAD (closed-book)...")
            sq_results = evaluate_squad(args, model, tokenizer)
            results["squad_em"] = sq_results["em"]
            results["squad_f1"] = sq_results["f1"]
            with open(os.path.join(model_path, "squad_results.json"), "w") as f:
                json.dump(sq_results, f, indent=2)

        elif task == "triviaqa":
            logger.info(f"Evaluating {model_name} on TriviaQA (closed-book)...")
            tq_results = evaluate_triviaqa(args, model, tokenizer)
            results["triviaqa_em"] = tq_results["em"]
            results["triviaqa_f1"] = tq_results["f1"]
            with open(os.path.join(model_path, "triviaqa_results.json"), "w") as f:
                json.dump(tq_results, f, indent=2)

        elif task == "ifeval":
            logger.info(f"Evaluating {model_name} on IFEval...")
            if_results = evaluate_ifeval(args, model, tokenizer)
            results["ifeval_prompt_strict"] = if_results["prompt_level_strict_acc"]
            results["ifeval_inst_strict"] = if_results["inst_level_strict_acc"]
            results["ifeval_prompt_loose"] = if_results["prompt_level_loose_acc"]
            results["ifeval_inst_loose"] = if_results["inst_level_loose_acc"]
            with open(os.path.join(model_path, "ifeval_results.json"), "w") as f:
                json.dump(if_results, f, indent=2)

        elif task == "ifbench":
            logger.info(f"Evaluating {model_name} on IFBench...")
            ifbench_results = evaluate_ifbench(args, model, tokenizer)
            results["ifbench_prompt_strict"] = ifbench_results["prompt_level_strict_acc"]
            results["ifbench_inst_strict"] = ifbench_results["inst_level_strict_acc"]
            results["ifbench_prompt_loose"] = ifbench_results["prompt_level_loose_acc"]
            results["ifbench_inst_loose"] = ifbench_results["inst_level_loose_acc"]
            with open(os.path.join(model_path, "ifbench_results.json"), "w") as f:
                json.dump(ifbench_results, f, indent=2)

        elif task == "math500":
            logger.info(f"Evaluating {model_name} on MATH-500...")
            math_results = evaluate_math500(args, model, tokenizer)
            results["math500_accuracy"] = math_results["accuracy"]
            with open(os.path.join(model_path, "math500_results.json"), "w") as f:
                json.dump(math_results, f, indent=2)

        elif task == "mbpp_plus":
            logger.info(f"Evaluating {model_name} on MBPP+...")
            mbpp_plus_results = evaluate_mbpp_plus(args, model, tokenizer)
            results["mbpp_plus_base_plus_extra_pass_at_1"] = mbpp_plus_results[
                "base_plus_extra_pass_at_1"
            ]
            results["mbpp_plus_pass_at_1"] = mbpp_plus_results["plus_pass_at_1"]
            results["mbpp_plus_base_pass_at_1"] = mbpp_plus_results["base_pass_at_1"]
            with open(os.path.join(model_path, "mbpp_plus_results.json"), "w") as f:
                json.dump(mbpp_plus_results, f, indent=2)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results["error"] = str(e)

    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
            # Try to reset CUDA state
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    if not results.get("error"):
        result_error = _validate_required_result(model_path, task)
        if result_error is not None:
            logger.error(result_error)
            results["error"] = result_error

    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    repo_root = os.environ.get(
        "DRPT_REPO_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    default_runs_dir = os.environ.get(
        "DRPT_RUNS_DIR", os.path.join(repo_root, "SFT", "runs")
    )
    parser = argparse.ArgumentParser(description="SFT Evaluation Script")

    # Model selection (batch is default)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models_dir", type=str,
        default=default_runs_dir,
        help="Directory containing trained models (default)")
    model_group.add_argument("--model_path", type=str,
        help="Path to single model to evaluate")

    parser.add_argument("--train", type=str, default=None,
        help="Filter by training dataset (e.g., alpaca, less, tulu3, wizardlm)")
    parser.add_argument("--task", type=str, default=None,
        choices=tuple(TASK_SPECS),
        help="Override auto-detected task (optional)")
    parser.add_argument("--subject", type=str, default=None,
        help="(legacy; unused in current scope)")
    parser.add_argument("--method", type=str, default=None,
        help="Filter by method (e.g., FullTraining-MeSO, LayerWiseSubset-Full)")
    parser.add_argument("--optimizer_type", type=str, default=None,
        choices=["adamw", "muon", "hybrid"],
        help="Filter by optimizer type saved in run name (adamw, muon, or hybrid)")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DRPT_DATA_DIR"),
        help="Data directory (default: auto-detect)")
    parser.add_argument("--n_test", type=int, default=-1,
        help="Number of test examples (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=None,
        help="Maximum tokens to generate (default: task registry policy)")
    parser.add_argument("--base_model", type=str, default=None,
        help="Base model for LoRA adapters")
    parser.add_argument("--seed", type=int, default=42,
        help="Training-run filter and random seed for reproducibility (default: 42)")
    parser.add_argument("--require_single_match", action="store_true",
        help="Fail unless batch run discovery resolves to exactly one model")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Auto-detect data directory
    if args.data_dir is None:
        sft_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        args.data_dir = sft_data_dir if os.path.exists(sft_data_dir) else "./data"

    logger.info(f"Data directory: {args.data_dir}")

    # Single model evaluation
    if args.model_path:
        results = evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            n_test=args.n_test,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            base_model=args.base_model,
            task_override=args.task,
            subject=args.subject,
        )
        print("\n" + "=" * 60)
        print("Results:")
        for k, v in results.items():
            if k not in ["model_path", "timestamp", "model_name"]:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if results.get("error"):
            logger.error("Evaluation failed for %s", args.model_path)
            sys.exit(1)
        return

    # Batch evaluation
    # Construct filter prefix from train, task, and subject
    filter_prefix = args.train
    if filter_prefix and args.task:
        filter_prefix = f"{filter_prefix}_{args.task}"
        if args.subject:
            filter_prefix = f"{filter_prefix}_{args.subject}"
    model_paths = find_models(
        args.models_dir,
        filter_prefix,
        args.method,
        args.optimizer_type,
        args.seed,
    )
    logger.info(f"Found {len(model_paths)} models to evaluate")
    if args.method:
        logger.info(f"Filtering by method: {args.method}")
    if args.optimizer_type:
        logger.info(f"Filtering by optimizer_type: {args.optimizer_type}")

    if args.require_single_match and len(model_paths) != 1:
        logger.error(
            "Expected exactly one model for train=%r task=%r method=%r "
            "optimizer=%r seed=%d, found %d",
            args.train,
            args.task,
            args.method,
            args.optimizer_type,
            args.seed,
            len(model_paths),
        )
        for model_path in model_paths:
            logger.error("Ambiguous match: %s", model_path)
        sys.exit(1)

    if not model_paths:
        logger.error(f"No seed-{args.seed} models found in {args.models_dir}")
        sys.exit(1)

    all_results = []
    for i, model_path in enumerate(model_paths):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(model_paths)}] {os.path.basename(model_path)}")
        print(f"{'=' * 70}")

        results = evaluate_model(
            model_path=model_path,
            data_dir=args.data_dir,
            n_test=args.n_test,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            base_model=args.base_model,
            task_override=args.task,
            subject=args.subject,
        )
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    # SamSUM results
    samsum_results = [r for r in all_results if "samsum_rougeL" in r]
    if samsum_results:
        print(f"\nSamSUM Results:")
        print(f"{'Model':<70} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
        print("-" * 100)
        for r in sorted(samsum_results, key=lambda x: x.get("samsum_rougeL", 0), reverse=True):
            print(f"{r['model_name'][:70]:<70} "
                  f"{r['samsum_rouge1']:>7.4f} {r['samsum_rouge2']:>7.4f} {r['samsum_rougeL']:>7.4f}")

    # TyDiQA results
    tydiqa_results = [r for r in all_results if "tydiqa_f1" in r]
    if tydiqa_results:
        print(f"\nTyDiQA Results:")
        print(f"{'Model':<80} {'F1':>8} {'EM':>8}")
        print("-" * 100)
        for r in sorted(tydiqa_results, key=lambda x: x.get("tydiqa_f1", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['tydiqa_f1']:>8.4f} {r.get('tydiqa_em', 0):>8.4f}")

    # NQ-open results
    nq_results = [r for r in all_results if "nq_open_em" in r]
    if nq_results:
        print(f"\nNQ-open Results:")
        print(f"{'Model':<80} {'EM':>8} {'F1':>8}")
        print("-" * 100)
        for r in sorted(nq_results, key=lambda x: x.get("nq_open_em", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['nq_open_em']:>8.4f} {r.get('nq_open_f1', 0):>8.4f}")

    # SQuAD (closed-book) results
    squad_results = [r for r in all_results if "squad_em" in r]
    if squad_results:
        print(f"\nSQuAD (closed-book) Results:")
        print(f"{'Model':<80} {'EM':>8} {'F1':>8}")
        print("-" * 100)
        for r in sorted(squad_results, key=lambda x: x.get("squad_em", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['squad_em']:>8.4f} {r.get('squad_f1', 0):>8.4f}")

    # Extended benchmarks
    ifeval_results = [r for r in all_results if "ifeval_prompt_strict" in r]
    if ifeval_results:
        print(f"\nIFEval Results (prompt/instruction level, strict and loose):")
        print(f"{'Model':<64} {'P-str':>7} {'I-str':>7} {'P-loo':>7} {'I-loo':>7}")
        print("-" * 100)
        for r in sorted(ifeval_results, key=lambda x: x.get("ifeval_prompt_strict", 0), reverse=True):
            print(f"{r['model_name'][:64]:<64} "
                  f"{r['ifeval_prompt_strict']:>7.2f} {r['ifeval_inst_strict']:>7.2f} "
                  f"{r['ifeval_prompt_loose']:>7.2f} {r['ifeval_inst_loose']:>7.2f}")

    errors = [r for r in all_results if r.get("error")]
    if errors:
        print(f"\nErrors: {len(errors)} models failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
