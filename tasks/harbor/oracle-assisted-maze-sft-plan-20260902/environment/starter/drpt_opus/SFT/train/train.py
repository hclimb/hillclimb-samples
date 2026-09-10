#!/usr/bin/env python
# coding=utf-8
"""
Training script for SFT with layer_wise_subset descent.
"""

import json
import hashlib
import logging
import os
import platform
import resource
import sys
import time
import warnings
from collections import Counter

import datasets
import torch
import torch.distributed as dist
import transformers

try:
    import wandb
except ImportError:
    wandb = None

# Suppress torch.compile warnings about custom CUDA kernels (SJLT)
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, set_seed)

from SFT.data.get_train_dataset import get_training_dataset
from SFT.data.get_val_dataset import (
    DEFAULT_SEQ_LENGTH_MULTIPLIER,
    ensure_chat_template,
    get_dataset,
    get_messages_file_dataset,
)
from SFT.data.dolci32k.artifacts import (
    resolve_current_build as resolve_dolci32k_build,
    validate_build as validate_dolci32k_build,
)
from SFT.data.dolci32k.common import file_sha256 as dolci32k_file_sha256
from SFT.data.dolci32k.profile import (
    DEFAULT_SIZES as DOLCI32K_SIZES,
    MAX_SEQ_LEN as DOLCI32K_MAX_SEQ_LEN,
    MODEL_PROFILES as DOLCI32K_MODEL_PROFILES,
    SETTINGS as DOLCI32K_SETTINGS,
    TOKENIZER_USE_FAST as DOLCI32K_TOKENIZER_USE_FAST,
    artifact_relative_path as dolci32k_relative_path,
)
from SFT.data.dolci32k.tokenization import (
    assert_no_zero_supervision as assert_dolci32k_no_zero_supervision,
    load_tokenization_report as load_dolci32k_tokenization_report,
    tokenization_report_path as dolci32k_tokenization_report_path,
)
from SFT.data.target_candidates import (
    candidates_manifest_path as target_candidates_manifest_path,
    candidates_path as target_candidates_path,
    load_candidate_groups as load_target_candidate_groups,
)
from SFT.data.target_signal_dataset import build_target_signal_features
from SFT.train.target_signal import (
    ANSWER_ONLY_CE as TARGET_SIGNAL_ANSWER_ONLY_CE,
    NLL as TARGET_SIGNAL_NLL,
    GroupedTargetCollator,
    canonicalize_target_signal_mode,
)

from drpt import (
    GradientHook,
    setup_model_compressors,
    create_sample_inputs,
    CompressionMode,
)
from SFT.train.trainer import LayerWiseSubsetTrainer
from SFT.train.collator import MetaIdxCollator

from SFT.train.data_arguments import DataArguments, get_data_statistics
from SFT.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from SFT.train.training_arguments import TrainingArguments


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _report_to_includes_wandb(report_to):
    if report_to is None:
        return False
    if isinstance(report_to, str):
        targets = [x.strip().lower() for x in report_to.replace(",", " ").split()]
    else:
        targets = [str(x).strip().lower() for x in report_to]
    return "wandb" in targets or "all" in targets


def _split_wandb_tags(raw_tags):
    if not raw_tags:
        return []
    return [tag.strip() for tag in str(raw_tags).split(",") if tag.strip()]


def _new_solver_config(training_args):
    """Serializable solver settings shared by local metadata and W&B."""
    return {
        "soft_weighting_steps": training_args.soft_weighting_steps,
        "soft_weighting_lr": training_args.soft_weighting_lr,
        "soft_weighting_tol": training_args.soft_weighting_tol,
        "soft_weighting_patience": training_args.soft_weighting_patience,
        "soft_weighting_gamma": training_args.soft_weighting_gamma,
        "soft_weighting_regularizer": "none",
        "soft_weighting_use_optimizer_state": training_args.soft_weighting_use_optimizer_state,
        "soft_weighting_constraint": training_args.soft_weighting_constraint,
        "soft_replay_precision": training_args.soft_replay_precision,
        "muon_surrogate_alpha": training_args.muon_surrogate_alpha,
        "muon_surrogate_rank": training_args.muon_surrogate_rank,
        "muon_surrogate_full_svd_max_dim": training_args.muon_surrogate_full_svd_max_dim,
        "muon_surrogate_rtol": training_args.muon_surrogate_rtol,
        "muon_surrogate_oversample": training_args.muon_surrogate_oversample,
        "muon_surrogate_power_iters": training_args.muon_surrogate_power_iters,
        "muon_surrogate_include_adamw_scores": (
            training_args.muon_surrogate_include_adamw_scores
        ),
        "muon_surrogate_mode_weighting": (
            training_args.muon_surrogate_mode_weighting
        ),
        "muon_surrogate_saturation": training_args.muon_surrogate_saturation,
        "optimizer_aware_token_normalized_selection": (
            training_args.optimizer_aware_token_normalized_selection
        ),
    }


def _process_peak_rss_bytes():
    """Return this process's high-water RSS in bytes when available.

    ``ru_maxrss`` is reported in KiB on Linux (and bytes on macOS).  The
    dolci32k campaigns run on Linux, but keeping the conversion explicit avoids
    silently recording KiB under a bytes-labelled field in local tests.
    """
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None
    scale = 1024 if sys.platform.startswith("linux") else 1
    return int(peak_rss * scale)


def _cuda_peak_memory_bytes():
    """Return peak CUDA allocator counters for the current device."""
    if not torch.cuda.is_available():
        return None, None
    try:
        device_index = torch.cuda.current_device()
        return (
            int(torch.cuda.max_memory_allocated(device_index)),
            int(torch.cuda.max_memory_reserved(device_index)),
        )
    except (AssertionError, RuntimeError):
        # Metadata collection must not turn an otherwise successful run into a
        # failure when a CUDA context is already shutting down.
        logger.warning("Unable to read CUDA peak-memory counters", exc_info=True)
        return None, None


def _reset_cuda_peak_memory_stats():
    """Start the measured run window after model/optimizer construction."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        logger.warning("Unable to reset CUDA peak-memory counters", exc_info=True)


def _runtime_environment(training_args):
    cuda_peak_allocated, cuda_peak_reserved = _cuda_peak_memory_bytes()
    payload = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "device": str(getattr(training_args, "device", "unknown")),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "cuda_max_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_max_memory_reserved_bytes": cuda_peak_reserved,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        payload.update({
            "gpu_index": device_index,
            "gpu_name": torch.cuda.get_device_name(device_index),
            "gpu_capability": list(torch.cuda.get_device_capability(device_index)),
        })
    return payload


def _write_run_metadata(
    training_args,
    model_args,
    data_args,
    optimizer=None,
    profile_metadata=None,
    lifecycle_phase=None,
):
    """Persist the run-defining solver configuration independently of W&B."""
    if getattr(training_args, "local_rank", -1) not in (-1, 0):
        return
    payload = {
        "method": training_args.method,
        "optimizer_type": training_args.optimizer_type,
        "learning_rate": training_args.learning_rate,
        "muon_learning_rate_requested": getattr(
            training_args, "muon_learning_rate", None
        ),
        "aux_adamw_learning_rate_requested": (
            getattr(training_args, "aux_adamw_learning_rate", None)
        ),
        "muon_backend_requested": (
            getattr(training_args, "optimizer_aware_muon_backend", "auto")
        ),
        "selection_frac": training_args.selection_frac,
        "selection_mode": training_args.selection_mode,
        "scoring_method": training_args.scoring_method,
        "subset_mode": training_args.subset_mode,
        "val_strategy": training_args.val_strategy,
        "seed": training_args.seed,
        "data_seed": (
            getattr(training_args, "data_seed", None)
            if getattr(training_args, "data_seed", None) is not None
            else training_args.seed
        ),
        "model_name_or_path": model_args.model_name_or_path,
        "model_revision": getattr(model_args, "model_revision", None),
        "tokenizer_name": (
            getattr(model_args, "tokenizer_name", None)
            or model_args.model_name_or_path
        ),
        "tokenizer_revision": (
            getattr(model_args, "tokenizer_revision", None)
            or getattr(model_args, "model_revision", None)
        ),
        "use_fast_tokenizer": getattr(model_args, "use_fast_tokenizer", None),
        "model_profile": getattr(model_args, "model_profile", None),
        "experiment_profile": getattr(data_args, "experiment_profile", None),
        "setting_id": getattr(data_args, "setting_id", None),
        "artifact_build_id": getattr(data_args, "artifact_build_id", None),
        "train_dataset_names": training_args.train_dataset_names,
        "target_task": training_args.analysis_dataset,
        "max_seq_length": data_args.max_seq_length,
        # The objective whose gradient the selection scores against. Analysis
        # tooling groups runs by this: two runs identical everywhere else but
        # here are optimizing different things.
        "target_signal_mode": getattr(training_args, "target_signal_mode", None),
        "target_signal_beta": getattr(training_args, "target_signal_beta", None),
        "target_signal_margin": getattr(training_args, "target_signal_margin", None),
        "target_signal_incorrect_reward": (
            getattr(training_args, "target_signal_incorrect_reward", None)
        ),
        "runtime_environment": _runtime_environment(training_args),
    }
    if profile_metadata is not None:
        payload["profile"] = profile_metadata
    if lifecycle_phase is not None:
        payload["metadata_lifecycle_phase"] = lifecycle_phase
    payload.update(_new_solver_config(training_args))
    if optimizer is not None:
        if hasattr(optimizer, "get_runtime_metadata"):
            payload.update(optimizer.get_runtime_metadata())
        else:
            payload["optimizer_runtime_class"] = (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            )
    os.makedirs(training_args.output_dir, exist_ok=True)
    metadata_path = os.path.join(training_args.output_dir, "run_metadata.json")
    temporary_path = f"{metadata_path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, metadata_path)
    logger.info(f"Saved run metadata to {metadata_path}")


def _require_dolci32k_contract(model_args, data_args, training_args):
    if data_args.setting_id not in DOLCI32K_SETTINGS:
        raise ValueError(
            f"Unknown dolci32k setting {data_args.setting_id!r}; expected one of "
            f"{tuple(DOLCI32K_SETTINGS)}"
        )
    model_profile = DOLCI32K_MODEL_PROFILES.get(model_args.model_profile)
    if model_profile is None:
        raise ValueError(
            f"Unknown dolci32k model profile {model_args.model_profile!r}; "
            f"expected one of {tuple(DOLCI32K_MODEL_PROFILES)}"
        )
    expected = {
        "model_name_or_path": (
            model_args.model_name_or_path,
            model_profile["model_name_or_path"],
        ),
        "model_revision": (
            model_args.model_revision,
            model_profile["model_revision"],
        ),
        "tokenizer_name": (
            model_args.tokenizer_name,
            model_profile["tokenizer_name"],
        ),
        "tokenizer_revision": (
            model_args.tokenizer_revision,
            model_profile["tokenizer_revision"],
        ),
        "use_fast_tokenizer": (
            model_args.use_fast_tokenizer,
            DOLCI32K_TOKENIZER_USE_FAST,
        ),
        "max_seq_length": (data_args.max_seq_length, DOLCI32K_MAX_SEQ_LEN),
        "percentage": (data_args.percentage, 1.0),
        "per_device_train_batch_size": (
            training_args.per_device_train_batch_size,
            16,
        ),
        "gradient_accumulation_steps": (
            training_args.gradient_accumulation_steps,
            1,
        ),
        "n_val": (training_args.n_val, DOLCI32K_SIZES.target_grad),
        "n_target_val": (
            training_args.n_target_val,
            DOLCI32K_SIZES.target_val,
        ),
        "n_eval": (training_args.n_eval, DOLCI32K_SIZES.general_val),
        "val_batch_size_for_selection": (
            training_args.val_batch_size_for_selection,
            2,
        ),
        "logical_candidate_batch_size": (
            training_args.logical_candidate_batch_size,
            16,
        ),
        "target_cache_pin_memory": (
            training_args.target_cache_pin_memory,
            False,
        ),
        "val_strategy": (training_args.val_strategy, "separate_batch"),
        "selection_frac": (training_args.selection_frac, 0.5),
        "selection_mode": (training_args.selection_mode, "topk"),
        "subset_mode": (training_args.subset_mode, "one_pass"),
        "scoring_method": (training_args.scoring_method, "reduced_ghost"),
        "num_train_epochs": (float(training_args.num_train_epochs), 1.0),
        "warmup_ratio": (float(training_args.warmup_ratio), 0.03),
        "weight_decay": (float(training_args.weight_decay), 0.0),
        "eval_steps": (int(training_args.eval_steps), 100),
        "world_size": (int(training_args.world_size), 1),
    }
    failures = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if failures:
        raise ValueError("dolci32k contract violation: " + "; ".join(failures))
    # The microbatch knobs chunk GPU compute only; the logical window pinned
    # above (N=16 candidates, T=2 target) is what fixes selection semantics.
    # tests/test_dolci32k_selection_window.py::...::
    # test_custom_autograd_gradient_and_single_outer_step_match_c16_c2_c1 pins
    # candidate C=16/2/1 and target C=1/2 to identical selected indices,
    # parameter gradients, and post-step parameters, so only the chunk size is
    # free -- restricted to the validated divisors of the logical window.
    candidate_microbatch_size = int(training_args.candidate_microbatch_size)
    c16_probe_enabled = (
        candidate_microbatch_size == 16
        and os.environ.get("DRPT_DOLCI_C16_PROBE") == "1"
        and 0 < int(training_args.max_steps) <= 3
    )
    if candidate_microbatch_size not in (1, 2, 4, 8) and not c16_probe_enabled:
        raise ValueError(
            "dolci32k candidate_microbatch_size must be 1, 2, 4, or 8 "
            "(or C=16 with DRPT_DOLCI_C16_PROBE=1 and max_steps<=3), got "
            f"{training_args.candidate_microbatch_size!r}"
        )
    if int(training_args.target_microbatch_size) not in (1, 2):
        raise ValueError(
            "dolci32k target_microbatch_size must be 1 or 2, got "
            f"{training_args.target_microbatch_size!r}"
        )
    # Storage location for the captured fp32 target gradient. cuda keeps a
    # model-sized tensor resident instead of round-tripping it to host memory
    # every step; the values staged into scoring are identical either way.
    if str(training_args.target_cache_device) not in ("cpu", "cuda"):
        raise ValueError(
            "dolci32k target_cache_device must be cpu or cuda, got "
            f"{training_args.target_cache_device!r}"
        )
    if not training_args.dataloader_drop_last:
        raise ValueError("dolci32k exact N=16 windows require dataloader_drop_last=true")
    if not training_args.bf16 or not training_args.tf32 or training_args.fp16:
        raise ValueError("dolci32k requires bf16=true, tf32=true, and fp16=false")
    if not training_args.gradient_checkpointing:
        raise ValueError("dolci32k requires non-reentrant gradient checkpointing")
    checkpoint_kwargs = training_args.gradient_checkpointing_kwargs or {}
    if checkpoint_kwargs.get("use_reentrant") is not False:
        raise ValueError(
            "dolci32k requires gradient_checkpointing_kwargs.use_reentrant=false"
        )
    if model_args.lora:
        raise ValueError("dolci32k requires full-parameter SFT (lora=false)")
    if not model_args.use_flash_attention:
        raise ValueError("dolci32k requires FlashAttention2")
    scheduler = getattr(
        training_args.lr_scheduler_type, "value", training_args.lr_scheduler_type
    )
    if str(scheduler).lower() != "linear":
        raise ValueError("dolci32k lr_scheduler_type must be linear")
    save_strategy = getattr(
        training_args.save_strategy, "value", training_args.save_strategy
    )
    if str(save_strategy).lower() != "no":
        raise ValueError("dolci32k saves only the final model (save_strategy=no)")
    max_steps = int(training_args.max_steps)
    if max_steps == 0 or max_steps < -1 or max_steps > 2000:
        raise ValueError(
            "dolci32k max_steps must be -1 (one formal epoch), 2000, or a "
            "positive smoke-prefix value below 2000"
        )
    if 0 < max_steps < 2000:
        logger.warning(
            "dolci32k smoke-prefix run: max_steps=%s; the stored order remains "
            "complete but only its prefix will be consumed",
            max_steps,
        )

    adamw_methods = {
        "NA", "LayerWiseSubset", "LayerWiseSoftWeighting",
        "LayerWiseSoftProbability", "LayerWiseOptimizerAwareSubset",
        # Architecture-axis controls: the same curated-data setting as
        # LayerWiseSubset/LayerWiseOptimizerAwareSubset but with one subset
        # shared by every layer, which isolates layer-wise selection from data
        # selection. Run by SFT/train/submit_dolci32k_global_axis.sh. These are
        # deliberately absent from profile.ADAMW_METHODS so the main campaign's
        # 0-24 array contract is unchanged.
        "GlobalSubset", "OptimizerAwareGlobalSubset",
    }
    muon_methods = {
        "NA", "LayerWiseSubset", "LayerWiseSoftWeighting",
        "LayerWiseSoftProbability", "LayerWiseMuonMatrixSpectral",
        "LayerWiseMuonMatrixSpectralP", "LayerWiseMuonMatrixSpectralSat",
        "LayerWiseMuonMatrixSpectralSatP",
    }
    if training_args.optimizer_type == "adamw":
        if training_args.method not in adamw_methods:
            raise ValueError(
                f"dolci32k AdamW method {training_args.method!r} is not one of "
                f"{sorted(adamw_methods)}"
            )
        if abs(float(training_args.learning_rate) - 1e-5) > 1e-12:
            raise ValueError("dolci32k AdamW learning_rate must be 1e-5")
    elif training_args.optimizer_type == "muon":
        if training_args.method not in muon_methods:
            raise ValueError(
                f"dolci32k Muon method {training_args.method!r} is not one of "
                f"{sorted(muon_methods)}"
            )
        if training_args.muon_learning_rate is None or abs(
            float(training_args.muon_learning_rate) - 3e-4
        ) > 1e-12:
            raise ValueError("dolci32k Muon matrix learning rate must be 3e-4")
        if training_args.aux_adamw_learning_rate is None or abs(
            float(training_args.aux_adamw_learning_rate) - 1e-5
        ) > 1e-12:
            raise ValueError("dolci32k auxiliary AdamW learning rate must be 1e-5")
    else:
        raise ValueError("dolci32k optimizer_type must be adamw or muon")


def _load_jsonl_identity(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            record = json.loads(line)
            rows.append({
                "id": str(record.get("id", f"row_{row_index}")),
                "source_dataset": str(record.get("source_dataset", "unknown")),
            })
    return rows


def _validate_dolci32k_tokenization_preflight(
    model_args, data_args, manifest
):
    """Bind runtime formatting to a complete, audited derived cache."""

    report_path = dolci32k_tokenization_report_path(
        data_args.data_dir,
        manifest["build_id"],
        data_args.max_seq_length,
    )
    if not report_path.is_file():
        raise RuntimeError(
            "Dolci32K tokenization diagnostics are required before training: "
            f"{report_path}. Generate them with `python "
            "SFT/data/prepare_dolci32k.py --audit-only "
            "--profile-tokenizers all`."
        )
    report = load_dolci32k_tokenization_report(
        report_path,
        raw_build_id=manifest["build_id"],
        max_seq_len=data_args.max_seq_length,
    )
    alias = str(model_args.model_profile)
    try:
        bundle = report["profiles"][alias]
    except KeyError as exc:
        raise RuntimeError(
            f"Dolci32K tokenization report has no {alias!r} entry; regenerate "
            "with --profile-tokenizers all"
        ) from exc
    requested = bundle.get("requested_tokenizer_profile") or {}
    canonical = DOLCI32K_MODEL_PROFILES[alias]
    for field in (
        "model_name_or_path",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
    ):
        if requested.get(field) != canonical[field]:
            raise RuntimeError(
                "Dolci32K tokenization report uses a stale model/tokenizer "
                f"profile for {alias!r}: {field}={requested.get(field)!r}, "
                f"expected {canonical[field]!r}"
            )
    if requested.get("use_fast") is not DOLCI32K_TOKENIZER_USE_FAST:
        raise RuntimeError(
            "Dolci32K tokenization report uses the wrong tokenizer "
            f"implementation: use_fast={requested.get('use_fast')!r}, "
            f"expected {DOLCI32K_TOKENIZER_USE_FAST!r}"
        )
    setting = DOLCI32K_SETTINGS[data_args.setting_id]
    pool = str(setting["general_pool"])
    target = str(setting["target"])
    required_artifacts = (
        f"general/{pool}/train",
        f"general/{pool}/val",
        f"targets/{target}/grad",
        f"targets/{target}/val",
    )
    artifact_statistics = (bundle.get("statistics") or {}).get("artifacts") or {}
    missing = sorted(set(required_artifacts) - set(artifact_statistics))
    if missing:
        raise RuntimeError(
            "Dolci32K tokenization report is incomplete for this setting: "
            f"{missing}"
        )
    assert_dolci32k_no_zero_supervision(
        bundle, artifact_names=required_artifacts
    )
    return {
        "report_path": str(report_path),
        "report_sha256": dolci32k_file_sha256(report_path),
        "model_profile": alias,
        "cache_key": bundle["cache_key"],
        "tokenizer_fingerprint": bundle["tokenizer_fingerprint"],
        "parquet_path": bundle["parquet_path"],
        "parquet_sha256": bundle["parquet_sha256"],
        "statistics": {
            name: artifact_statistics[name] for name in required_artifacts
        },
    }


def _load_manifest_order(order_path, pool_metadata, manifest_entry):
    metadata_ids = [str(row["id"]) for row in pool_metadata]
    if len(set(metadata_ids)) != len(metadata_ids):
        raise RuntimeError("dolci32k train artifact contains duplicate stable IDs")
    row_by_id = {example_id: index for index, example_id in enumerate(metadata_ids)}
    ordered_ids = []
    with open(order_path, "r", encoding="utf-8") as handle:
        for expected_position, line in enumerate(handle):
            record = json.loads(line)
            if record.get("position") != expected_position:
                raise RuntimeError(
                    "dolci32k candidate-order position mismatch at row "
                    f"{expected_position}: {record.get('position')!r}"
                )
            ordered_ids.append(str(record["id"]))
    if len(ordered_ids) != len(metadata_ids):
        raise RuntimeError(
            "dolci32k candidate order must cover all training rows: "
            f"order={len(ordered_ids)}, train={len(metadata_ids)}"
        )
    if len(set(ordered_ids)) != len(ordered_ids):
        raise RuntimeError("dolci32k candidate order contains duplicate IDs")
    missing = set(metadata_ids) - set(ordered_ids)
    unknown = set(ordered_ids) - set(metadata_ids)
    if missing or unknown:
        raise RuntimeError(
            "dolci32k candidate order and train IDs differ: "
            f"missing={sorted(missing)[:5]}, unknown={sorted(unknown)[:5]}"
        )
    digest = hashlib.sha256()
    for example_id in ordered_ids:
        digest.update(example_id.encode("utf-8"))
        digest.update(b"\n")
    ordered_id_sha256 = digest.hexdigest()
    expected_hash = manifest_entry.get("ordered_id_sha256")
    if not expected_hash or expected_hash != ordered_id_sha256:
        raise RuntimeError(
            "dolci32k candidate-order ID hash mismatch: "
            f"computed={ordered_id_sha256}, manifest={expected_hash!r}"
        )
    return [row_by_id[example_id] for example_id in ordered_ids], ordered_id_sha256


def _assert_nonzero_supervision(dataset, identities, role, model_profile):
    bad = []
    sources = Counter()
    for row_index, labels in enumerate(dataset["labels"]):
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().tolist()
        if not any(int(label) != -100 for label in labels[1:]):
            identity = identities[row_index]
            bad.append(identity["id"])
            sources[identity["source_dataset"]] += 1
    if bad:
        raise RuntimeError(
            f"dolci32k {model_profile} produces zero supervised tokens after "
            f"right truncation for {len(bad)} {role} rows; "
            f"source_counts={dict(sorted(sources.items()))}; ids={bad[:20]}"
        )


def _load_dolci32k_roles(model_args, data_args, training_args, tokenizer):
    _require_dolci32k_contract(model_args, data_args, training_args)
    build = resolve_dolci32k_build(data_args.data_dir, data_args.artifact_build_id)
    manifest = validate_dolci32k_build(build)
    tokenization_preflight = _validate_dolci32k_tokenization_preflight(
        model_args, data_args, manifest
    )
    data_args.artifact_build_id = manifest["build_id"]
    setting = DOLCI32K_SETTINGS[data_args.setting_id]
    pool = str(setting["general_pool"])
    target = str(setting["target"])
    if training_args.analysis_dataset != target:
        raise ValueError(
            f"dolci32k setting {data_args.setting_id} requires "
            f"target_task={target!r}; got {training_args.analysis_dataset!r}"
        )

    def path(role, name):
        return build / dolci32k_relative_path(role, name)

    general_train_path = path("general", f"{pool}/train")
    general_val_path = path("general", f"{pool}/val")
    target_grad_path = path("targets", f"{target}/grad")
    target_val_path = path("targets", f"{target}/val")
    order_relative = dolci32k_relative_path("candidate_orders", pool)
    order_path = build / order_relative
    train_dataset, pool_metadata, train_stats = get_training_dataset(
        data_dir=data_args.data_dir,
        task=target,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=1.0,
        seed=(
            training_args.data_seed
            if training_args.data_seed is not None
            else training_args.seed
        ),
        train_files=[str(general_train_path)],
        return_source_metadata=True,
        return_tokenization_stats=True,
    )
    if len(train_dataset) != DOLCI32K_SIZES.general_train:
        raise RuntimeError(
            f"dolci32k train pool must contain 32000 rows, got {len(train_dataset)}"
        )
    order_indices, ordered_id_sha256 = _load_manifest_order(
        order_path,
        pool_metadata,
        manifest["artifacts"][str(order_relative)],
    )
    if len(order_indices) // 16 != 2000:
        raise RuntimeError("dolci32k stored traversal must form exactly 2000 N=16 windows")
    general_val_dataset, general_val_stats = get_messages_file_dataset(
        str(general_val_path), tokenizer, data_args.max_seq_length,
        expected_count=DOLCI32K_SIZES.general_val,
        return_tokenization_stats=True,
    )
    target_grad_dataset, target_grad_stats = get_messages_file_dataset(
        str(target_grad_path), tokenizer, data_args.max_seq_length,
        expected_count=DOLCI32K_SIZES.target_grad,
        return_tokenization_stats=True,
    )
    target_val_dataset, target_val_stats = get_messages_file_dataset(
        str(target_val_path), tokenizer, data_args.max_seq_length,
        expected_count=DOLCI32K_SIZES.target_val,
        return_tokenization_stats=True,
    )
    role_datasets = {
        "general_train": (train_dataset, _load_jsonl_identity(general_train_path)),
        "general_val": (general_val_dataset, _load_jsonl_identity(general_val_path)),
        "target_grad": (target_grad_dataset, _load_jsonl_identity(target_grad_path)),
        "target_val": (target_val_dataset, _load_jsonl_identity(target_val_path)),
    }
    for role, (dataset, identities) in role_datasets.items():
        _assert_nonzero_supervision(
            dataset, identities, role, model_args.model_profile
        )

    relative_paths = [
        dolci32k_relative_path("general", f"{pool}/train"),
        dolci32k_relative_path("general", f"{pool}/val"),
        dolci32k_relative_path("targets", f"{target}/grad"),
        dolci32k_relative_path("targets", f"{target}/val"),
        order_relative,
    ]
    profile_metadata = {
        "name": manifest["profile_name"],
        "version": manifest["profile_version"],
        "profile_fingerprint": manifest["profile_fingerprint"],
        "build_id": manifest["build_id"],
        "manifest_sha256": dolci32k_file_sha256(build / "manifest.json"),
        "model_profile": model_args.model_profile,
        "general_pool": pool,
        "target": target,
        "final_benchmarks": list(setting["benchmarks"]),
        "training_process_loaded_final_benchmarks": False,
        "source_revisions": manifest["sources"],
        "candidate_traversal": {
            "count": len(order_indices),
            "batch_size": 16,
            "batches": 2000,
            "ordered_id_sha256": ordered_id_sha256,
            "artifact": str(order_relative),
        },
        "tokenization_preflight": tokenization_preflight,
        "artifacts": {
            str(relative): manifest["artifacts"][str(relative)]
            for relative in relative_paths
        },
        "loader_tokenization": {
            "general_train": train_stats,
            "general_val": general_val_stats,
            "target_grad": target_grad_stats,
            "target_val": target_val_stats,
        },
    }
    return (
        train_dataset,
        pool_metadata,
        target_grad_dataset,
        target_val_dataset,
        general_val_dataset,
        order_indices,
        profile_metadata,
    )


def _build_target_signal_dataset(data_args, training_args, tokenizer):
    """Rebuild the target-gradient dataset for a non-default target signal.

    Returns ``(dataset, collator, stats)``, or ``(None, None, None)`` for the
    historical ``nll`` signal so that path keeps its own loader, collator, and
    loss verbatim. Everything read here is a pre-generated offline artifact:
    switching signals never adds a rollout to the training loop.
    """

    mode = canonicalize_target_signal_mode(training_args.target_signal_mode)
    align_prompts = bool(getattr(training_args, "target_signal_align_prompts", False))
    if mode == TARGET_SIGNAL_NLL and not align_prompts:
        return None, None, None

    setting = DOLCI32K_SETTINGS[data_args.setting_id]
    target = str(setting["target"])
    build = resolve_dolci32k_build(data_args.data_dir, data_args.artifact_build_id)
    grad_path = build / dolci32k_relative_path("targets", f"{target}/grad")
    rows = [json.loads(line) for line in grad_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != DOLCI32K_SIZES.target_grad:
        raise RuntimeError(
            f"target grad split must contain {DOLCI32K_SIZES.target_grad} rows, "
            f"got {len(rows)}"
        )

    candidate_groups = None
    candidates_file = target_candidates_path(
        data_args.data_dir, data_args.artifact_build_id, target
    )
    if candidates_file.exists():
        candidate_groups = load_target_candidate_groups(candidates_file)
    elif align_prompts or mode != TARGET_SIGNAL_ANSWER_ONLY_CE:
        raise FileNotFoundError(
            f"target_signal_mode={mode!r} needs pre-generated candidates at "
            f"{candidates_file}. Build them once with:\n"
            f"  python -m SFT.data.build_target_candidates --target {target} "
            f"--artifact_build_id {data_args.artifact_build_id}"
        )

    features, stats = build_target_signal_features(
        mode,
        rows,
        tokenizer,
        data_args.max_seq_length,
        target=target,
        candidate_groups=candidate_groups,
        incorrect_reward=float(training_args.target_signal_incorrect_reward),
        max_candidates_per_prompt=training_args.target_signal_max_candidates,
        align_prompts=align_prompts,
    )
    collator = GroupedTargetCollator(pad_token_id=int(tokenizer.pad_token_id))
    manifest_file = target_candidates_manifest_path(
        data_args.data_dir, data_args.artifact_build_id, target
    )
    stats = dict(stats)
    stats["candidates_artifact"] = str(candidates_file) if candidate_groups else None
    if candidate_groups and manifest_file.exists():
        stats["candidates_manifest"] = json.loads(manifest_file.read_text(encoding="utf-8"))
    logger.info("target-gradient signal %s: %s", mode, json.dumps(stats, default=str))
    return features, collator, stats


def _preflight_dolci32k(model_args, data_args, training_args):
    """Fail before W&B, output writes, tokenizer download, or model allocation."""
    _require_dolci32k_contract(model_args, data_args, training_args)
    build = resolve_dolci32k_build(data_args.data_dir, data_args.artifact_build_id)
    manifest = validate_dolci32k_build(build)
    _validate_dolci32k_tokenization_preflight(model_args, data_args, manifest)
    data_args.artifact_build_id = manifest["build_id"]
    return manifest


def _setup_wandb(training_args, model_args, data_args):
    if not _report_to_includes_wandb(getattr(training_args, "report_to", None)):
        return False
    if wandb is None:
        raise ImportError(
            "report_to includes 'wandb', but the wandb package is not installed. "
            "Install it or run with REPORT_TO=none / --report_to none."
        )

    is_main_process = getattr(training_args, "local_rank", -1) in (-1, 0)
    if not is_main_process:
        return False

    project = training_args.wandb_project or os.environ.get("WANDB_PROJECT") or "drpt-opus-sft"
    run_name = training_args.wandb_run_name or getattr(training_args, "run_name", None)
    group = training_args.wandb_group or os.environ.get("WANDB_GROUP")
    tags = _split_wandb_tags(training_args.wandb_tags or os.environ.get("WANDB_TAGS"))
    tags.extend([
        f"method:{training_args.method}",
        f"optimizer:{training_args.optimizer_type}",
        f"target:{training_args.analysis_dataset}",
        f"val_strategy:{training_args.val_strategy}",
    ])
    tags = list(dict.fromkeys(tags))

    if run_name:
        training_args.run_name = run_name
    os.environ.setdefault("WANDB_PROJECT", project)

    config = {
        "method": training_args.method,
        "optimizer_type": training_args.optimizer_type,
        "learning_rate": training_args.learning_rate,
        "muon_learning_rate_requested": getattr(
            training_args, "muon_learning_rate", None
        ),
        "aux_adamw_learning_rate_requested": (
            getattr(training_args, "aux_adamw_learning_rate", None)
        ),
        "optimizer_aware_matrix_geometry": training_args.optimizer_aware_matrix_geometry,
        "optimizer_aware_vector_geometry": training_args.optimizer_aware_vector_geometry,
        "optimizer_aware_target_mode": training_args.optimizer_aware_target_mode,
        "optimizer_aware_muon_backend": training_args.optimizer_aware_muon_backend,
        "optimizer_aware_muon_nesterov": training_args.optimizer_aware_muon_nesterov,
        "optimizer_aware_muon_adjust_lr_fn": training_args.optimizer_aware_muon_adjust_lr_fn,
        "optimizer_aware_muon_steps": training_args.optimizer_aware_muon_steps,
        "scoring_method": training_args.scoring_method,
        "selection_frac": training_args.selection_frac,
        "selection_mode": training_args.selection_mode,
        "val_strategy": training_args.val_strategy,
        "subset_mode": training_args.subset_mode,
        "train_dataset_names": training_args.train_dataset_names,
        "target_task": training_args.analysis_dataset,
        "model_name_or_path": model_args.model_name_or_path,
        "max_seq_length": data_args.max_seq_length,
        "eval_split": data_args.eval_split,
        "n_val": training_args.n_val,
        "n_eval": training_args.n_eval,
        "seed": training_args.seed,
        "data_seed": (
            training_args.data_seed
            if training_args.data_seed is not None
            else training_args.seed
        ),
    }
    config.update(_new_solver_config(training_args))

    if wandb.run is None:
        wandb.init(
            project=project,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
        )
    else:
        wandb.config.update(config, allow_val_change=True)

    logger.info(
        "Weights & Biases logging enabled: "
        f"project={project}, run={run_name}, group={group}, tags={tags}"
    )
    return True


def find_trainable_layers(model, lora_only=True):
    """
    Find trainable layers in the model.

    Args:
        model: The model to search
        lora_only: If True, only find LoRA layers (lora_A and lora_B). If False, find all Linear layers.

    Returns:
        List of layer names
        - For LoRA: returns paths to lora_A and lora_B modules (e.g., "layer.lora_A.default", "layer.lora_B.default")
        - For full fine-tuning: returns paths to all Linear layers
    """
    layer_names = []

    for name, module in model.named_modules():
        if lora_only:
            # Find LoRA adapter layers - these are the actual trainable parameters
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # This is a PEFT LoRA wrapper
                # lora_A and lora_B can be either ModuleDict or direct nn.Linear

                # Handle lora_A
                if hasattr(module.lora_A, 'default'):
                    # ModuleDict case: lora_A['default']
                    lora_a_name = f"{name}.lora_A.default"
                    layer_names.append(lora_a_name)
                elif isinstance(module.lora_A, torch.nn.Linear):
                    # Direct nn.Linear case
                    lora_a_name = f"{name}.lora_A"
                    layer_names.append(lora_a_name)

                # Handle lora_B
                if hasattr(module.lora_B, 'default'):
                    # ModuleDict case: lora_B['default']
                    lora_b_name = f"{name}.lora_B.default"
                    layer_names.append(lora_b_name)
                elif isinstance(module.lora_B, torch.nn.Linear):
                    # Direct nn.Linear case
                    lora_b_name = f"{name}.lora_B"
                    layer_names.append(lora_b_name)
        else:
            # Find all Linear and Embedding layers (for full fine-tuning)
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                layer_names.append(name)

    return layer_names


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.experiment_profile == "dolci32k":
        _preflight_dolci32k(model_args, data_args, training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    wandb_enabled = _setup_wandb(training_args, model_args, data_args)
    _write_run_metadata(training_args, model_args, data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        revision=model_args.tokenizer_revision or model_args.model_revision,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    native_chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    ensure_chat_template(tokenizer)
    if data_args.experiment_profile == "dolci32k":
        if not native_chat_template:
            raise RuntimeError(
                "The pinned dolci32k tokenizer must provide its own chat template"
            )
        if model_args.model_profile in {"qwen3_1_7b", "qwen3_4b", "qwen3_8b"}:
            template = str(getattr(tokenizer, "chat_template", ""))
            if "<|im_start|>" not in template or "<|im_end|>" not in template:
                raise RuntimeError(
                    "The pinned Qwen dolci32k tokenizer must expose its native "
                    "<|im_start|>/<|im_end|> chat template"
                )
    # Load training dataset. The source/domain sidecar rides alongside via a
    # `meta_idx` column, which requires the Trainer to stop pruning columns it
    # does not recognize; the MetaIdxCollator keeps `meta_idx` out of the model.
    profile_metadata = None
    target_grad_dataset = None
    target_val_dataset = None
    general_val_dataset = None
    train_order_indices = None
    target_signal_collator = None
    if data_args.experiment_profile == "dolci32k":
        (
            train_dataset,
            pool_metadata,
            target_grad_dataset,
            target_val_dataset,
            general_val_dataset,
            train_order_indices,
            profile_metadata,
        ) = _load_dolci32k_roles(
            model_args, data_args, training_args, tokenizer
        )
        # An alternate target signal replaces only D*: the candidate pool, the
        # general validation split, and the immutable traversal above are
        # untouched, so the comparison isolates the target gradient.
        signal_dataset, target_signal_collator, target_signal_stats = (
            _build_target_signal_dataset(data_args, training_args, tokenizer)
        )
        if signal_dataset is not None:
            target_grad_dataset = signal_dataset
        profile_metadata["target_signal"] = target_signal_stats or {
            "mode": TARGET_SIGNAL_NLL
        }
        _write_run_metadata(
            training_args,
            model_args,
            data_args,
            profile_metadata=profile_metadata,
        )
    else:
        if canonicalize_target_signal_mode(training_args.target_signal_mode) != TARGET_SIGNAL_NLL:
            # The alternate signals are defined against the dolci32k target
            # splits and their candidate artifacts. Silently ignoring the flag
            # here would report a signal the run never actually optimized.
            raise ValueError(
                f"target_signal_mode={training_args.target_signal_mode!r} is only "
                f"supported by experiment_profile=dolci32k, got "
                f"{data_args.experiment_profile!r}"
            )
        train_dataset, pool_metadata = get_training_dataset(
            data_dir=data_args.data_dir,
            task=training_args.analysis_dataset,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            seed=training_args.data_seed if training_args.data_seed is not None else training_args.seed,
            train_files=data_args.train_files if data_args.train_files else None,
            train_dataset_names=training_args.train_dataset_names,
            return_source_metadata=True,
        )
    training_args.remove_unused_columns = False

    avg_train_seq_length = get_data_statistics(train_dataset, return_avg_length=True)
    logger.info(f"Average training sequence length: {avg_train_seq_length:.1f}")

    # Load model - NO CUSTOM LAYER REPLACEMENT!
    # Auto-derive model dtype from training precision if not explicitly set
    # This ensures model dtype matches training precision (critical for flash attention)
    if model_args.torch_dtype is None and (training_args.bf16 or training_args.fp16):
        model_args.torch_dtype = "bfloat16" if training_args.bf16 else "float16"
        logger.info(f"Auto-setting model dtype to {model_args.torch_dtype} to match training precision")

    model_kwargs = {
        "torch_dtype": model_args.torch_dtype,
        "revision": model_args.model_revision,
        "cache_dir": model_args.cache_dir,
    }
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs)
    if data_args.experiment_profile == "dolci32k":
        model.config.use_cache = False
    add_padding_to_tokenizer(tokenizer)

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    # Apply LoRA using standard PEFT (no custom layers!)
    if not isinstance(model, PeftModel) and model_args.lora:
        target_modules = model_args.lora_target_modules
        if isinstance(target_modules, list) and target_modules == ["all-linear"]:
            target_modules = "all-linear"
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA to model using PEFT.")
        model.print_trainable_parameters()

        # For checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Find trainable layers (LoRA layers or all Linear layers)
    layer_names = find_trainable_layers(model, lora_only=model_args.lora)

    # Log layer names for verification
    if len(layer_names) > 0:
        logger.info("=== Layer Names for Hook Attachment ===")
        for i, name in enumerate(layer_names[:5]):  # Show first 5
            logger.info(f"  [{i}] {name}")
        if len(layer_names) > 5:
            logger.info(f"  ... and {len(layer_names) - 5} more layers")
    else:
        logger.warning("WARNING: No trainable layers found! Check model and lora_only setting.")

    # Determine if gradient hooks are needed based on training method
    # Hooks are needed for:
    # 1. Selection method is not NA (LayerWiseSubset or GlobalSubset)
    # 2. Compression enabled (implies MeSO optimizer)
    has_explicit_compression = (training_args.sparsification is not None or training_args.projection is not None)
    optimizer_aware_methods = (
        'LayerWiseOptimizerAwareSubset',
        'OptimizerAwareGlobalSubset',
        'OptimizerAwareGroupWise',
        'GlobalSoftWeighting',
        'LayerWiseSoftWeighting',
        'LayerWiseSoftProbability',
        'GlobalMuonSpectral',
        'LayerWiseMuonSpectral',
        'GlobalMuonMatrixSpectral',
        'LayerWiseMuonMatrixSpectral',
        'LayerWiseMuonMatrixSpectralP',
        'LayerWiseMuonMatrixSpectralSat',
        'LayerWiseMuonMatrixSpectralSatP',
    )
    selection_methods = (
        'LayerWiseSubset',
        'LayerWiseOptimizerAwareSubset',
        'GlobalSubset',
        'OptimizerAwareGlobalSubset',
        'OptimizerGroupWise',
        'OptimizerAwareGroupWise',
        'GlobalRandomSubset',
        'LayerWiseRandomSubset',
        'GlobalSoftWeighting',
        'LayerWiseSoftWeighting',
        'LayerWiseSoftProbability',
        'GlobalMuonSpectral',
        'LayerWiseMuonSpectral',
        'GlobalMuonMatrixSpectral',
        'LayerWiseMuonMatrixSpectral',
        'LayerWiseMuonMatrixSpectralP',
        'LayerWiseMuonMatrixSpectralSat',
        'LayerWiseMuonMatrixSpectralSatP',
    )
    needs_selection = training_args.method in selection_methods
    needs_grad_hook = needs_selection or has_explicit_compression

    # Determine compression needs independently for score and update
    has_score_compression = (needs_selection
                             and training_args.score_compression is not None)
    has_update_compression = (training_args.sparsification is not None
                              or training_args.projection is not None)

    # Create gradient hook only when needed
    grad_hook = None
    if needs_grad_hook:
        grad_hook = GradientHook(
            model=model,
            layer_names=layer_names,
            device=str(training_args.device),
        )
        if training_args.method in optimizer_aware_methods:
            grad_hook.configure_optimizer_aware(
                optimizer_type=training_args.optimizer_type,
                matrix_geometry=training_args.optimizer_aware_matrix_geometry,
                vector_geometry=training_args.optimizer_aware_vector_geometry,
                target_mode=training_args.optimizer_aware_target_mode,
                adam_eps=training_args.optimizer_aware_adam_eps,
                muon_reference=training_args.optimizer_aware_muon_reference,
                muon_momentum=training_args.optimizer_aware_muon_momentum,
                muon_nesterov=training_args.optimizer_aware_muon_nesterov,
                muon_steps=training_args.optimizer_aware_muon_steps,
                muon_eps=training_args.optimizer_aware_muon_eps,
                muon_max_dim=training_args.optimizer_aware_muon_max_dim,
                muon_lr_shape_scale=training_args.optimizer_aware_muon_lr_shape_scale,
                muon_adjust_lr_fn=training_args.optimizer_aware_muon_adjust_lr_fn,
                muon_backend=training_args.optimizer_aware_muon_backend,
                lora_optimizer=training_args.optimizer_aware_lora_optimizer,
                spectral_lambda=training_args.optimizer_aware_spectral_lambda,
                spectral_eps=training_args.optimizer_aware_spectral_eps,
                token_normalized_selection=(
                    training_args.optimizer_aware_token_normalized_selection
                ),
                soft_weighting_steps=training_args.soft_weighting_steps,
                soft_weighting_lr=training_args.soft_weighting_lr,
                soft_weighting_tol=training_args.soft_weighting_tol,
                soft_weighting_patience=training_args.soft_weighting_patience,
                soft_weighting_gamma=training_args.soft_weighting_gamma,
                soft_weighting_use_optimizer_state=training_args.soft_weighting_use_optimizer_state,
                soft_weighting_constraint=training_args.soft_weighting_constraint,
                muon_surrogate_alpha=training_args.muon_surrogate_alpha,
                muon_surrogate_rank=training_args.muon_surrogate_rank,
                muon_surrogate_full_svd_max_dim=training_args.muon_surrogate_full_svd_max_dim,
                muon_surrogate_rtol=training_args.muon_surrogate_rtol,
                muon_surrogate_oversample=training_args.muon_surrogate_oversample,
                muon_surrogate_power_iters=training_args.muon_surrogate_power_iters,
                muon_surrogate_include_adamw_scores=(
                    training_args.muon_surrogate_include_adamw_scores
                ),
                muon_surrogate_mode_weighting=(
                    training_args.muon_surrogate_mode_weighting
                ),
                muon_surrogate_saturation=(
                    training_args.muon_surrogate_saturation
                ),
            )
            logger.info(
                "Optimizer-aware scoring enabled: "
                f"optimizer_type={training_args.optimizer_type}, "
                f"matrix_geometry={training_args.optimizer_aware_matrix_geometry}, "
                f"vector_geometry={training_args.optimizer_aware_vector_geometry}, "
                f"target_mode={training_args.optimizer_aware_target_mode}, "
                f"muon_reference={training_args.optimizer_aware_muon_reference}, "
                f"muon_nesterov={training_args.optimizer_aware_muon_nesterov}, "
                f"muon_steps={training_args.optimizer_aware_muon_steps}, "
                f"muon_lr_shape_scale={training_args.optimizer_aware_muon_lr_shape_scale}, "
                f"muon_adjust_lr_fn={training_args.optimizer_aware_muon_adjust_lr_fn}, "
                f"muon_backend={training_args.optimizer_aware_muon_backend}, "
                f"lora_optimizer={training_args.optimizer_aware_lora_optimizer}, "
                f"spectral_lambda={training_args.optimizer_aware_spectral_lambda}, "
                f"soft_steps={training_args.soft_weighting_steps}, "
                f"soft_lr={training_args.soft_weighting_lr}, "
                f"soft_gamma={training_args.soft_weighting_gamma}, "
                f"soft_constraint={training_args.soft_weighting_constraint}, "
                f"soft_replay_precision={training_args.soft_replay_precision}, "
                f"muon_surrogate_rank={training_args.muon_surrogate_rank}, "
                f"muon_surrogate_alpha={training_args.muon_surrogate_alpha}, "
                "muon_surrogate_include_adamw_scores="
                f"{training_args.muon_surrogate_include_adamw_scores}, "
                "muon_surrogate_mode_weighting="
                f"{training_args.muon_surrogate_mode_weighting}, "
                "muon_surrogate_saturation="
                f"{training_args.muon_surrogate_saturation}, "
                "token_normalized_selection="
                f"{training_args.optimizer_aware_token_normalized_selection}"
            )
    else:
        logger.info(f"Training method: {training_args.method} - No gradient hooks needed")

    # Helper: parse sparsification string (e.g., "normal-512*512") into kwargs
    def _parse_sparsifier(spec_str):
        method, dim_str = spec_str.split("-")
        assert "*" in dim_str, f"Sparsification dimension must be factorized (e.g., 'normal-64*64'), got '{spec_str}'"
        dim = int(dim_str.split("*")[0])
        return {
            "proj_dim": dim, "proj_max_batch_size": 64,
            "proj_seed": training_args.seed, "device": str(training_args.device),
            "proj_type": method,
        }, f"{method}-{dim}*{dim}"

    def _parse_projector(spec_str):
        method, dim_str = spec_str.split("-")
        assert "*" not in dim_str, f"Projection dimension must not be factorized, got '{spec_str}'"
        dim = int(dim_str)
        return {
            "proj_dim": dim, "proj_max_batch_size": 64,
            "proj_seed": training_args.seed, "device": str(training_args.device),
            "proj_type": method,
        }, f"{method}-{dim}"

    _identity_sparsifier = {
        "proj_dim": -1, "proj_max_batch_size": 64,
        "proj_seed": training_args.seed, "device": str(training_args.device),
        "proj_type": "identity",
    }
    _identity_projector = {
        "proj_dim": -1, "proj_max_batch_size": 64,
        "proj_seed": training_args.seed, "device": str(training_args.device),
        "proj_type": "identity",
    }

    # Set up compressors
    if has_score_compression or has_update_compression:
        logger.info("=== Gradient Compression Setup ===")

        # Create sample inputs (shared for both compressor sets)
        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            device=str(training_args.device)
        )

        # --- Update compressors (MeSO optimizer) ---
        if has_update_compression:
            update_sparsifier_kwargs, update_desc = _parse_sparsifier(training_args.sparsification)
            if training_args.projection is not None:
                update_projector_kwargs, proj_desc = _parse_projector(training_args.projection)
            else:
                update_projector_kwargs = _identity_projector
                proj_desc = "none"
            logger.info(f"  Update compression (MeSO): sparsifier={update_desc}, projector={proj_desc}")

            update_compressors = setup_model_compressors(
                model=model, layer_names=layer_names,
                sparsifier_kwargs=update_sparsifier_kwargs,
                projector_kwargs=update_projector_kwargs,
                sample_inputs=sample_inputs,
                device=str(training_args.device),
                update_freq=training_args.update_compressor_freq,
            )
            grad_hook.set_update_compressors(update_compressors)
        else:
            logger.info("  Update compression: none")

        # --- Score compressors (influence scoring) ---
        if has_score_compression:
            # Check if score compression matches update compression → share objects
            score_spec = training_args.score_compression
            update_spec = training_args.sparsification
            if has_update_compression and score_spec == update_spec:
                logger.info(f"  Score compression: same as update ({score_spec}) → sharing compressors")
                grad_hook.set_score_compressors(grad_hook.update_compressors)
            else:
                score_sparsifier_kwargs, score_desc = _parse_sparsifier(score_spec)
                logger.info(f"  Score compression: sparsifier={score_desc}")

                score_compressors = setup_model_compressors(
                    model=model, layer_names=layer_names,
                    sparsifier_kwargs=score_sparsifier_kwargs,
                    projector_kwargs=_identity_projector,
                    sample_inputs=sample_inputs,
                    device=str(training_args.device),
                    update_freq=training_args.update_compressor_freq,
                )
                grad_hook.set_score_compressors(score_compressors)
        else:
            logger.info("  Score compression: none (exact scoring)")

        logger.info(f"  Compression mode: {grad_hook.compression_mode.value}")
        logger.info("Gradient compression setup completed!")
    else:
        if grad_hook is not None:
            logger.info(f"Compression mode: {grad_hook.compression_mode.value} (no compressors)")
        if needs_selection:
            logger.info("Gradient compression disabled (exact scoring)")
        else:
            logger.info("Gradient compression disabled (no selection and no MeSO)")

    # Validate scoring_method vs compression config (strict — no auto-detection)
    if has_score_compression and training_args.scoring_method != "compress":
        raise ValueError(
            f"scoring.compression is set ({training_args.score_compression}) but "
            f"scoring.method='{training_args.scoring_method}'. "
            f"Set scoring.method to 'compress' or remove scoring.compression."
        )
    if training_args.scoring_method == "compress" and not has_score_compression:
        raise ValueError(
            "scoring.method='compress' requires scoring.compression to be set. "
            "Add 'compression: normal-64*64' under the scoring section."
        )
    if training_args.method in optimizer_aware_methods and has_score_compression:
        raise ValueError(
            f"{training_args.method} currently uses optimizer-aware reduced-ghost "
            "scoring and does not yet wire score_compression/CountSketch into "
            "the dual-probe path. Remove scoring.compression from the method config."
        )
    if training_args.optimizer_type in ("muon", "hybrid") and has_update_compression:
        raise ValueError(
            f"optimizer_type={training_args.optimizer_type!r} is not compatible "
            "with MeSO update compression. "
            "Remove optimizer.compression or set optimizer.type: adamw."
        )

    # Prepare validation dataset (used for data selection in layer_wise_subset descent)
    # Use rejection sampling to filter out validation samples that are significantly
    # longer than the average training sequence length. A non-positive multiplier
    # disables the heuristic entirely, which dolci32k relies on: long reasoning
    # explanations are legitimately far longer than the train
    # average and would otherwise be rejected wholesale.
    seq_length_multiplier = getattr(
        training_args, "val_seq_length_multiplier", DEFAULT_SEQ_LENGTH_MULTIPLIER
    )
    if seq_length_multiplier and seq_length_multiplier > 0:
        val_seq_length_threshold = int(avg_train_seq_length * seq_length_multiplier)
        logger.info(
            f"Validation sequence length threshold: {val_seq_length_threshold} "
            f"({seq_length_multiplier}x avg train length)"
        )
    else:
        val_seq_length_threshold = None
        logger.info(
            "Validation sequence length rejection disabled "
            "(val_seq_length_multiplier <= 0); targets are capped only by max_seq_length"
        )

    if data_args.experiment_profile == "dolci32k":
        # These roles were loaded from one audited immutable build above. The
        # benchmark/test artifacts are intentionally not opened in this process.
        val_dataset = target_grad_dataset
        eval_dataset = general_val_dataset
    else:
        val_dataset = get_dataset(
            task=training_args.analysis_dataset,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            split="validation",
            k=training_args.n_val,
            seed=training_args.seed,
            max_seq_length_threshold=val_seq_length_threshold
        )

        # Legacy profiles retain their existing held-out task evaluation role.
        eval_dataset = get_dataset(
            task=training_args.analysis_dataset,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            split=data_args.eval_split,
            k=training_args.n_eval,
        )

    # Data collator. The wrapper passes `meta_idx` around the padding collator so
    # selected examples stay traceable to their pool row without ever reaching
    # model(**batch); val/eval batches simply carry no `meta_idx`.
    data_collator = MetaIdxCollator(
        DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    )

    # Initialize layer_wise_subset trainer with data selection capabilities
    trainer = LayerWiseSubsetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        target_grad_dataset=target_grad_dataset,
        target_val_dataset=target_val_dataset,
        general_val_dataset=general_val_dataset,
        train_order_indices=train_order_indices,
        target_collator=target_signal_collator,
        processing_class=tokenizer,
        data_collator=data_collator,
        grad_hook=grad_hook,
    )
    trainer.set_pool_metadata(pool_metadata)

    # Resolve the optimizer before training so the actual official/local Muon
    # backend, exact parameter grouping, and both learning rates are persisted.
    # LayerWiseSubsetTrainer.create_optimizer is idempotent, so Trainer.train()
    # will reuse this instance when it creates the scheduler.
    trainer.create_optimizer()
    runtime_optimizer = trainer._get_unwrapped_optimizer()
    _write_run_metadata(
        training_args,
        model_args,
        data_args,
        optimizer=runtime_optimizer,
        profile_metadata=profile_metadata,
        lifecycle_phase="optimizer_created",
    )
    if wandb_enabled and wandb is not None and wandb.run is not None:
        if hasattr(runtime_optimizer, "get_runtime_metadata"):
            wandb.config.update(
                runtime_optimizer.get_runtime_metadata(),
                allow_val_change=True,
            )

    # Measure the complete execution window shared by smoke and main runs:
    # step-0 evaluation, optimizer-state materialization, training, final
    # evaluation, and final checkpoint serialization.  Resetting here retains
    # the model's current allocation as the baseline while excluding transient
    # tokenizer/data-loading allocations.
    _reset_cuda_peak_memory_stats()

    skip_probe_evaluation = os.environ.get(
        "DRPT_DOLCI_CANDIDATE_PROBE_SKIP_EVAL"
    ) == "1"
    if skip_probe_evaluation and (
        data_args.experiment_profile != "dolci32k"
        or not 0 < int(training_args.max_steps) <= 3
    ):
        raise RuntimeError(
            "DRPT_DOLCI_CANDIDATE_PROBE_SKIP_EVAL is restricted to "
            "dolci32k probes with max_steps<=3"
        )

    # Initial/final generalization evaluation is deliberately excluded from the
    # short candidate-microbatch memory probe. It does not exercise selection,
    # replay, or the optimizer step and can dominate a one-step experiment.
    if skip_probe_evaluation:
        logger.info("*** Skipping initial evaluation for guarded candidate probe ***")
    else:
        logger.info("*** Running initial evaluation before training ***")
        trainer.evaluate()

    # Train
    logger.info("*** Starting training ***")
    train_result = trainer.train()
    if data_args.experiment_profile == "dolci32k":
        expected_steps = (
            int(training_args.max_steps)
            if int(training_args.max_steps) > 0
            else 2000
        )
        if trainer.state.global_step != expected_steps:
            raise RuntimeError(
                "dolci32k candidate traversal ended at an unexpected step: "
                f"actual={trainer.state.global_step}, expected={expected_steps}"
            )
    trainer._save_selection_diagnostics()

    # Final evaluation after training. A formal 32k/16 profile epoch ends at step
    # 2000, which is itself an eval_steps=100 boundary; do not append the same
    # point twice when HF already evaluated that exact final global step.
    last_evaluated_step = (
        trainer.evaluation_results[-1].get("step")
        if trainer.evaluation_results
        else None
    )
    final_step_already_evaluated = (
        data_args.experiment_profile == "dolci32k"
        and last_evaluated_step == trainer.state.global_step
    )
    if skip_probe_evaluation:
        logger.info("*** Skipping final evaluation for guarded candidate probe ***")
    elif not final_step_already_evaluated:
        logger.info("*** Running final evaluation after training ***")
        trainer.evaluate()
    else:
        logger.info(
            "*** Final global step %s was already evaluated; reusing it ***",
            trainer.state.global_step,
        )

    # Save final evaluation results
    trainer.on_train_end()

    # Save model
    trainer.save_model()

    # Rewrite after all successful training work so run_metadata.json contains
    # the actual peak counters rather than the pre-training snapshot.  Keep the
    # already-resolved optimizer implementation and learning rates attached.
    _write_run_metadata(
        training_args,
        model_args,
        data_args,
        optimizer=runtime_optimizer,
        profile_metadata=profile_metadata,
        lifecycle_phase="training_complete",
    )

    # Clean up hooks (only if grad_hook was created)
    if grad_hook is not None and grad_hook.hooks_registered:
        grad_hook.remove_hooks()
        logger.info("Removed gradient hooks")

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")

    if wandb_enabled and wandb is not None and wandb.run is not None:
        wandb.finish()

    return train_result


if __name__ == "__main__":
    main()
