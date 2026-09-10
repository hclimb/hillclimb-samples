import contextlib
from functools import partial
from typing import List, Union

import torch
from datasets import load_dataset

from SFT.data.chat_format import (
    TOKENIZATION_DIAGNOSTIC_COLUMNS,
    encode_assistant_only,
)


@contextlib.contextmanager
def temp_seed(seed):
    torch_state = torch.get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)


def get_train_files_for_dataset(data_dir: str, dataset_name: str) -> List[str]:
    """
    Map a training dataset name to its file path(s).

    Args:
        data_dir: Base directory containing training data
        dataset_name: Name of the training dataset

    Returns:
        List of file paths for the training dataset
    """
    dataset_mapping = {
        # Single dataset files
        "alpaca":   [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "dolly":    [f"{data_dir}/train/dolly/dolly_data.jsonl"],
        "flan_v2":  [f"{data_dir}/train/flan_v2/flan_v2_data.jsonl"],
        "cot":      [f"{data_dir}/train/cot/cot_data.jsonl"],
        "oasst1":   [f"{data_dir}/train/oasst1/oasst1_data.jsonl"],
        "tulu3":    [f"{data_dir}/train/tulu3/tulu3_data.jsonl"],
        "samsum":   [f"{data_dir}/train/samsum/samsum_train_data.jsonl"],
        "nq_open":  [f"{data_dir}/train/nq_open/nq_open_data.jsonl"],
        "triviaqa": [f"{data_dir}/train/triviaqa/triviaqa_data.jsonl"],
        "squad":    [f"{data_dir}/train/squad/squad_data.jsonl"],
        # LESS mixture (flan_v2 + cot + dolly + oasst1)
        "less": [
            f"{data_dir}/train/flan_v2/flan_v2_data.jsonl",
            f"{data_dir}/train/cot/cot_data.jsonl",
            f"{data_dir}/train/dolly/dolly_data.jsonl",
            f"{data_dir}/train/oasst1/oasst1_data.jsonl",
        ],
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unknown training dataset: {dataset_name}. "
                        f"Available: {list(dataset_mapping.keys())}")

    return dataset_mapping[dataset_name]


def _get_default_train_files(data_dir: str, task: str) -> List[str]:
    """
    Get default training files based on task.

    Args:
        data_dir: Base directory containing training data
        task: Evaluation task name

    Returns:
        List of default training file paths for the task
    """
    # LESS mixture for general instruction tuning evaluation tasks
    less_mixture = [
        f"{data_dir}/train/flan_v2/flan_v2_data.jsonl",
        f"{data_dir}/train/cot/cot_data.jsonl",
        f"{data_dir}/train/dolly/dolly_data.jsonl",
        f"{data_dir}/train/oasst1/oasst1_data.jsonl"
    ]

    task_defaults = {
        "samsum":   [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "tydiqa":   less_mixture,
        "triviaqa": [f"{data_dir}/train/nq_open/nq_open_data.jsonl"],
        "nq_open":  [f"{data_dir}/train/triviaqa/triviaqa_data.jsonl"],
    }

    return task_defaults.get(task, less_mixture)


def extract_source_metadata(raw_datasets) -> List[dict]:
    """Per-row ``{id, dataset, domain}`` aligned with the encoded dataset order.

    This is the sidecar the trainer joins against ``meta_idx`` to report which
    source domains a curation method actually selects from a mixed pool. Pools
    Pools built before domain tracking have no ``domain`` column; their
    ``dataset`` name doubles as the domain so the summary is still well-defined.
    """
    columns = set(raw_datasets.column_names)
    ids = raw_datasets["id"] if "id" in columns else None
    source_datasets = (
        raw_datasets["source_dataset"] if "source_dataset" in columns else None
    )
    datasets_col = raw_datasets["dataset"] if "dataset" in columns else None
    domains = raw_datasets["domain"] if "domain" in columns else None

    metadata = []
    for index in range(len(raw_datasets)):
        dataset_name = (
            source_datasets[index]
            if source_datasets is not None
            else datasets_col[index] if datasets_col is not None else "unknown"
        )
        metadata.append({
            "id": ids[index] if ids is not None else f"row_{index}",
            "dataset": dataset_name,
            "source_dataset": dataset_name,
            "domain": domains[index] if domains is not None else dataset_name,
        })
    return metadata


def get_training_dataset(data_dir: str, task: str, tokenizer, max_seq_length,
                         sample_percentage=1.0, seed=0, train_files: List[str] = None,
                         train_dataset_names: List[str] = None,
                         return_source_metadata: bool = False,
                         return_tokenization_stats: bool = False):
    """
    Get training dataset with a specified seed.

    Args:
        data_dir: Base directory containing training data
        task: Evaluation task name (mmlu, samsum, tydiqa, bbh, gsm8k, math500)
        tokenizer: Tokenizer to use for encoding
        max_seq_length: Maximum sequence length
        sample_percentage: Percentage of data to sample
        seed: Random seed for sampling
        train_files: Optional explicit list of training files (overrides all other selection)
        train_dataset_names: Optional list of training dataset names (e.g., ['wizardlm', 'alpaca'])
        return_source_metadata: Also return the per-row source/domain sidecar and
            attach a `meta_idx` column so selected examples can be traced back
            to their source.

    Returns:
        Encoded training dataset, or `(dataset, source_metadata)` when
        `return_source_metadata` is True.
    """
    # Priority: train_files > train_dataset_names > task-based default
    if train_files is None:
        if train_dataset_names is not None:
            # Use explicitly specified training datasets
            train_files = []
            for name in train_dataset_names:
                train_files.extend(get_train_files_for_dataset(data_dir, name))
        else:
            # Fall back to task-based defaults
            train_files = _get_default_train_files(data_dir, task)

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    encoded = encode_data(
        raw_datasets, tokenizer, max_seq_length,
        attach_meta_idx=return_source_metadata,
        return_tokenization_stats=return_tokenization_stats,
    )
    if return_tokenization_stats:
        lm_datasets, tokenization_stats = encoded
    else:
        lm_datasets = encoded
        tokenization_stats = None
    if return_source_metadata and return_tokenization_stats:
        return (
            lm_datasets,
            extract_source_metadata(raw_datasets),
            tokenization_stats,
        )
    if return_source_metadata:
        return lm_datasets, extract_source_metadata(raw_datasets)
    if return_tokenization_stats:
        return lm_datasets, tokenization_stats
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = torch.randperm(len(processed_datasets))[:sample_size].tolist()

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10,
                overwrite_cache=False, attach_meta_idx=False,
                return_tokenization_stats=False):
    """Encode messages-format examples with the tokenizer's native chat template.

    With `attach_meta_idx=True` the encoded dataset carries a `meta_idx` column
    (the row's position in `raw_datasets`) and drops every original column, so a
    batch reaching the collator holds only tensors plus that index. Callers that
    use this must run the Trainer with `remove_unused_columns=False`, otherwise
    `meta_idx` is stripped before the collator ever sees it.
    """
    if "input_ids" in raw_datasets.features:
        if return_tokenization_stats:
            return raw_datasets, {
                "examples": len(raw_datasets),
                "truncated": None,
                "assistant_end_retained": None,
                "right_truncation_zero_supervision": None,
                "supervision_preserving_fallback_applied": None,
                "zero_supervision_after_truncation": None,
            }
        return raw_datasets
    if "messages" not in raw_datasets.column_names:
        raise ValueError(
            "Training data must have a 'messages' column. Got columns: "
            f"{raw_datasets.column_names}"
        )
    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    if attach_meta_idx:
        def encode_with_index(example, idx):
            encoded = encode_function(example)
            encoded["meta_idx"] = idx
            return encoded

        lm_datasets = raw_datasets.map(
            encode_with_index,
            with_indices=True,
            batched=False,
            num_proc=processing_num_workers,
            load_from_cache_file=not overwrite_cache,
            remove_columns=list(raw_datasets.column_names),
            desc="Tokenizing and reformatting instruction data",
        )
        return _finalize_tokenization_diagnostics(
            lm_datasets, return_tokenization_stats=return_tokenization_stats
        )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    # Keep examples in their Arrow/Python representation. The Trainer's
    # DataCollatorForSeq2Seq converts each padded batch to tensors. Setting the
    # entire Dataset to the Torch formatter is redundant for text data and also
    # forces datasets to import optional torchvision video APIs on every fetch.
    return _finalize_tokenization_diagnostics(
        lm_datasets, return_tokenization_stats=return_tokenization_stats
    )


def _finalize_tokenization_diagnostics(dataset, *, return_tokenization_stats=False):
    columns = set(dataset.column_names)
    available = [name for name in TOKENIZATION_DIAGNOSTIC_COLUMNS if name in columns]
    stats = {
        "examples": len(dataset),
        "truncated": None,
        "assistant_end_retained": None,
        "right_truncation_zero_supervision": None,
        "supervision_preserving_fallback_applied": None,
        "zero_supervision_after_truncation": None,
        "max_untruncated_length": None,
    }
    if available:
        lengths = dataset["_tokenization_untruncated_length"]
        truncated = dataset["_tokenization_was_truncated"]
        retained = dataset["_tokenization_assistant_end_retained"]
        right_zero = dataset[
            "_tokenization_zero_supervised_after_right_truncation"
        ]
        fallback = dataset[
            "_tokenization_supervision_preserving_fallback_applied"
        ]
        final_zero = dataset[
            "_tokenization_zero_supervised_after_truncation"
        ]
        stats.update({
            "truncated": int(sum(bool(value) for value in truncated)),
            "assistant_end_retained": int(sum(bool(value) for value in retained)),
            "right_truncation_zero_supervision": int(
                sum(bool(value) for value in right_zero)
            ),
            "supervision_preserving_fallback_applied": int(
                sum(bool(value) for value in fallback)
            ),
            "zero_supervision_after_truncation": int(
                sum(bool(value) for value in final_zero)
            ),
            "max_untruncated_length": int(max(lengths)) if lengths else 0,
        })
        dataset = dataset.remove_columns(available)
    if return_tokenization_stats:
        return dataset, stats
    return dataset


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """Render a chat record and supervise every assistant/tool-call span.

    Qwen3 uses native token delimiters rather than the historical prefix
    comparison, fixing intermediate assistant turns and Dolci tool-use rows.
    """
    return encode_assistant_only(example, tokenizer, max_seq_length)
