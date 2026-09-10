#!/usr/bin/env python3
"""Target-only sanity check: SFT directly on D* (the 64 target-gradient rows).

Why this exists
---------------
The dolci32k curation methods score candidates by how well their gradients align
with a target gradient built from ``targets/<task>/grad.jsonl`` (64 rows). On the
math and code settings those methods lower target validation loss yet lose on
MATH500 / MBPP+, and they *under*-select the target's own domain (lift < 1). Two
explanations fit that: the selection machinery is at fault, or D* itself is a
poor gradient proxy for the downstream objective.

Training on D* alone separates them. No curation, no candidate pool -- just SFT
on the same 64 rows the target gradient is computed from:

  target val loss down AND downstream up   -> D* is a fine proxy; suspect selection
  target val loss down BUT downstream down -> D* is the problem, and no selection
                                              method built on it can do better

Formatting is deliberately identical to the campaign: same pinned model and
revision, same ``encode_assistant_only`` chat rendering and assistant-only label
masking, same max_seq_length / optimizer / schedule shape. Only the training set
and the step count differ, so a downstream drop cannot be blamed on a different
data format.

Example
-------
  python -m SFT.train.train_target_only \\
      --setting reason_math --epochs 15 \\
      --output_dir SFT/runs/target_only/reason_math
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from SFT.data.chat_format import encode_assistant_only
from SFT.data.dolci32k.profile import MODEL_PROFILES, SETTINGS

logger = logging.getLogger(__name__)

# Which target directory each setting scores against, mirroring
# SFT/data/dolci32k/profile.py::SETTINGS[...]["target"].
SETTING_TARGET = {name: cfg["target"] for name, cfg in SETTINGS.items()}


def artifact_root(data_dir: Path, build_id: str) -> Path:
    return data_dir / "dolci32k_artifacts" / "builds" / build_id


def load_target_rows(data_dir: Path, build_id: str, target: str, split: str):
    path = artifact_root(data_dir, build_id) / "targets" / target / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"target split not found: {path}")
    rows = [json.loads(line) for line in path.open()]
    logger.info("loaded %d rows from %s", len(rows), path)
    return rows


def build_dataset(rows, tokenizer, max_seq_length: int) -> Dataset:
    """Encode with the campaign's own renderer so the format is identical.

    The renderer emits the chat template verbatim -- including the empty
    ``<think>\\n\\n</think>`` pair Qwen3 inserts before the assistant turn -- and
    masks everything but assistant tokens to -100. It also attaches a pile of
    ``_tokenization_*`` audit columns that the collator cannot batch, so keep
    only the three tensors the model consumes.
    """
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    encoded = ds.map(
        lambda ex: encode_assistant_only(ex, tokenizer, max_seq_length),
        remove_columns=["messages"],
        desc="encoding target rows",
    )
    keep = {"input_ids", "labels", "attention_mask"}
    return encoded.remove_columns([c for c in encoded.column_names if c not in keep])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", required=True, choices=sorted(SETTING_TARGET))
    p.add_argument("--model_profile", default="qwen3_1_7b")
    p.add_argument("--data_dir", default=os.environ.get("DRPT_DATA_DIR", "SFT/data"))
    p.add_argument("--artifact_build_id", default=os.environ.get("DRPT_ARTIFACT_BUILD_ID"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=15.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=16,
                   help="effective batch size (kept identical to the campaign)")
    p.add_argument("--micro_batch_size", type=int, default=2,
                   help=(
                       "rows per forward pass. The collator pads to the longest row "
                       "in the microbatch, and Qwen3's 151936-token vocab makes the "
                       "fp32 logits ~2.3GB per 4096-token row, so 16 at once OOMs on "
                       "a 46GB A40 for the longer targets. Gradient accumulation "
                       "restores the effective batch, leaving the update unchanged."
                   ))
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.artifact_build_id:
        raise SystemExit("--artifact_build_id (or DRPT_ARTIFACT_BUILD_ID) is required")
    set_seed(args.seed)

    profile = MODEL_PROFILES[args.model_profile]
    model_name = profile["model_name_or_path"]
    revision = profile.get("model_revision")
    target = SETTING_TARGET[args.setting]
    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        profile.get("tokenizer_name", model_name),
        revision=profile.get("tokenizer_revision", revision),
        use_fast=True,
    )
    train_rows = load_target_rows(data_dir, args.artifact_build_id, target, "grad")
    train_ds = build_dataset(train_rows, tokenizer, args.max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    micro = max(1, min(args.micro_batch_size, args.batch_size))
    if args.batch_size % micro:
        raise SystemExit(
            f"--batch_size {args.batch_size} must be divisible by --micro_batch_size {micro}"
        )
    accum = args.batch_size // micro
    logger.info("effective batch %d = micro %d x accum %d", args.batch_size, micro, accum)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=True,
        tf32=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest",
        ),
    )
    result = trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "target_only_metadata.json").write_text(json.dumps({
        "setting": args.setting,
        "target": target,
        "n_train_rows": len(train_rows),
        "model_name_or_path": model_name,
        "model_revision": revision,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "artifact_build_id": args.artifact_build_id,
        "train_runtime_sec": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "note": "SFT on D* (target grad split) only -- no curation, no candidate pool",
    }, indent=2) + "\n")
    Path(args.output_dir, "_SUCCESS").touch()
    logger.info("done: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
