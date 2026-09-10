#!/usr/bin/env python3
"""Teacher-forced loss on the downstream test items themselves.

The campaign already logs two losses -- ``target_val_loss`` on 128 held-out rows
of the target split, and ``general_val_loss`` on 512 rows of the candidate pool --
but neither is measured on the benchmark that decides the headline number. That
leaves a gap in the story: curation lowers target val loss on the math settings
and still loses MATH500 accuracy, and from those two numbers alone it is
impossible to tell whether the model got worse at the task or only worse at
emitting a parseable answer.

Loss on the benchmark's own reference solutions closes it:

  test loss down, accuracy down  -> the model fits the task, generation is at fault
  test loss up,   accuracy down  -> a genuine capability regression

Only settings whose benchmark ships reference solutions can be measured:
MATH500 (``solution``) and MBPP+ (``canonical_solution``). IFEval / IFBench are
rule-verified with no reference text, so they are out of scope by construction.

Scoring is forward-only -- no sampling -- so this is far cheaper than the
generation evals, and it reuses the campaign's own ``encode_assistant_only``
rendering so the loss is computed over exactly the tokens training supervised.

  python -m SFT.eval.test_set_task_loss --out SFT/eval/reports/temp_inst_if_adamw
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from SFT.data.chat_format import encode_assistant_only

logger = logging.getLogger(__name__)

ROOT = Path("/home/nakyungl/drpt-next/Dr.Post-Training-Next/drpt_opus")
CAMPAIGN = ROOT / "SFT/runs/campaigns/dolci32k-qwen3_1_7b-s42"

METHODS = ("FullTraining", "LayerwiseRaw", "LayerwiseSoft", "LayerwiseSoftP", "LayerwiseOptA")
# setting -> benchmark with reference solutions
SETTING_BENCH = {
    "reason_math": "math500",
    "mixed_math": "math500",
    "reason_code": "mbpp_plus",
}
MBPP_PROMPT = (
    "Complete this MBPP+ task. Return one complete, self-contained Python "
    "solution including the required function signature. Do not include prose "
    "outside the code.\n\n{prompt}"
)
MATH_PROMPT = (
    "Solve the following mathematics problem. Show your reasoning, then put "
    "only the final answer inside \\boxed{{}}.\n\n{problem}"
)


def math500_pairs():
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        (MATH_PROMPT.format(problem=r["problem"]), r["solution"])
        for r in ds
        if r.get("solution")
    ]


def mbpp_plus_pairs():
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    out = []
    for task in problems.values():
        ref = (task.get("prompt") or "").rstrip() + "\n" + (task.get("canonical_solution") or "")
        if task.get("canonical_solution"):
            out.append((MBPP_PROMPT.format(prompt=task["prompt"]), ref))
    return out


LOADERS = {"math500": math500_pairs, "mbpp_plus": mbpp_plus_pairs}


@torch.no_grad()
def mean_token_loss(model, tokenizer, pairs, max_seq_length: int, device) -> dict:
    """Token-weighted mean NLL over assistant tokens only."""
    total_nll = 0.0
    total_tok = 0
    skipped = 0
    for user, assistant in pairs:
        enc = encode_assistant_only(
            {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]},
            tokenizer,
            max_seq_length,
        )
        ids = torch.tensor(enc["input_ids"], device=device).unsqueeze(0)
        labels = torch.tensor(enc["labels"], device=device).unsqueeze(0)
        if int((labels != -100).sum()) == 0:
            skipped += 1
            continue
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        n = int((labels[:, 1:] != -100).sum())
        total_nll += float(out.loss) * n
        total_tok += n
    loss = total_nll / max(total_tok, 1)
    return {
        "test_task_loss": loss,
        "test_task_perplexity": float(torch.exp(torch.tensor(loss))),
        "n_items": len(pairs) - skipped,
        "n_supervised_tokens": total_tok,
        "n_skipped": skipped,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", nargs="*", default=sorted(SETTING_BENCH))
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--extra_model", action="append", default=[],
                   help="label=/path/to/model, e.g. a target-only checkpoint")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0, help="0 = all items")
    p.add_argument("--out", default=str(ROOT / "SFT/eval/reports/temp_inst_if_adamw"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_set_task_loss.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    cache = {}
    jobs = []
    for setting in args.settings:
        bench = SETTING_BENCH[setting]
        for method in args.methods:
            hits = glob.glob(str(CAMPAIGN / f"{setting}-{method}-adamw-*"))
            if hits:
                jobs.append((setting, bench, method, hits[0]))
    for spec in args.extra_model:
        label, _, path = spec.partition("=")
        setting = label.split(":")[0]
        if setting in SETTING_BENCH:
            jobs.append((setting, SETTING_BENCH[setting], label, path))

    for setting, bench, method, path in jobs:
        key = f"{setting}|{method}"
        if key in results:
            logger.info("skip %s (already measured)", key)
            continue
        if bench not in cache:
            cache[bench] = LOADERS[bench]()
            logger.info("%s: %d reference pairs", bench, len(cache[bench]))
        pairs = cache[bench][: args.limit] if args.limit else cache[bench]
        logger.info("scoring %s on %s (%d items)", key, bench, len(pairs))
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).to(device).eval()
        stats = mean_token_loss(model, tok, pairs, args.max_seq_length, device)
        stats.update({"setting": setting, "method": method, "benchmark": bench,
                      "model_path": path})
        results[key] = stats
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        logger.info("  %s -> loss %.4f (ppl %.2f)", key, stats["test_task_loss"],
                    stats["test_task_perplexity"])
        del model
        torch.cuda.empty_cache()

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
