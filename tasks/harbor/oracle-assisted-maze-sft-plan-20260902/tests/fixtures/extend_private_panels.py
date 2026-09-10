#!/usr/bin/env python3
"""Append three fixed, previously unused private panels; keep the original panels."""

import argparse
import gzip
import json
from pathlib import Path
import ssl
import sys
import urllib.request

import pack_real_data as packer

HERE = Path(__file__).resolve().parent


def base_rows(name, plan):
    raw = gzip.decompress((HERE / f"{name}.jsonl.gz").read_bytes())
    lines = raw.splitlines(keepends=True)[:plan["base_counts"][name]]
    raw = b"".join(lines)
    if packer.sha256(raw) != plan["base_raw_sha256"][name]:
        raise RuntimeError(f"Original private panel content changed: {name}")
    return raw, [json.loads(line) for line in lines]


def validate_selection(selected, selection, plan):
    if (selection["source_sha256"] != packer.SOURCE_SHA256
            or selection["source_bytes"] != packer.SOURCE_BYTES):
        raise RuntimeError("Upstream source does not match the pinned content")
    ids = [row["prompt_id"] for _, row in selected]
    grids = [packer.grid_digest(row) for _, row in selected]
    if (len(ids) != 864 or len(set(ids)) != 864 or len(set(grids)) != 864
            or set(ids) & set(plan["prompt_ids"]) or set(grids) & set(plan["grid_sha256"])):
        raise RuntimeError("Private extension has missing, duplicate, or previously used mazes")


def pack_extension(selected, selection, output):
    import torch
    from utils.maze import _group
    from utils.model import INITIAL_SHA256, initialized_model, rollout

    plan = json.loads((HERE / "private_extension_plan.json").read_text())
    validate_selection(selected, selection, plan)
    manifest = json.loads((HERE / "manifest.json").read_text())
    base = {name: base_rows(name, plan) for name in plan["base_counts"]}
    labels = json.loads(gzip.decompress((HERE / "frozen_failures.json.gz").read_bytes()))
    base_ids = set(manifest["partitions"]["public_train"]["prompt_ids"])
    base_ids.update(row["prompt_id"] for row in base["private_train"][1])
    labels["actions_by_prompt_id"] = {key: value for key, value in labels["actions_by_prompt_id"].items()
                                      if int(key) in base_ids}
    raw_labels = (json.dumps(labels, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if (packer.sha256(raw_labels) != plan["base_failures_raw_sha256"]
            or labels["checkpoint_sha256"] != INITIAL_SHA256):
        raise RuntimeError("Original frozen failure labels changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = initialized_model().to(torch.device("cpu")).eval()
    for name, additions in (("private_train", selected[:576]), ("private_eval", selected[576:])):
        raw_base, rows_base = base[name]
        raw_added = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for _, row in additions)
        rows = rows_base + [row for _, row in additions]
        metadata = packer.write_payload(output, f"{name}.jsonl.gz", raw_base + raw_added, len(rows))
        metadata.update(prompt_ids=[row["prompt_id"] for row in rows],
                        grid_sha256=[packer.grid_digest(row) for row in rows],
                        source_line_numbers=(manifest["partitions"][name]["source_line_numbers"][:len(rows_base)]
                                             + [line for line, _ in additions]))
        manifest["partitions"][name] = metadata
        for index, (_, row) in enumerate(additions):
            group = _group(row)
            if name == "private_train":
                labels["actions_by_prompt_id"][str(row["prompt_id"])] = rollout(model, group, torch.device("cpu"))
            if (index + 1) % 96 == 0:
                print(f"Validated new {name}: {index + 1}/{len(additions)}", flush=True)
    raw = (json.dumps(labels, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest["frozen_failures"] = {
        **manifest["frozen_failures"],
        **packer.write_payload(output, "frozen_failures.json.gz", raw, len(labels["actions_by_prompt_id"])),
    }
    manifest["dataset_version"] = "private-six-panels-v3"
    manifest["panels"]["private"] = 6
    manifest["selection"]["private_extension"] = {
        "algorithm": "Uniform full-source reservoir sample excluding every previously used ID and grid; "
                     "one seeded shuffle; append three panels after the original private panels",
        "seed": plan["seed"], **selection, "added_train": 576, "added_eval": 288,
        "plan_sha256": packer.sha256((HERE / "private_extension_plan.json").read_bytes()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist; validate staged data before installing it")
    plan = json.loads((HERE / "private_extension_plan.json").read_text())
    context = ssl.create_default_context(
        cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None)
    url = f"https://huggingface.co/datasets/{packer.REPOSITORY}/resolve/{packer.REVISION}/{packer.SOURCE}?download=true"
    print("Streaming the pinned 4.02 GB source once; original public and private panels stay fixed.", flush=True)
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        selected, selection = packer.sample_rows(response, 864, plan, seed=plan["seed"], progress=True)
    validate_selection(selected, selection, plan)
    sys.path.insert(0, str(HERE.parent))
    pack_extension(selected, selection, args.output)
    print(f"Private six-panel fixtures staged at {args.output}", flush=True)


if __name__ == "__main__":
    main()
