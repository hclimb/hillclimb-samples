#!/usr/bin/env python3
"""Construction-only uniform reservoir sample of the pinned grouped maze JSONL."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import ssl
import sys
import urllib.request

REPOSITORY = "max-rl/maze_17x17_diverse_1.3m"
REVISION = "9b9ed56991cb045ba4227d9120dad337085db439"
SOURCE = "main_1.3M.jsonl"
SOURCE_SHA256 = "c1d055238c80a042f89b5f661e6779773f16cadb084c29d39f0c3eddb4fd59bc"
SOURCE_BYTES = 4020375999
# Fixed, evaluator-only seed. Do not publish it in the agent-visible bundle.
SELECTION_SEED = 59920858472998682440866748306434244058272847769274311964042659556379968795092
COUNTS = {"public_train": 1152, "public_eval": 576,
          "private_train": 576, "private_eval": 288}
HERE = Path(__file__).resolve().parent


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def grid_digest(row):
    return sha256(bytes(cell for line in row["grid"] for cell in line))


def valid_row(row):
    return (set(("prompt_id", "grid", "L_star", "ub", "n_samples", "samples")) <= row.keys()
            and len(row["grid"]) == 17 and all(len(line) == 17 for line in row["grid"])
            and all(cell in (0, 1) for line in row["grid"] for cell in line)
            and row["ub"] == 60 and row["n_samples"] == len(row["samples"]))


def sample_rows(lines, needed, exclusions, *, seed=SELECTION_SEED, progress=False):
    """Algorithm R over eligible unique grids; retain the first occurrence of each grid."""
    rng = random.Random(seed)
    old_ids, old_grids = set(exclusions["prompt_ids"]), set(exclusions["grid_sha256"])
    seen_ids, seen_grids, selected = set(), set(), []
    source_hash = hashlib.sha256()
    source_bytes = eligible = rejected = duplicates = line_number = 0
    for line_number, line in enumerate(lines, 1):
        source_hash.update(line)
        source_bytes += len(line)
        row = json.loads(line)
        if not valid_row(row):
            rejected += 1
            continue
        grid = grid_digest(row)
        prompt_id = row["prompt_id"]
        if prompt_id in old_ids or grid in old_grids:
            rejected += 1
            continue
        if prompt_id in seen_ids or grid in seen_grids:
            duplicates += 1
            continue
        seen_ids.add(prompt_id)
        seen_grids.add(grid)
        eligible += 1
        if len(selected) < needed:
            selected.append((line_number, row))
        else:
            index = rng.randrange(eligible)
            if index < needed:
                selected[index] = (line_number, row)
        if progress and line_number % 50000 == 0:
            print(f"Scanned {line_number:,} rows ({source_bytes / 1e9:.2f} GB); "
                  f"retained {len(selected):,}", flush=True)
    if len(selected) != needed:
        raise RuntimeError(f"source has only {eligible} eligible unique mazes; need {needed}")
    rng.shuffle(selected)
    return selected, {"source_rows_scanned": line_number, "source_bytes": source_bytes,
                      "source_sha256": source_hash.hexdigest(), "eligible_unique_mazes": eligible,
                      "excluded_or_invalid_rows": rejected, "duplicate_rows": duplicates}


def write_payload(output, filename, raw, rows):
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    (output / filename).write_bytes(compressed)
    return {"file": filename, "rows": rows, "compressed_bytes": len(compressed),
            "uncompressed_bytes": len(raw), "sha256": sha256(compressed),
            "uncompressed_sha256": sha256(raw)}


def pack(selected, selection, output):
    import torch
    from utils.maze import _group
    from utils.model import INITIAL_SHA256, initialized_model, rollout

    torch.set_num_threads(1)
    original = json.loads((HERE / "manifest.json").read_text())
    manifest = {
        "dataset_version": "expanded-panels-v2",
        "upstream": {"repository": REPOSITORY, "revision": REVISION,
                     "license": "Apache-2.0", "source_file": SOURCE,
                     "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES},
        "selection": {"algorithm": "Algorithm R over all eligible unique maze grids, followed by "
                      "one seeded shuffle and partition slicing; old maze IDs and grids excluded",
                      "seed": SELECTION_SEED, **selection,
                      "exclusions_sha256": sha256((HERE / "legacy_exclusions.json").read_bytes())},
        "panels": {"public": 6, "private": 3, "train_count": 192, "eval_count": 96},
        "partitions": {"initialization": original["partitions"]["initialization"]},
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("initialization.jsonl.gz", "maze_policy_init.pt", "legacy_exclusions.json"):
        shutil.copyfile(HERE / filename, output / filename)
    actions, offset = {}, 0
    model = initialized_model().to(torch.device("cpu")).eval()
    for name, count in COUNTS.items():
        records = selected[offset:offset + count]
        offset += count
        raw = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for _, row in records)
        metadata = write_payload(output, f"{name}.jsonl.gz", raw, count)
        metadata.update(source_line_numbers=[line for line, _ in records],
                        prompt_ids=[row["prompt_id"] for _, row in records],
                        grid_sha256=[grid_digest(row) for _, row in records])
        manifest["partitions"][name] = metadata
        for index, (_, row) in enumerate(records):
            group = _group(row)  # Validate every selected grid, BFS length, and supplied path.
            if name.endswith("_train"):
                actions[str(row["prompt_id"])] = rollout(model, group, torch.device("cpu"))
            if (index + 1) % 64 == 0 or index + 1 == count:
                print(f"Validated {name}: {index + 1}/{count}", flush=True)
    raw = (json.dumps({
        "checkpoint_sha256": INITIAL_SHA256,
        "generation": "evaluator-owned greedy rollout with wall and revisit masking and UB=60",
        "actions_by_prompt_id": actions,
    }, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest["frozen_failures"] = {
        **write_payload(output, "frozen_failures.json.gz", raw, len(actions)),
        "checkpoint_sha256": INITIAL_SHA256,
        "generation": "evaluator-owned greedy rollout with wall and revisit masking and UB=60",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Publish only the public fixtures and public failure records to the agent bundle.
    public = output / "public"
    public.mkdir()
    for filename in ("public_train.jsonl.gz", "public_eval.jsonl.gz", "maze_policy_init.pt"):
        shutil.copyfile(output / filename, public / filename)
    public_ids = set(manifest["partitions"]["public_train"]["prompt_ids"])
    public_actions = {key: value for key, value in actions.items() if int(key) in public_ids}
    raw = (json.dumps({"checkpoint_sha256": INITIAL_SHA256,
                       "generation": manifest["frozen_failures"]["generation"],
                       "actions_by_prompt_id": public_actions},
                      sort_keys=True, separators=(",", ":")) + "\n").encode()
    public_manifest = {
        "dataset_version": manifest["dataset_version"], "upstream": manifest["upstream"],
        "selection": {"algorithm": "Public slice of a full-source unique-grid reservoir sample; "
                      "legacy grids excluded and evaluator-only selection seed omitted"},
        "panels": {"public": 6, "train_count": 192, "eval_count": 96},
        "partitions": {name: value for name, value in manifest["partitions"].items()
                       if name.startswith("public_")},
        "frozen_failures": {**write_payload(public, "frozen_failures.json.gz", raw, len(public_actions)),
                            "checkpoint_sha256": INITIAL_SHA256,
                            "generation": manifest["frozen_failures"]["generation"]},
    }
    (public / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist; validate staged data before installing it")
    exclusions = json.loads((HERE / "legacy_exclusions.json").read_text())
    context = ssl.create_default_context(
        cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None)
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{SOURCE}?download=true"
    print("Streaming the 4.02 GB pinned source once; the full download is not saved.", flush=True)
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        selected, selection = sample_rows(response, sum(COUNTS.values()), exclusions, progress=True)
    if selection["source_sha256"] != SOURCE_SHA256 or selection["source_bytes"] != SOURCE_BYTES:
        raise RuntimeError("upstream source does not match its pinned LFS SHA-256 and byte count")
    sys.path.insert(0, str(HERE.parent))
    pack(selected, selection, args.output)
    print(f"Validated staged fixtures: {args.output}", flush=True)


if __name__ == "__main__":
    main()
