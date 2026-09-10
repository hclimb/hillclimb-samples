"""Copy the two mihir-1999 hard-neg datasets to vm2825, appending `-think`.

Sources:
  mihir-1999/nemotron-hard-neg             -> vm2825/science-qa-hard-neg-think
  mihir-1999/nemotron-combined-hard-neg    -> vm2825/giverseqa-hard-neg-think
                                              (also duplicate first train parquet as validation)

Both sources share the schema:
  query, pos_doc, answer, neg_docs, neg_scores, think, generated_answer

Streaming copy: iterates parquet files one at a time via HfFileSystem, downloads
each to a tmp file, re-uploads with the same path, then deletes the local tmp.
This avoids filling the disk with a full snapshot.
"""
import argparse
import os
import shutil
import sys
import tempfile

from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download, login

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN missing — set it in .env or the environment")
login(token=HF_TOKEN)

PAIRS = [
    # (source, target, add_val_from_train)
    ("mihir-1999/nemotron-hard-neg",          "vm2825/science-qa-hard-neg-think",   False),
    ("mihir-1999/nemotron-combined-hard-neg", "vm2825/diverseqa-hard-neg-think",    True),
]


def list_parquets(fs: HfFileSystem, repo_id: str) -> list[str]:
    """Find parquets under the dataset, relative to the repo root."""
    prefix = f"datasets/{repo_id}/"
    # common layout: data/{split}-*.parquet
    candidates: list[str] = []
    for pattern in [
        f"{prefix}data/**/*.parquet",
        f"{prefix}**/*.parquet",
    ]:
        candidates = sorted(fs.glob(pattern))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"no parquet files under {repo_id}")
    # strip the 'datasets/<repo>/' prefix so we keep the in-repo path
    return [c[len(prefix):] for c in candidates]


def copy_one(src: str, dst: str, add_val_from_train: bool, fs: HfFileSystem, api: HfApi) -> None:
    print(f"\n{'=' * 80}\nCOPY: {src} -> {dst}\n{'=' * 80}")
    api.create_repo(repo_id=dst, repo_type="dataset", exist_ok=True, private=False)

    paths = list_parquets(fs, src)
    print(f"  {len(paths)} parquet files under {src}")

    # also copy any README / loading script so the dataset card isn't lost — optional
    # Focus on parquet files for correctness.

    first_train_local = None  # capture first train parquet for val duplication
    for i, rel in enumerate(paths, 1):
        print(f"  [{i}/{len(paths)}] {rel}")
        local = hf_hub_download(repo_id=src, repo_type="dataset", filename=rel, token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=rel,
            repo_id=dst,
            repo_type="dataset",
        )
        if add_val_from_train and first_train_local is None and "train-" in os.path.basename(rel):
            # keep this parquet around for duplicating into validation
            tmp_dir = tempfile.mkdtemp()
            first_train_local = os.path.join(tmp_dir, os.path.basename(rel))
            shutil.copyfile(local, first_train_local)
            first_train_rel = rel  # remember repo path for rename
        # Delete the cached snapshot file to free space.
        try:
            os.remove(local)
        except OSError:
            pass
        # also drop the containing blob symlink target if it's big and orphaned — skip for simplicity

    if add_val_from_train:
        if first_train_local is None:
            print("  (no train-*.parquet found, skipping validation duplication)")
        else:
            val_rel = first_train_rel.replace("train-", "validation-", 1)
            print(f"  duplicating {first_train_rel} -> {val_rel}")
            api.upload_file(
                path_or_fileobj=first_train_local,
                path_in_repo=val_rel,
                repo_id=dst,
                repo_type="dataset",
            )
            try:
                os.remove(first_train_local)
                os.rmdir(os.path.dirname(first_train_local))
            except OSError:
                pass

    print(f"  done -> {dst}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   help="substring filter on source name (e.g. 'combined')")
    args = p.parse_args()
    fs = HfFileSystem(token=HF_TOKEN)
    api = HfApi()
    for src, dst, add_val in PAIRS:
        if args.only and args.only not in src and args.only not in dst:
            continue
        try:
            copy_one(src, dst, add_val, fs, api)
        except Exception as e:
            import traceback
            print(f"FAILED {src} -> {dst}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
