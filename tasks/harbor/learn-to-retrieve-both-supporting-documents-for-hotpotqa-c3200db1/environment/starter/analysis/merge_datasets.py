"""
Merges HuggingFace datasets from vm2825/nemotron-cc-v21-Parsed-QA4-filtered-1.7B-parts-X-X-X
into ragrawal36/nemotron-cc-v21-Parsed-QA4-filtered-1.7B, one parquet at a time.
"""

import os
import tempfile
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

api = HfApi()

SOURCE_USER = "vm2825"
SOURCE_BASE = "nemotron-cc-v21-Parsed-QA4-filtered-1.7B-parts"
TARGET_REPO = "ragrawal36/nemotron-cc-v21-Parsed-QA4-filtered-1.7B"

# All batches in order, skipping 19-20-21
BATCHES = [
    (0, 1, 2),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
    (16, 17, 18),
    (22, 23, 24),
]


def main():
    # Ensure target repo exists as a dataset
    try:
        api.repo_info(repo_id=TARGET_REPO, repo_type="dataset")
        print(f"Target repo {TARGET_REPO} already exists.")
    except Exception:
        print(f"Creating target repo {TARGET_REPO}...")
        api.create_repo(repo_id=TARGET_REPO, repo_type="dataset", private=False)

    for batch in BATCHES:
        a, b, c = batch
        batch_tag = f"batch-{a}-{b}-{c}"
        source_repo = f"{SOURCE_USER}/{SOURCE_BASE}-{a}-{b}-{c}"
        print(f"\n=== Processing {source_repo} -> {batch_tag} ===")

        # List all parquet files in the source dataset
        try:
            all_files = list(list_repo_files(source_repo, repo_type="dataset"))
        except Exception as e:
            print(f"  ERROR listing files in {source_repo}: {e}")
            continue

        parquet_files = [f for f in all_files if f.endswith(".parquet")]
        print(f"  Found {len(parquet_files)} parquet file(s).")

        for remote_path in parquet_files:
            filename = os.path.basename(remote_path)
            target_path = f"{batch_tag}-{filename}"

            print(f"  Downloading {remote_path}...")
            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = hf_hub_download(
                    repo_id=source_repo,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=tmpdir,
                    local_dir_use_symlinks=False,
                )
                print(f"  Uploading as {target_path}...")
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=target_path,
                    repo_id=TARGET_REPO,
                    repo_type="dataset",
                )
                print(f"  Done. Local file will be deleted on context exit.")
            # tmpdir and local_file are deleted here automatically

    print("\nAll batches processed.")


if __name__ == "__main__":
    main()
