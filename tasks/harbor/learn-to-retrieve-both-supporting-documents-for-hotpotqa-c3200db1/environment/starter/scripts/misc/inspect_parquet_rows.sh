#!/bin/bash
# Runs ON a box. Pulls ONE parquet shard from each named dataset repo and reports
# bytes/row + schema, to compare raw-text vs pre-tokenized storage density.
#
# Why one shard and not the repo: repos differ in row count, so total GB is not comparable —
# bytes/row is. Reading the parquet footer gives exact rows without loading the data.
#
#   REPOS="a/b c/d" bash scripts/misc/inspect_parquet_rows.sh
set -uo pipefail
cd "${REPO_DIR:-$HOME/memory-layers}"
set -a; . "$HOME/.env" 2>/dev/null || true; set +a
. .venv/bin/activate 2>/dev/null || true

REPOS="${REPOS:-ragrawal36/qa-hard-neg-think-membed-full ragrawal36/qa-hard-neg-think-membed-tokenized vm2825/science-qa-hard-neg-think}"

REPOS="$REPOS" python - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

tok = os.environ.get("HF_TOKEN")
api = HfApi()
for repo in os.environ["REPOS"].split():
    print(f"\n=========== {repo} ===========", flush=True)
    try:
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", token=tok)
                       if f.endswith(".parquet"))
        if not files:
            print("  no parquet files"); continue
        f0 = files[0]
        p = hf_hub_download(repo, f0, repo_type="dataset", token=tok,
                            local_dir=f"/tmp/pqinspect/{repo.replace('/','__')}")
        size = os.path.getsize(p)
        md = pq.ParquetFile(p).metadata
        rows = md.num_rows
        print(f"  shard          : {f0}  (1 of {len(files)})")
        print(f"  file size      : {size/1e6:,.1f} MB")
        print(f"  rows           : {rows:,}")
        print(f"  BYTES/ROW      : {size/rows:,.0f}")
        sch = pq.ParquetFile(p).schema_arrow
        print(f"  columns        : {len(sch)}")
        for name, typ in zip(sch.names, sch.types):
            print(f"      {name:28s} {str(typ)[:44]}")
        # Per-column compressed bytes: shows WHERE the bytes actually are.
        tot = {}
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for c in range(g.num_columns):
                col = g.column(c)
                tot[col.path_in_schema] = tot.get(col.path_in_schema, 0) + col.total_compressed_size
        print("  compressed bytes/row by column:")
        for k, v in sorted(tot.items(), key=lambda x: -x[1]):
            print(f"      {k:28s} {v/rows:10,.0f}  ({100*v/size:5.1f}%)")
    except Exception as e:
        print(f"  ERROR: {str(e)[:160]}")
PY
