#!/usr/bin/env python3
import glob
import pyarrow.parquet as pq
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "./temp_scienceqa_upload"
files = sorted(glob.glob(f"{folder}/*.parquet"))
if not files:
    print(f"No parquets found in {folder}")
    sys.exit(1)

t = pq.read_table(files[0])
df = t.to_pandas()
print(f"File: {files[0]}  ({len(df)} rows)")
print(f"Schema: {list(df.columns)}\n")
row = df.iloc[0]
for col in df.columns:
    print(f"{'='*60}")
    print(f"[{col}]")
    print(row[col])
print("="*60)
