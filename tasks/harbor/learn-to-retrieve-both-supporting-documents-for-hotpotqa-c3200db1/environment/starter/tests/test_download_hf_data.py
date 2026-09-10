"""Standalone test for data/download_hf_data.py's planning math.

The planning layer is deliberately pure (no network/disk) so the levers can be pinned exactly:
row targets vs the interleave, shard sizing under row-count variance, and selection determinism.

Run:  uv run python tests/test_download_hf_data.py
"""
import sys

from data.download_hf_data import (
    SourcePlan,
    build_plan,
    select_shards,
    shards_for_rows,
    target_rows_for,
)

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def check_true(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# ── target_rows_for: must mirror qa.py's interleave ──────────────────────────────────────
# No weights => uniform round-robin => equal share. This is the 1.6M/4 = 400k case.
check("uniform 4 sources", target_rows_for(1_600_000, 1.0, 4.0), 400_000)
# Weighted => proportional, matching qa.py's probabilities=[w/sum(w)] branch.
check("weighted 3:1 of 4", target_rows_for(1_600_000, 3.0, 4.0), 1_200_000)
check("weighted 1:1:2", target_rows_for(1_000_000, 2.0, 4.0), 500_000)
# Rounds UP: never ask for fewer rows than the interleave will consume.
check("rounds up", target_rows_for(10, 1.0, 3.0), 4)
try:
    target_rows_for(100, 1.0, 0.0)
    check("zero weights raises", "no raise", "ValueError")
except ValueError:
    check_true("zero weights raises", True)

# ── shards_for_rows: headroom + round-up ─────────────────────────────────────────────────
# science-qa: 400k rows / ~44.9k per shard = 8.9 -> 9 shards at headroom 1.0
check("science-qa @1.0x", shards_for_rows(400_000, 44_871, 1.0), 9)
# multihop's shards are ~6x smaller (7.6k rows) -> needs ~6x the shards for the SAME rows.
# This is the whole reason a global GROUND_DATA_SHARDS=N unbalances the mix.
check("multihop @1.0x", shards_for_rows(400_000, 7_619, 1.0), 53)
check("headroom 1.25x", shards_for_rows(400_000, 44_871, 1.25), 12)
check("always rounds up", shards_for_rows(1, 1000, 1.0), 1)
check("zero rows_per_shard -> 0", shards_for_rows(400_000, 0, 1.0), 0)

# ── select_shards ────────────────────────────────────────────────────────────────────────
files = [f"data-{i:05d}.parquet" for i in range(74)]
check("head takes the front", select_shards(files, 3, "head", 42, "r"), files[:3])
a = select_shards(files, 9, "random", 42, "repo/a")
b = select_shards(files, 9, "random", 42, "repo/a")
check_true("random is deterministic (=> resumable)", a == b)
check_true("random honours n", len(a) == 9)
check_true("random is not the head", a != files[:9], f"got {a[:3]}...")
check_true("random is a subset, no dupes", set(a) <= set(files) and len(set(a)) == 9)
check_true("different seed -> different pick",
           select_shards(files, 9, "random", 43, "repo/a") != a)
# Seed is mixed with the repo name so two repos don't select correlated shard positions.
check_true("different repo -> different pick",
           select_shards(files, 9, "random", 42, "repo/b") != a)
# Asking for more shards than exist clamps rather than raising.
check("clamps to available", len(select_shards(files, 999, "random", 42, "r")), 74)
check("n=0 -> empty", select_shards(files, 0, "random", 42, "r"), [])
try:
    select_shards(files, 3, "sideways", 42, "r")
    check("bad mode raises", "no raise", "ValueError")
except ValueError:
    check_true("bad mode raises", True)

# ── build_plan end-to-end, on the REAL measured numbers ──────────────────────────────────
# Measured 2026-07-16 from the HF API (see wiki/data/hf-rate-limits.md).
real = [
    SourcePlan(repo="vm2825/science-qa-hard-neg-think", total_rows=3_320_432,
               total_shards=74, total_bytes=33_460_000_000),
    SourcePlan(repo="vm2825/diverseqa-hard-neg-think", total_rows=2_676_612,
               total_shards=68, total_bytes=34_480_000_000),
    SourcePlan(repo="vm2825/triviaqa-hotpotqa-nq-squad-msmarco-hard-neg-sft4b",
               total_rows=759_858, total_shards=16, total_bytes=1_590_000_000),
    SourcePlan(repo="ragrawal36/multihop_qa_sft", total_rows=1_348_595,
               total_shards=177, total_bytes=5_470_000_000),
]
fbr = {s.repo: [f"s-{i:05d}.parquet" for i in range(s.total_shards)] for s in real}
build_plan(real, 1_600_000, headroom=1.0, mode="random", seed=42, files_by_repo=fbr)

for s in real:
    check(f"target rows {s.repo.split('/')[-1][:18]}", s.target_rows, 400_000)
check("science-qa shards", len(real[0].selected), 9)
check("diverseqa shards", len(real[1].selected), 11)
check("triviaqa shards", len(real[2].selected), 9)
check("multihop shards", len(real[3].selected), 53)

total_gb = sum(s.est_bytes for s in real) / 1e9
print(f"\n  full-epoch plan @1.0x headroom = {total_gb:.2f} GB (vs 75.00 GB for the whole mix)")
check_true("epoch plan is ~12 GB", 11.0 < total_gb < 13.5, f"{total_gb:.2f} GB")
check_true("epoch plan fits a 97GB boot disk", total_gb < 30)

# Every source must be able to supply its share, or all_exhausted will repeat it.
for s in real:
    check_true(f"{s.repo.split('/')[-1][:18]} can supply its share",
               s.total_rows >= s.target_rows, f"{s.total_rows:,} >= {s.target_rows:,}")

# Headroom must only ever grow the plan.
h = [SourcePlan(repo=s.repo, total_rows=s.total_rows, total_shards=s.total_shards,
                total_bytes=s.total_bytes) for s in real]
build_plan(h, 1_600_000, headroom=1.25, mode="random", seed=42, files_by_repo=fbr)
check_true("headroom grows the plan",
           all(len(x.selected) >= len(y.selected) for x, y in zip(h, real)))
print(f"  @1.25x headroom = {sum(s.est_bytes for s in h)/1e9:.2f} GB")

# A weighted mix must shift rows, not total them differently.
w = [SourcePlan(repo=s.repo, weight=(3.0 if i == 0 else 1.0), total_rows=s.total_rows,
                total_shards=s.total_shards, total_bytes=s.total_bytes)
     for i, s in enumerate(real)]
build_plan(w, 1_200_000, headroom=1.0, mode="random", seed=42, files_by_repo=fbr)
check("weighted: 3x source gets half", w[0].target_rows, 600_000)
check("weighted: 1x source gets a sixth", w[1].target_rows, 200_000)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("ALL PASS")
