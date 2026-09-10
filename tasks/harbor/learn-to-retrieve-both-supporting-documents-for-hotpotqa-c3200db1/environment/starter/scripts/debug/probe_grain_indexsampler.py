"""Phase 0 spike: which Grain API supports ArrayRecord + O(1)-resume cleanly?

Answers three questions against the installed Grain version:
  1. Does DataLoader + IndexSampler work end-to-end on an ArrayRecordDataSource?
  2. Can we seek to arbitrary index N by re-instantiating the sampler with a
     start_index / last_seen_index? (i.e. is resume truly O(1)?)
  3. Does IndexSampler's ckpt-validation signature include `num_records` +
     `num_epochs` + `seed`, such that changing any of them invalidates?

Also spikes the alternative MapDataset path:
  4. Does grain.MapDataset with .shuffle(seed).repeat(num_epochs) provide an
     equivalent O(1)-resume story? (In particular, can we skip N samples
     without walking through them?)

Prints a verdict at the end recommending which API to build the pipeline on.

Run:
  cd $HOME/memory-layers && source .venv/bin/activate
  python scripts/debug/probe_grain_indexsampler.py
"""
import os, sys, shutil, tempfile, traceback
import grain.python as gp


def title(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


def write_toy_arrayrecord(path, n=1000):
    """Write n records to path as an ArrayRecord shard."""
    try:
        from array_record.python.array_record_module import ArrayRecordWriter
    except ImportError as e:
        print(f"  array_record module not importable: {e}")
        return False
    w = ArrayRecordWriter(path, "group_size:1")
    for i in range(n):
        w.write(f"record-{i:06d}".encode())
    w.close()
    print(f"  wrote {n} records to {path} ({os.path.getsize(path)} bytes)")
    return True


def q1_dataloader_indexsampler(shard_path, n):
    title("Q1: DataLoader + IndexSampler end-to-end on ArrayRecordDataSource")
    src = gp.ArrayRecordDataSource([shard_path])
    print(f"  source len: {len(src)}")
    sampler = gp.IndexSampler(
        num_records=n, shuffle=True, seed=42, num_epochs=2,
        shard_options=gp.NoSharding(),
    )
    dl = gp.DataLoader(
        data_source=src, sampler=sampler, operations=[], worker_count=0
    )
    it = iter(dl)
    first_100 = [next(it) for _ in range(100)]
    print(f"  first 3 records: {first_100[:3]}")
    print(f"  fetched 100 records, PASS")
    return first_100


def q2_seek_resume(shard_path, n, first_100):
    title("Q2: seek to arbitrary index by re-instantiating sampler (O(1) resume?)")
    src = gp.ArrayRecordDataSource([shard_path])
    # Method A: last_seen_index (Grain's checkpoint state key)
    sampler = gp.IndexSampler(
        num_records=n, shuffle=True, seed=42, num_epochs=2,
        shard_options=gp.NoSharding(),
    )
    dl = gp.DataLoader(
        data_source=src, sampler=sampler, operations=[], worker_count=0,
    )
    # Grain's checkpoint API: dl.__iter__().get_state() / set_state()
    it = iter(dl)
    [next(it) for _ in range(50)]
    state = it.get_state()
    print(f"  after 50 items, iterator.get_state() = {state!r}")

    # Continue for 50 more items — the "uninterrupted" continuation
    continuation = [next(it) for _ in range(50)]
    print(f"  continuation items 50-59: {continuation[:10]}")

    # Restart: new DataLoader, new iterator, set_state to the saved state
    dl2 = gp.DataLoader(
        data_source=src, sampler=sampler, operations=[], worker_count=0,
    )
    it2 = iter(dl2)
    it2.set_state(state)
    resumed = [next(it2) for _ in range(50)]
    print(f"  after set_state + 50 items, resumed items 0-9: {resumed[:10]}")

    match = continuation == resumed
    print(f"  continuation == resumed?  {'PASS' if match else 'FAIL'}")
    if not match:
        print(f"    first diff: cont[0]={continuation[0]!r} vs resumed[0]={resumed[0]!r}")
    return match


def q3_signature_validation(shard_path, n):
    title("Q3: IndexSampler signature includes num_records / num_epochs / seed")
    # Build a checkpoint from a run with num_records=n, num_epochs=2, seed=42.
    # Then try to restore with a different num_records — expect an error.
    src = gp.ArrayRecordDataSource([shard_path])
    s_orig = gp.IndexSampler(num_records=n, shuffle=True, seed=42, num_epochs=2,
                             shard_options=gp.NoSharding())
    print(f"  IndexSampler repr: {repr(s_orig)}")
    s_diffn = gp.IndexSampler(num_records=n - 1, shuffle=True, seed=42, num_epochs=2,
                              shard_options=gp.NoSharding())
    print(f"  IndexSampler(num_records={n-1}) repr: {repr(s_diffn)}")
    s_diffe = gp.IndexSampler(num_records=n, shuffle=True, seed=42, num_epochs=3,
                              shard_options=gp.NoSharding())
    print(f"  IndexSampler(num_epochs=3) repr: {repr(s_diffe)}")
    s_diffs = gp.IndexSampler(num_records=n, shuffle=True, seed=43, num_epochs=2,
                              shard_options=gp.NoSharding())
    print(f"  IndexSampler(seed=43) repr: {repr(s_diffs)}")
    reprs_differ = len({repr(s_orig), repr(s_diffn), repr(s_diffe), repr(s_diffs)}) == 4
    print(f"  all 4 reprs distinct?  {'PASS' if reprs_differ else 'FAIL'}")
    print("  (If reprs are distinct, Grain's ckpt-string-repr validation will catch "
          "any mismatch loudly rather than silently reusing wrong-N shards.)")
    return reprs_differ


def q4_mapdataset_path(shard_path, n):
    title("Q4 (alternative): grain.MapDataset with shuffle().repeat() — O(1) skip?")
    import grain
    # Grain 0.2.18: MapDataset.source(...) is the constructor.
    src = grain.MapDataset.source(gp.ArrayRecordDataSource([shard_path]))
    ds = src.shuffle(seed=42).repeat(num_epochs=2)
    # Random access: ds[50] should be O(1)
    r_via_index = ds[50]
    r_via_iter  = None
    it = iter(ds)
    for i in range(51):
        r_via_iter = next(it)
    print(f"  ds[50] via __getitem__: {r_via_index!r}")
    print(f"  50th via iter:          {r_via_iter!r}")
    match = r_via_index == r_via_iter
    print(f"  match?  {'PASS' if match else 'FAIL'}")
    print("  Note: MapDataset supports __getitem__, so 'resume from index N' is literally ds[N] + slice.")
    return match


def main():
    tmpdir = tempfile.mkdtemp(prefix="grain_probe_")
    print(f"Using tmpdir: {tmpdir}")
    shard = os.path.join(tmpdir, "toy.arrayrecord")
    n = 1000
    if not write_toy_arrayrecord(shard, n=n):
        print("FATAL: could not write toy shard; aborting")
        sys.exit(2)
    results = {}
    try:
        first_100 = q1_dataloader_indexsampler(shard, n)
        results["q1"] = True
    except Exception:
        traceback.print_exc()
        results["q1"] = False
        first_100 = None

    if first_100 is not None:
        try:
            results["q2"] = q2_seek_resume(shard, n, first_100)
        except Exception:
            traceback.print_exc()
            results["q2"] = False
    else:
        results["q2"] = None

    try:
        results["q3"] = q3_signature_validation(shard, n)
    except Exception:
        traceback.print_exc()
        results["q3"] = False

    try:
        results["q4"] = q4_mapdataset_path(shard, n)
    except Exception:
        traceback.print_exc()
        results["q4"] = False

    title("VERDICT")
    for k, v in results.items():
        print(f"  {k}: {v}")
    if results.get("q2") and results.get("q3"):
        print("\n  → RECOMMEND: DataLoader + IndexSampler path (q1/q2/q3 all green).")
    elif results.get("q4"):
        print("\n  → RECOMMEND: MapDataset path (q4 green, DataLoader path fails).")
    else:
        print("\n  → NO GREEN PATH. Investigate before proceeding.")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
