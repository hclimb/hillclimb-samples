# Indexed data-loader resume + warm-start

O(1) resume for `QADatasetIndexed` (`storage: arrayrecord`). Any ckpt is
resumable at the correct data position — even pre-fix ckpts with no sibling
`dataloader_state.json`, and even ckpts whose state file is stale or corrupt
— because the trainer synthesizes the state from `step * batch_size` when the
file path fails a sanity check. Two workflows: **warm-start** (fresh fine-tune
from a ckpt, no data continuity required) and — as an emergency lever, rarely
needed now — **manual mid-run rescue**.

## Prerequisites: two commits

1. **`db492d76`** on `lr` — `trainer: actually save/restore Grain iterator
   state under is_indexed`. Turned save/restore back on after the original
   `8227fcab` skip-under-`is_indexed` early-return.
2. **`data,trainer: synthesize indexed loader state on restore`** on
   `data_resume` (also cherry-picked to `lr`). Adds the fallback that closes the
   tpunanny preempt-and-respawn loophole: whenever the sibling state file is
   missing, undecodable, or has `max(last_seen_indices)` too far from
   `step*batch_size − 1`, the trainer queues a synth position K via
   `data.set_loader_state(K)` and `QADatasetIndexed.generator` materializes
   the exact Grain state on first iterator construction. See
   `wiki/implementations/2026-08-12-indexed-loader-synth-on-restore.md`.

Combined effect: any ckpt (pre-fix `8227fcab`, post-fix `db492d76`, or
corrupted-state-file) resumes at the correct data position with **no manual
intervention**.

## Workflow A: warm-start from a ckpt (recommended)

Use when the goal is only weight initialization — teammate is fine-tuning a
different task, or the original data trajectory is not worth preserving. The
iterator yields a fresh permutation from position 0 and every ckpt saved
after launch writes its own correct `dataloader_state.json` sibling, so all
subsequent resumes are byte-consistent.

1. Point trainer at the warm-start ckpt (`trainer.resume_from=gs://...`) and
   set indexed storage:
   ```yaml
   dataset.storage: arrayrecord
   dataset.indexed_uri: gs://memory-layers-training/indexed/<config-hash>
   ```
2. If the warm-start ckpt was written by a prior run and has a sibling
   `dataloader_state.json` you do NOT want to inherit, delete it:
   ```bash
   gsutil rm gs://<run>/qwen3_mem_embed/<ckpt_step>/dataloader_state.json
   ```
3. Launch. Log will show `No dataloader state at step N; stream starts from 0.`
   — that's the intended path for a warm-start.

From here on every ckpt saves a correct sibling state file; resumes are O(1)
and byte-consistent with an uninterrupted run.

## Workflow B: manual mid-run rescue (emergency lever, rarely needed)

**With the synthesis-on-restore prerequisite above, this is redundant** — the
trainer does exactly this synthesis automatically at resume time, using the
same `K = ckpt_step × batch_size` formula. Keep this section as an emergency
lever for two edge cases:

1. You want to *inspect or edit* the state file before launching (e.g., to
   simulate resume from a specific fabricated position).
2. You're launching a run on a stale branch that doesn't yet have the
   synth-on-restore fallback but does have `db492d76`.

For everything else: just launch. Log will show `[loader-restore]
synthesized indexed state at K=… (file=missing)` — that's the fallback firing.

To synthesize by hand: build the state file at global position
`K = ckpt_step × batch_size` and upload it as a sibling to the ckpt.

### Step 1: obtain the exact `sampler` + `data_source` strings

Grain's `_validate_state` requires byte-for-byte match on these two `repr()`
strings — a single character off fails validation and the iterator falls back
to position 0. Cheapest way to get them: read from any existing sibling state
file produced by the same config.

```bash
gsutil cat gs://<run>/qwen3_mem_embed/<any_ckpt_with_sibling>/dataloader_state.json \
  | python3 -c 'import json,base64,sys; \
                w=json.load(sys.stdin); \
                s=json.loads(base64.b64decode(w["__bytes_b64"]).decode()); \
                print("SAMPLER:", s["sampler"]); \
                print("DATA_SOURCE:", s["data_source"])'
```

If no such file exists yet, launch the same-config run briefly, let one ckpt
save, then read it as above.

### Step 2: synthesize and upload the state file

```python
import json, base64

SAMPLER      = "IndexSampler(num_records=1553096, shard_options=NoSharding(shard_index=0, shard_count=1, drop_remainder=False), shuffle=True, num_epochs=10, seed=42)"
DATA_SOURCE  = "ArrayRecordDataSource(hash_of_paths=3a923784d02d050bc53a0f296e623639c4f3faa1)"
NUM_WORKERS  = 16    # must match cfg.dataset.num_workers
BATCH_SIZE   = 16
CKPT_STEP    = 100000

K = CKPT_STEP * BATCH_SIZE

state = {
    "version": 2,
    "last_seen_indices": {str(i): i + K - NUM_WORKERS for i in range(NUM_WORKERS)},
    "last_worker_index": NUM_WORKERS - 1,
    "worker_count": NUM_WORKERS,
    "sampler": SAMPLER,
    "data_source": DATA_SOURCE,
}
bytes_state = json.dumps(state, indent=4).encode()
wrapped = {"__bytes_b64": base64.b64encode(bytes_state).decode("ascii")}
with open("dataloader_state.json", "w") as f:
    json.dump(wrapped, f)
```

Upload next to the target ckpt:

```bash
gsutil cp dataloader_state.json \
  gs://<run>/qwen3_mem_embed/<CKPT_STEP>/dataloader_state.json
```

### Step 3: verify at launch

Log MUST show both lines in order:

```
Restored dataloader state from gs://.../<CKPT_STEP>/dataloader_state.json
[QADatasetIndexed] restored Grain iterator state (byte-len XXX)
```

Failure mode — validation mismatch — shows instead as:

```
UserWarning: Could not restore indexed dataloader state
(<field> in checkpoint does not match <field> in dataloader); starting from
index 0.
```

The message names which of `sampler` / `data_source` / `worker_count` didn't
match; fix and re-upload.

## Formula (why this works)

Grain's `IndexSampler` produces a global permutation of `[0, num_records)` of
length `num_epochs × num_records`. The `DataLoader` shards this among
`worker_count` workers round-robin: worker `i` owns permutation positions
`{i, i + worker_count, i + 2·worker_count, ...}`.

`get_state()` returns per-worker `last_seen_indices[i] = -worker_count + i +
next_index[i] · worker_count`. Inverting: for each worker to have consumed
`K / worker_count` items, its last-seen permutation position is
`i + K - worker_count`. `last_worker_index = worker_count - 1` makes worker 0
yield next (at position `K`).

`set_state()` inverts back via `next_index[i] = (last_seen[i] + worker_count -
i) // worker_count`, so the round-trip is exact.

## File format

Written by `_save_loader_state` in `trainer/trainer.py`. Grain returns
`iterator.get_state()` as `bytes` (internally `json.dumps(state).encode()`);
the trainer base64-wraps them for JSON-safe storage so the same on-disk file
format works for both the streaming path (dict state) and the indexed path
(bytes state):

```json
{"__bytes_b64": "<base64(iterator.get_state())>"}
```

On restore, `_restore_loader_state` detects the `__bytes_b64` key, decodes,
and hands raw bytes to `iterator.set_state()`. The streaming path (dict
state) round-trips unchanged.
