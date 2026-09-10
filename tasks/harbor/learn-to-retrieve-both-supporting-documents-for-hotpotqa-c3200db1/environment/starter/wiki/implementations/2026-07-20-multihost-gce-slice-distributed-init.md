# Distributed init on a multi-host GCE TPU slice

**Date:** 2026-07-20 · **Author:** rohunagrawal · **Status:** done · **Branch:** working tree (uncommitted)

## What changed

`utils.py::init_jax_distributed` now distinguishes a **single-host** GCE-attached TPU from a
**multi-host GCE slice** by the *number* of `worker-network-endpoints`, instead of treating
bare-name metadata as proof of single-host:

| endpoints | shape | action |
|---|---|---|
| 1 | single-host flex box (`ct6e-*`/`ct5p-*`) | skip — init is a no-op at `process_count 1` (unchanged) |
| 2+ | **multi-host slice** (e.g. 2 × `ct6e-standard-4t`, 2x4) | **`jax.distributed.initialize()` with explicit `coordinator_address` / `num_processes` / `process_id`** |

Also added `_gce_instance_attribute(name)` (generalised out of `_tpu_worker_endpoints`) and
`_JAX_COORDINATOR_PORT` (default 8476, override via `JAX_COORDINATOR_PORT`).

## Motivation & context

The old code skipped `jax.distributed.initialize()` whenever no endpoint contained `::`, because
JAX's TPU auto-detect does `worker.split(':')[2]` and dies with `IndexError` on the bare instance
names a GCE-attached TPU publishes. When written, every GCE-attached box in use was single-host,
so skipping was correct **and** the only option.

The **multi-host flex slice** ([runbook §2.3](../infrastructure/experiment-launch-instructions.md))
breaks that equivalence: it publishes bare names *and* has 2 hosts. It landed in the skip branch, so
the coordination service was never started.

**Why this was not caught by inspection — the trap is worth naming.** On such a slice libtpu forms
the ICI mesh unaided, so a probe reports everything healthy with init skipped:

```
process_index/count: 0/2   device_count: 8   local_device_count: 4   # full 2x4 coords
```

I read that as "the skip is harmless here" and said so. It is not: **device discovery and the
distributed system are different things.** The failure surfaces far away, at the first component
that needs coordination — orbax:

```
File "utils.py", line 389, in setup_checkpointing
    ocp.StandardCheckpointer(),
ValueError: Distributed system is not available; please initialize it via
`jax.distributed.initialize()` at the start of your program.
```

That is how the bug reached an actual training launch (the `ground_s1` main-unfreeze smoke test)
rather than being caught by the slice bring-up probe.

## Options weighed & tradeoffs

- **Split on endpoint count + explicit init args** (chosen). The count is exactly the fact that
  distinguishes the two shapes, and explicit args sidestep the parser bug that motivated the
  original skip. Keeps single-host behaviour bit-identical.
- **Always call `jax.distributed.initialize()`** — reintroduces the `IndexError` on single-host
  GCE boxes; that crash is the whole reason the branch exists.
- **Patch/monkeypatch JAX's `cloud_tpu_cluster` parser** to accept bare names — fixes it "properly"
  upstream-style, but pins us to JAX internals across upgrades for no gain over passing the three
  arguments ourselves.
- **`process_id` source:** `agent-worker-number` (chosen) over parsing `tpu-env`'s `TPU_WORKER_ID`
  — a single scalar metadata key rather than a substring of a YAML-ish blob. **Verified they agree**
  on both hosts (1wjb→0, 1z9d→1). Raises a clear error if the key is absent rather than guessing 0,
  which would give two rank-0 processes.
- **Coordinator:** worker 0's bare instance name, resolvable over the VPC's internal DNS.

## Known quirk — JAX does not adopt the `process_id` you pass (cosmetic)

On this slice, `jax.process_index()` is assigned by JAX's own ordering, **inverted** vs the
metadata rank we pass:

| host | metadata rank (both keys) | `process_id` passed | `jax.process_index()` | local device ids |
|---|---|---|---|---|
| `…-1wjb` | 0 | 0 | **1** | `[4,5,6,7]` |
| `…-1z9d` | 1 | 1 | **0** | `[0,1,2,3]` |

**This is safe and was verified, not assumed** (see Tests): ownership is a consistent bijection —
disjoint, complete, and matching each host's claimed index — exactly one process is rank 0, and a
global collective returns the correct value. Nothing in the codebase depends on *which physical
host* is rank 0, only that exactly one is (orbax `primary_host=0`, wandb and `_save_loader_state`
guard on `process_index()==0`). The same inversion appears with auto-detect on Cloud TPU nodes, so
it is not introduced by this change. **Do not "fix" it by trying to force the mapping.**

## How it was built & integrated

- `utils.py`: `_gce_instance_attribute` helper; `_tpu_worker_endpoints` now delegates to it;
  `init_jax_distributed` gains the endpoint-count split and the explicit-init path.
- Docstring rewritten to warn that healthy `device_count` on a multi-host slice does **not** imply
  the distributed system is up, and to name where it actually fails.
- No call sites changed — every entry point already funnels through `init_jax_distributed()`.

## Tests

`scripts/infrastructure/probe_jax_slice.sh` (extended here with local-device ids, a
`process_allgather`, and a checked global collective — the mesh-coherence test the original probe
lacked). Run on **both** workers of `tpu-v6e-slice-mig` (2 × `ct6e-standard-4t`, 2x4,
europe-west4-a):

```bash
TRANSPORT=gce ZONE=europe-west4-a PROJECT_ID=memory-layers RUN_ENV="JAX_PROBE_DISTRIBUTED=1" \
bash scripts/infrastructure/multi-tpu-box-run.sh \
  tpu-v6e-slice-mig-1wjb=scripts/infrastructure/probe_jax_slice.sh \
  tpu-v6e-slice-mig-1z9d=scripts/infrastructure/probe_jax_slice.sh
```

Result — both workers, `rc=0`:

```
[jax] multi-host GCE TPU slice: 2 workers, process_id=0, coordinator=tpu-v6e-slice-mig-1wjb:8476
[probe] process_index/count: 1 / 2      [probe] LOCAL device ids: [4, 5, 6, 7]
[probe] allgather(process_index) = [0, 1]
[probe] global psum: got 28.0 expect 28.0 -> OK
--- other worker ---
[probe] process_index/count: 0 / 2      [probe] LOCAL device ids: [0, 1, 2, 3]
[probe] allgather(process_index) = [0, 1]
[probe] global psum: got 28.0 expect 28.0 -> OK
```

Before the fix, the same slice reached `setup_checkpointing` and died on
`ValueError: Distributed system is not available` (`__RUN_EXIT__=1`, both workers).

**Single-host regression not re-run** — no single-host GCE box was live at the time. The path is
unchanged apart from the `len(workers) <= 1` guard that now precedes the identical print/return,
but it is worth one run on the next `ct6e-standard-4t`/`ct5p-hightpu-4t` flex box.

## Reference pages updated

[experiment-launch-instructions.md §2.3](../infrastructure/experiment-launch-instructions.md) —
the "init is skipped and that's fine" note was **wrong** and is replaced by the real behaviour.

## Follow-ups & risks

- **`JAX_COORDINATOR_PORT` 8476 must be free and identical across workers.** Two concurrent
  multi-host runs on the same slice would collide; override for the second.
- A slice of **>2 hosts** is untested — the logic is count-based so it should hold, but worker 0 as
  sole coordinator becomes a bigger single point of failure.
- The multi-host branch has **no `_tpu_worker_endpoints() is None` fallback**: a box with no
  metadata at all still takes the original auto-detect path, unchanged.
