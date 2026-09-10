# Eval checkpoint restore: pass explicit `restore_args` for cross-topology resharding

**Date:** 2026-08-24 · **Author:** rohunagrawal (with Claude) · **Status:** done · **Commit:** _TBD_ (uncommitted on `multihop-finetuning`)

## What changed

`utils.py::load_inference_checkpoint` now passes `restore_args=ocp.checkpoint_utils.
construct_restore_args(abstract_state)` to both `checkpoint_manager.restore(...)` calls (the
primary `PyTreeRestore` path and the `PyTreeCheckpointer` fallback). One line of real change
plus a comment; no behavior change on a box with enough chips to satisfy the checkpoint's
original device ids (every prior eval run), and it fixes a hard crash restoring an
8-chip-trained checkpoint on a 4-chip box.

## Motivation & context

Found while smoke-testing the GRPO-readiness multi-sample hybrid eval
([2026-08-24-grpo-readiness-multisample-hybrid-eval](2026-08-24-grpo-readiness-multisample-hybrid-eval.md))
on a freshly-provisioned single-host `tpu-v6e-4-flex` (4 chips): restoring
`qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55/qwen3_mem_embed/90000`
(saved on an 8-chip box) crashed inside orbax before the evaluator ever ran:

```
UserWarning: Sharding info not provided when restoring. Populating sharding info from sharding
file. ... this option is unsafe when restoring on a different topology than the checkpoint was
saved with.
ERROR:root: The available devices are different from the devices used to save the checkpoint.
  Original=[[DeviceMetadata(id=0)], ..., [DeviceMetadata(id=7)], ..., [DeviceMetadata(id=4)]]
  current available=[TpuDevice(id=0), TpuDevice(id=1), TpuDevice(id=2), TpuDevice(id=3)]
...
ValueError: sharding passed to deserialization should be specified, concrete and an instance
of `jax.sharding.Sharding`. Got None
```

**Root cause, verified on the box (not guessed):** `load_inference_checkpoint`'s
`ocp.args.PyTreeRestore(item=abstract_state, partial_restore=True)` never passed `restore_args`
(a separate, distinct field from `item` in orbax's `PyTreeRestoreArgs` — confirmed by reading
`inspect.signature`/`inspect.getsource` on the box's installed `orbax-checkpoint==0.12.4`).
Without `restore_args`, orbax has no target sharding to reshard onto, so it falls back to the
checkpoint's own on-disk sharding metadata verbatim — literal saved `DeviceMetadata` ids. On any
box with **at least** as many chips as the checkpoint was saved with, those ids trivially exist
(TPU device ids are always `0..N-1` per host) and the restore silently "works" by accident, which
is why this had never surfaced in ~a dozen prior eval runs (all on 8-chip `rohun-v6e-8-*` boxes
or the 2-host v6e-8 slice). On a 4-chip box, ids 4-7 don't exist and orbax's internal fallback
produces `sharding=None` for the affected leaves, which the deserializer then rejects outright.
This is a general property of the restore call (affects most/all weight leaves, not a
memory-bank-specific issue — an earlier theory of mine, that the crash was specific to the
dynamically-rebuilt `mem_k`/`mem_v`/`mem_mask` arrays, was wrong and got corrected mid-investigation).

## Options weighed

| Decision | Chosen | Rejected, and why |
|---|---|---|
| How to supply the target sharding | `ocp.checkpoint_utils.construct_restore_args(abstract_state)` — a stock orbax helper that reads `.sharding` straight off `abstract_state`'s own concrete `jax.Array` leaves (i.e. `model.weights`, freshly built for THIS box's mesh by `get_model()` moments earlier) | Hand-building a `restore_args` pytree with explicit `ArrayRestoreArgs(sharding=...)` per leaf — functionally identical to what the stock helper already does, more code to maintain for no benefit |
| Scope | `load_inference_checkpoint` only (the eval-only, weights-only restore path) | Also fixing `load_checkpoint`'s two similar restore calls (training resume / warm-start, `utils.py:782,808`) — same latent gap exists there in principle, but training always resumes on the same box shape it started with, so there's no reported failure and no reason to touch shared training-resume code in this change; flagged as a follow-up instead |
| Where to fix it | The restore call itself | Reprovisioning a v6e-8 box instead and dropping the 4-chip requirement — was on the table (see the parent GRPO note's box discussion) but the user asked to debug the 4-chip failure specifically, and the fix turned out to be a genuine, narrowly-scoped correctness gap rather than a fundamental 4-chip limitation |

## How it was built & integrated

`utils.py::load_inference_checkpoint` — one line added before both restore attempts:
```python
restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
```
then `restore_args=restore_args` passed into both `ocp.args.PyTreeRestore(...)` (primary) and
`checkpointer.restore(...)` (the `PyTreeCheckpointer` fallback for managers using
`StandardCheckpointer`). Everything downstream (`restore_none`, `_put_local`, the returned
`step`) is unchanged.

Diagnosis method, for reference: read `utils.py::load_checkpoint`/`load_inference_checkpoint`,
`models/qwen3_mem_embed.py`'s `init_empty` path and `models/memory_utils.py::add_memory_layer`
to check (and rule out) a memory-bank-specific theory, then went straight to the box and used
`inspect.signature`/`inspect.getsource` on the installed orbax package via SSH to find the real
API (`ocp.args.PyTreeRestore`'s `restore_args` field, `ocp.checkpoint_utils.
construct_restore_args`) rather than guessing at a fix blind.

## Reference pages updated

- [infrastructure/checkpointing.md](../infrastructure/checkpointing.md) — new bullet on
  `load_inference_checkpoint()` and the `restore_args` fix.

## Tests

No CPU-only unit test (this is a live orbax/GCS restore path — mocking it would test the mock,
not the bug). Verified directly on hardware: same command, before and after.

**Before** (uncommitted `utils.py` at the parent commit, `tpu-v6e-4-flex`, 4 chips):
```
ERROR:root:The available devices are different from the devices used to save the checkpoint. ...
ValueError: sharding passed to deserialization should be specified, concrete and an instance of
`jax.sharding.Sharding`. Got None
subprocess.CalledProcessError: ... returned non-zero exit status 1.
```

**After** (this fix applied, same box, same checkpoint/step):
```
Starting GenLargeMemRagHybrid eval (top_k=50, oracle=False, approx_topk=True)...
  corpus (scanned 9811): 8 gold + 9803 distractor chunks = 9811 (target=10000)
  multi_sample=true: group size = mesh data-axis size = 4 independent completions/query (temperature=0.6)
  auto-K: rag.top_k=200 (CAP — no candidate cleared 0.96; candidates=[5, 10, 25, 50, 100, 150, 200])
  retrieval pre-pass done in 82.1s
  gather-bank mode: host bank 2511616 slots; per-row block 200 docs x 256 = 51200 slots
Saved generations to .../eval_results/step_90000/msa_hybrid/outputs/msa_hotpotqa_c10000_hybrid_multisample_smoketest.json
[msa-hybrid-multisample] DONE
```
Full run: `TPU_NAME=tpu-v6e-4-flex ZONE=europe-west4-a PROJECT_ID=memory-layers TRANSPORT=gce
RUN_SCRIPT_PATH=scripts/embed/eval_msa_hybrid_multisample.sh RUN_ENV="DS=hotpotqa RUN_DIR=qa_hard_neg_think_sft4b_topk64_seq512_chunks16_bs16-2026-08-22-17-39-55 STEP=90000 NUM_SAMPLES=4"
bash scripts/infrastructure/multi-vm-tpu-run.sh`, exit code 0.

## Follow-ups & risks

- `load_checkpoint` (training resume/warm-start, `utils.py:782,808`) has the same latent gap —
  untested and unfixed here since it's never been exercised cross-topology (training always
  resumes on the box shape it started on). Worth the same one-line fix before anyone tries to
  resume a training run on a differently-sized box.
- `construct_restore_args` reads sharding off `model.weights`' CURRENT concrete arrays — if a
  future change makes any `model.weights` leaf non-concrete (e.g. a lazy/deferred array) at the
  point `load_inference_checkpoint` is called, this fix silently stops helping and the original
  crash could resurface. Not expected given how `get_model()` builds weights today.
