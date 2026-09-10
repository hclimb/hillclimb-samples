# Checkpointing

Saving and restoring model + optimizer state with Orbax, locally or on GCS.

**Covers:**
- `setup_checkpointing()`: `ocp.CheckpointManager` + `StandardCheckpointer`,
  `CheckpointManagerOptions`, `MultiprocessingOptions` for multi-host
- Directory layout: `{gcs_run_dir}/{model.name}/{step}/` (bucket / run_name / model / step)
- `load_checkpoint()`: `restore_and_reshard` re-applies TP sharding on restore;
  `partial_restore` for weights-only loads; `resume_step` / `resume_from_dir`
- `load_inference_checkpoint()`: the eval-only restore path (weights only, no opt_state).
  Passes `restore_args=ocp.checkpoint_utils.construct_restore_args(abstract_state)` — built
  from the freshly-initialized `model.weights`' OWN concrete sharding (i.e. the CURRENT box's
  mesh) — so orbax reshards onto whatever chip count/topology this box actually has. Without
  it, orbax falls back to the checkpoint's on-disk sharding metadata (the literal device ids
  the checkpoint was SAVED with); harmless when the eval box has at least as many chips as
  training used (those ids still exist), but a hard crash restoring an 8-chip-trained
  checkpoint on a 4-chip box (`sharding passed to deserialization should be specified,
  concrete... Got None`) — see
  [2026-08-24-checkpoint-restore-cross-topology-sharding](../implementations/2026-08-24-checkpoint-restore-cross-topology-sharding.md).
- `checkpoint_interval` (trainer config) and process-0-only save semantics
- Note: eval auto-loads the training config from inside `checkpoint_dir`

**Source:** `utils.py` (`setup_checkpointing`, `load_checkpoint`, `load_inference_checkpoint`); `trainer/trainer.py`
