# Infrastructure & Setup

How to stand up compute and run experiments on the TRC TPUs, plus the checkpoint format.
Agent-facing runbooks — getting-started is folded in here rather than split across pages.

## Pages
- **➡ [experiment-launch-instructions.md](experiment-launch-instructions.md)** — start here.
  The end-to-end runbook: TRC TPU coordinates + auth, provisioning a box with tpunanny, what
  `multi-vm-tpu-run.sh` does, manual-launch best practices, and checkpoint/resume.
- [checkpointing.md](checkpointing.md) — Orbax save/restore internals: run-dir layout,
  reshard-on-restore, and resume semantics.

Environment details (`.env` variables, GCS credentials, disk limits, HF pre-caching) live
inline in the launch runbook, next to the step that needs them.
