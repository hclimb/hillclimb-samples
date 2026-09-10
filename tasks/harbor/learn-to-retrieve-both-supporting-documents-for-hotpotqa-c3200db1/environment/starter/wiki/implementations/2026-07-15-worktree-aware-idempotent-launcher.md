# Worktree-aware, idempotent TPU launcher

**Date:** 2026-07-15 · **Author:** rohunagrawal · **Status:** done · **Branch:** `train_approx_top_k`

## What changed

`scripts/infrastructure/multi-vm-tpu-run.sh` and `multi-vm-tpu-setup.sh` were made **correct from a
git worktree**, **env-overridable**, and **idempotent on a pre-existing box** (plus a fail-fast
guard). Launching a run from a worktree now syncs the worktree's code (not the main checkout) and
skips the venv rebuild when the box already has it.

## Motivation & context

The launcher hardcoded `cd /home/rohunagrawal/memory-layers` for the tar, so running it from a
worktree silently synced the **main checkout** instead — omitting worktree-only/uncommitted files
— while reading `.env` from the *invocation* CWD (which a worktree doesn't have, since `.env` is
gitignored and not inherited). The setup script `rm -rf memory-layers` + `uv venv` +
`uv pip install -e .` on **every** run, so even a pre-existing, fully-provisioned box paid a
multi-minute rebuild. See [experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md) §3.

## Options weighed & tradeoffs

- **Derive paths from git** (chosen) vs. keep a hardcoded path and require running from main:
  `git rev-parse --show-toplevel` gives the tree you're in (→ tar source), and
  `dirname $(git rev-parse --git-common-dir)` always resolves to the main checkout (→ `.env`
  source). Works identically from main or any worktree, no per-user edits.
- **Idempotent setup (hash-marked deps)** vs. always-rebuild: rebuild is simplest but wastes
  minutes on a warm box. A sha256 marker of `pyproject.toml`(+`uv.lock`) in `.venv/.deps-sha256`
  reinstalls only when deps actually change. Trade-off: a corrupt venv isn't auto-healed — delete
  `.venv` to force a clean rebuild.
- **Preserve `.venv` by extracting over the tree** (chosen) vs. `rm -rf` + clone: to keep the venv
  we stopped wiping the dir; instead we clear tracked files but keep `.venv`/`.git`, then extract.
- **Env-overridable vars with defaults** vs. edit-in-place: `: "${VAR:=default}"` makes every knob
  settable from the environment (`TPU_NAME=… RUN_SCRIPT_PATH=… bash …`) and self-documents that
  they're meant to change. Default `TPU_NAME` updated `rohun-tn-v6e-8-a-1` → `rohun-v6e-8-0`.

## How it was built & integrated

`multi-vm-tpu-run.sh`: `set -euo pipefail`; all six vars → `${VAR:-default}` (+ `WORKER`);
`REPO_ROOT`/`MAIN_CHECKOUT`/`ENV_FILE` from git; tar from `REPO_ROOT`; **fail-fast guard** that the
`RUN_SCRIPT_PATH` is in the tarball; GCS ADC copy made **best-effort** (warn+skip if the local ADC
is absent — it was, for `ra3440@columbia.edu`). `multi-vm-tpu-setup.sh`: install uv only if
missing; extract preserving `.venv`/`.git`; `uv venv` only if absent; deps install gated on the
sha256 marker; `wandb login` only if `~/.netrc` lacks `api.wandb.ai`.

**Subtle bug found & fixed:** the guard was `tar -tzf … | grep -qx …`. Under `set -o pipefail`,
`grep -q` exits on first match and closes the pipe → `tar` gets SIGPIPE (exit 141) → the pipeline
reports failure *on a match*. Fixed by listing to a var first (`TAR_LIST=$(tar -tzf …); grep -qx
… <<<"$TAR_LIST"`). This is why the first real launch false-failed the guard.

## Reference pages updated

[experiment-launch-instructions.md](../infrastructure/experiment-launch-instructions.md) §0 TL;DR
+ §3 — worktree-aware, env-overridable, idempotent setup, best-effort ADC, fail-fast guard.

## Tests

Infra scripts (no `tests/` entry — validated by use):

- `bash -n` on both scripts → clean.
- Guard fix under `pipefail`: old pattern `FALSE-FAIL (rc=141)`, new pattern `MATCH`.
- End-to-end from the `train_approx_top_k` **worktree** to `rohun-v6e-8-0`: sync + guard pass +
  idempotent setup all succeeded — log showed `[setup] refreshing code (preserving .venv)`,
  `[setup] deps up to date — skipping install` (2nd run), `[setup] wandb already logged in`,
  `[setup] done`, then the run script launched. The bench harness it launched
  (`scripts/embed/bench_approx_topk.py --mode check`) returned `compose+imports OK`.

## Follow-ups & risks

- Deleted-since files linger if you only ever extract-over (we clear tracked files first, so this
  is bounded to non-tracked leftovers). A corrupt `.venv` needs a manual `rm -rf .venv`.
- Multi-host pods: unchanged (`--worker=all`); only exercised on single-host `v6e-8` here.
