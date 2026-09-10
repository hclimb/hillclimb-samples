#!/usr/bin/env bash
#
# Sync this repo to a TPU box and run a script on it.
#
# Works from the main checkout OR any git worktree: it tars the tree you are standing in
# and pulls the (gitignored) .env from the main checkout. Every setting below is a shell
# variable with a default and can be overridden from the environment WITHOUT editing this
# file, e.g.:
#
#   TPU_NAME=rohun-v6e-8-0 \
#   RUN_SCRIPT_PATH=scripts/embed/bench_approx_topk.sh \
#     bash scripts/infrastructure/multi-vm-tpu-run.sh
#
# The run is launched inside a detached tmux session ON THE BOX (DETACH=1, default), so it
# survives this SSH connection — and this process — dying. You no longer have to wrap the
# launcher in tmux/nohup yourself. We then tail the box-side log so you still see live output
# and get the real exit code; Ctrl-C here stops the tailing, NOT the run.
#   DETACH=0        legacy: run over one blocking SSH (dies with the connection)
#   FOLLOW=0        start the run and return immediately, don't tail
#   TMUX_SESSION=x  session name on the box (default: derived from the run script)
# Re-attach later:  gcloud alpha compute tpus tpu-vm ssh <box> ... --command='tmux attach -t <session>'
#
# To run several boxes at once (e.g. training + an eval box), use multi-tpu-box-run.sh, which
# calls this script per box with a distinct TARBALL and session.
set -euo pipefail

# ─────────────────────────── Settable config (override via env) ───────────────────────────
TPU_NAME="${TPU_NAME:-rohun-v6e-8-0}"                       # target box (see `gcloud ... tpus tpu-vm list`)
ZONE="${ZONE:-europe-west4-a}"
PROJECT_ID="${PROJECT_ID:-memorylayers}"
RUN_SCRIPT_PATH="${RUN_SCRIPT_PATH:-scripts/embed/pretraining_cot.sh}"  # repo-relative .sh to run on the box
# GCS_USER_EMAIL: identity whose ADC the box uses for GCS. Resolved below from .env (the single
# source of truth — utils.py::setup_gcs_credentials and the box run scripts read the same var),
# unless already set in this environment.
GCLOUD_USERNAME="${GCLOUD_USERNAME:-rohunagrawal}"          # your username on the box
WORKER="${WORKER:-all}"                                     # gcloud --worker (all | 0 | ...); tpu transport only
# Transport for reaching the box:
#   tpu (default) — Cloud TPU VM, addressed via `gcloud alpha compute tpus tpu-vm ssh/scp`
#   gce           — a Compute Engine VM with an attached TPU (machine type ct6e-*, e.g. tpu-v6e-vm).
#                   These are invisible to the Cloud TPU API, so they need `gcloud compute ssh/scp`.
TRANSPORT="${TRANSPORT:-tpu}"
DETACH="${DETACH:-1}"                                       # 1 = run in tmux on the box (survives SSH drop)
FOLLOW="${FOLLOW:-1}"                                       # 1 = tail the box-side log until the run exits
# Space-separated KEY=VAL exported for the box-side run script. ssh does not forward env, so this
# is the only way to hand it a variable — e.g. RUN_ENV="RUN_START_TIME=2026-07-16-18-00-00" so a
# training box and an eval box agree on one run identity. See multi-tpu-box-run.sh.
RUN_ENV="${RUN_ENV:-}"
# tmux session name on the box; default from the script basename (train_hard_neg_think.sh -> run-train_hard_neg_think)
TMUX_SESSION="${TMUX_SESSION:-run-$(basename "${RUN_SCRIPT_PATH%.sh}")}"
# ──────────────────────────────────────────────────────────────────────────────────────────

# Resolve repo paths from git so this is correct from a worktree too:
#   REPO_ROOT     = the tree you're standing in  -> what we tar & sync (picks up uncommitted edits)
#   MAIN_CHECKOUT = the primary working tree      -> where the gitignored .env lives
#                   (--git-common-dir always points at the main checkout's .git)
REPO_ROOT="$(git rev-parse --show-toplevel)"
MAIN_CHECKOUT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
ENV_FILE="${ENV_FILE:-$MAIN_CHECKOUT/.env}"
TARBALL="${TARBALL:-/tmp/memory_layers_sync.tar.gz}"

echo "[run] tpu=$TPU_NAME  zone=$ZONE  worker=$WORKER  transport=$TRANSPORT"
echo "[run] run-script=$RUN_SCRIPT_PATH"
echo "[run] sync tree=$REPO_ROOT"
echo "[run] env file=$ENV_FILE"

[ -f "$ENV_FILE" ] || { echo "ERROR: .env not found at $ENV_FILE" >&2; exit 1; }

# Inherit the GCS identity from .env rather than hardcoding it here: .env is what actually lands on
# the box and drives utils.py::setup_gcs_credentials + the box run scripts, so a default baked in
# here could disagree with what the box really uses — copying one identity's ADC while the box
# reads another's. That mismatch is invisible until something touches GCS and dies on
# DefaultCredentialsError. An explicit env var still wins.
if [ -z "${GCS_USER_EMAIL:-}" ]; then
  GCS_USER_EMAIL="$(grep -E '^(export )?GCS_USER_EMAIL=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
fi
[ -n "$GCS_USER_EMAIL" ] || { echo "ERROR: GCS_USER_EMAIL not set and not found in $ENV_FILE" >&2; exit 1; }
echo "[run] gcs identity=$GCS_USER_EMAIL"

# Tar the tree you're in (tracked + untracked, .gitignore-aware).
( cd "$REPO_ROOT" && git ls-files --cached --others --exclude-standard \
    | tar -czf "$TARBALL" -C . -T - )

# Fail fast if the script we're about to run didn't make it into the synced tree — this is
# the exact failure mode of syncing the wrong checkout (e.g. a worktree-only script omitted).
# Note: list to a var first — `tar | grep -q` would SIGPIPE tar (grep exits early) and, under
# `set -o pipefail`, report a false failure even on a match.
TAR_LIST="$(tar -tzf "$TARBALL")"
if ! grep -qx "$RUN_SCRIPT_PATH" <<<"$TAR_LIST"; then
  echo "ERROR: '$RUN_SCRIPT_PATH' is not in the synced tarball." >&2
  echo "       Is it tracked/untracked (not gitignored) under $REPO_ROOT?" >&2
  exit 1
fi

if [ "$TRANSPORT" = "gce" ]; then
  # A TPU attached to a plain Compute Engine VM (machine type ct6e-*) is NOT a Cloud TPU node:
  # `gcloud compute tpus tpu-vm describe` returns NOT_FOUND for it, so the tpu-vm ssh/scp above
  # can't address it at all. Everything else (tarball, idempotent setup, tmux launch, log tail)
  # is transport-agnostic, so only these two functions change.
  # --worker is a multi-host tpu-vm concept with no GCE equivalent; strip it so the shared call
  # sites below (which pass --worker=0) keep working unchanged.
  _gce_args() { local a; _A=(); for a in "$@"; do [ "${a#--worker=}" = "$a" ] && _A+=("$a"); done; }
  gc_ssh() { _gce_args "$@"; gcloud compute ssh "$TPU_NAME" --tunnel-through-iap \
               --zone="$ZONE" --project="$PROJECT_ID" ${_A[@]+"${_A[@]}"}; }
  gc_scp() { _gce_args "$@"; gcloud compute scp --tunnel-through-iap \
               --zone="$ZONE" --project="$PROJECT_ID" ${_A[@]+"${_A[@]}"}; }
else
  gc_ssh() { gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --tunnel-through-iap \
               --zone="$ZONE" --project="$PROJECT_ID" --worker="$WORKER" "$@"; }
  gc_scp() { gcloud alpha compute tpus tpu-vm scp --tunnel-through-iap \
               --zone="$ZONE" --project="$PROJECT_ID" --worker="$WORKER" "$@"; }
fi

# Copy the GCS ADC the box uses for checkpointing. If it isn't local, CHECK the box rather than
# assume — this used to warn "the box may already have it" and carry on, which turned a missing
# credential into a DefaultCredentialsError deep inside a run minutes later (and stayed invisible
# for as long as every box happened to have been set up by hand).
ADC_REL=".config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/adc.json"
ADC_SRC="$HOME/$ADC_REL"
if [ -f "$ADC_SRC" ]; then
  gc_ssh --command="mkdir -p ~/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}"
  gc_scp "$ADC_SRC" \
    "${TPU_NAME}:/home/${GCLOUD_USERNAME}/.config/gcloud/legacy_credentials/${GCS_USER_EMAIL}/"
else
  echo "[run] local ADC $ADC_SRC not found; checking the box..." >&2
  if gc_ssh --worker=0 --command="test -f \$HOME/$ADC_REL" >/dev/null 2>&1; then
    echo "[run] box already has the ADC for $GCS_USER_EMAIL — continuing" >&2
  else
    echo "ERROR: no GCS credentials for '$GCS_USER_EMAIL' locally OR on $TPU_NAME." >&2
    echo "       Anything touching GCS (checkpoints, eval boxes) will die on" >&2
    echo "       DefaultCredentialsError. Fix with:  gcloud auth login $GCS_USER_EMAIL" >&2
    echo "       (creates ~/$ADC_REL, which this script then copies), then re-run." >&2
    echo "       GCS_USER_EMAIL comes from $ENV_FILE." >&2
    exit 1
  fi
fi

# Ship setup script + .env + code tarball, run idempotent setup, then launch.
gc_scp "$REPO_ROOT/scripts/infrastructure/multi-vm-tpu-setup.sh" "$ENV_FILE" "$TARBALL" "${TPU_NAME}:"
gc_ssh --command="chmod u+x multi-vm-tpu-setup.sh && ./multi-vm-tpu-setup.sh"

[ -n "$RUN_ENV" ] && echo "[run] forwarding env: $RUN_ENV"

if [ "$DETACH" != "1" ]; then
  # Legacy path: one blocking SSH. The run dies if this connection does. RUN_ENV is exported
  # inline here so it behaves the same as the tmux path rather than silently no-op'ing.
  gc_ssh --command="source memory-layers/scripts/infrastructure/setup_shell.sh && ${RUN_ENV:+export $RUN_ENV; }cd memory-layers && source ${RUN_SCRIPT_PATH}"
  exit $?
fi

REMOTE_LOG="\$HOME/runs/${TMUX_SESSION}.log"
echo "[run] launching in tmux session '$TMUX_SESSION' on $TPU_NAME"
gc_ssh --command="bash memory-layers/scripts/infrastructure/tmux_launch.sh '${TMUX_SESSION}' '${RUN_SCRIPT_PATH}' ${RUN_ENV}"

if [ "$FOLLOW" != "1" ]; then
  echo "[run] started detached. Follow with:"
  if [ "$TRANSPORT" = "gce" ]; then
    echo "      gcloud compute ssh $TPU_NAME --tunnel-through-iap --zone=$ZONE --project=$PROJECT_ID --command='tail -f $REMOTE_LOG'"
  else
    echo "      gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --tunnel-through-iap --zone=$ZONE --project=$PROJECT_ID --worker=0 --command='tail -f $REMOTE_LOG'"
  fi
  exit 0
fi

# Tail the box-side log until the sentinel tmux_launch.sh appends on exit. `sed -u .../q` quits at
# the sentinel, which SIGPIPEs tail -> the SSH command returns instead of hanging forever.
# Ctrl-C here only kills the tail; the tmux session on the box keeps running.
echo "[run] following $REMOTE_LOG (Ctrl-C detaches; the run keeps going)"
set +e
gc_ssh --worker=0 --command="tail -n +1 -f $REMOTE_LOG | sed -u '/__RUN_EXIT__=/q'"
RC="$(gc_ssh --worker=0 --command="grep -ao '__RUN_EXIT__=[0-9]*' $REMOTE_LOG | tail -1 | cut -d= -f2" 2>/dev/null | tr -dc '0-9')"
set -e
if [ -z "$RC" ]; then
  echo "[run] WARNING: run still going or log lost its sentinel; session '$TMUX_SESSION' on $TPU_NAME" >&2
  exit 0
fi
echo "[run] $TPU_NAME:$RUN_SCRIPT_PATH exited rc=$RC"
exit "$RC"
