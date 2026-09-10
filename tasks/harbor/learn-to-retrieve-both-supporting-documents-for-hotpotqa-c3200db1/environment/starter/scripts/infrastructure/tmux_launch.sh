#!/usr/bin/env bash
#
# Runs ON a TPU box. Starts a run script inside a detached tmux session so it survives the
# launching SSH connection dropping — the thing the runbook used to tell you to arrange yourself.
# Invoked by multi-vm-tpu-run.sh (DETACH=1, the default); you can also call it by hand:
#
#   bash scripts/infrastructure/tmux_launch.sh <session> <repo-relative-run-script> [KEY=VAL ...]
#
# Trailing KEY=VAL args are exported for the run script. ssh does not forward env, so this is the
# only channel a launcher has to hand a variable to the box-side script (e.g. RUN_START_TIME, so
# a training box and an eval box agree on one run identity).
#
# Output goes to ~/runs/<session>.log. The last line is always __RUN_EXIT__=<rc>, which is how
# the launcher knows the run finished and with what status (tmux itself won't tell you).
# Re-running kills an existing session of the same name first, so a relaunch is clean.
set -uo pipefail

SESSION="${1:?usage: tmux_launch.sh <session> <run_script_path> [KEY=VAL ...]}"
SCRIPT="${2:?usage: tmux_launch.sh <session> <run_script_path> [KEY=VAL ...]}"
shift 2 || true

# Build `export K='V';` for each KEY=VAL. Single-quoted so values with spaces/globs survive, and
# any embedded quote is escaped — this string is interpolated into the tmux command below.
EXPORTS=""
for kv in "$@"; do
  case "$kv" in
    *=*) ;;
    *) echo "[tmux_launch] ERROR: '$kv' is not KEY=VAL" >&2; exit 2 ;;
  esac
  k="${kv%%=*}"; v="${kv#*=}"
  EXPORTS="${EXPORTS}export ${k}='$(printf "%s" "$v" | sed "s/'/'\\\\''/g")'; "
done
REPO="${REPO_DIR:-$HOME/memory-layers}"
LOG_DIR="$HOME/runs"
LOG="$LOG_DIR/$SESSION.log"

command -v tmux >/dev/null 2>&1 || {
  echo "[tmux_launch] ERROR: tmux not installed on this box" >&2; exit 127; }
[ -f "$REPO/$SCRIPT" ] || {
  echo "[tmux_launch] ERROR: $REPO/$SCRIPT not found (wrong tree synced?)" >&2; exit 1; }

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launch] killing existing session '$SESSION'"
  tmux kill-session -t "$SESSION"
fi
: > "$LOG"

# `source`, not `bash`, to match how runs have always been invoked here (the run scripts set env
# and expect the sourced shell).
#
# The ( ) SUBSHELL around the source is load-bearing. Several run scripts end in `exec`
# (hard_neg_eval_box_run.sh, ground_box_run.sh, box_run.sh) — exec REPLACES the shell, so a
# sentinel echoed in that same shell would never run: the launcher's `tail | sed /__RUN_EXIT__=/q`
# would then wait forever on a line that can't arrive (an orphaned tail long after the run died).
# Running the script inside a subshell means exec only replaces the SUBSHELL; when it exits, the
# outer shell survives to record $? — which is exactly the exec'd program's exit code.
# EXPORTS goes AFTER setup_shell.sh so a forwarded value wins over anything .env sets.
tmux new-session -d -s "$SESSION" \
  "{ ( source '$REPO/scripts/infrastructure/setup_shell.sh'; ${EXPORTS}cd '$REPO' || exit 1; source '$SCRIPT' ); echo \"__RUN_EXIT__=\$?\"; } >> '$LOG' 2>&1"

echo "[tmux_launch] session=$SESSION log=$LOG"
[ -n "$EXPORTS" ] && echo "[tmux_launch] forwarded: $*"
echo "[tmux_launch] attach with: tmux attach -t $SESSION   |   tail -f $LOG"
