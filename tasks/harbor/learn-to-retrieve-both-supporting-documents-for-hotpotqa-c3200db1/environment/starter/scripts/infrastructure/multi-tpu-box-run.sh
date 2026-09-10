#!/usr/bin/env bash
#
# Launch several run scripts on several TPU boxes AT THE SAME TIME, one script per box.
# Wraps multi-vm-tpu-run.sh (same sync/setup/tmux behaviour) once per box, in parallel.
#
#   bash scripts/infrastructure/multi-tpu-box-run.sh \
#       rohun-v6e-8-0=scripts/embed/train_hard_neg_think.sh \
#       rohun-v6e-8-1=scripts/embed/hard_neg_eval_box_run.sh
#
# Each pair is <tpu-name>=<repo-relative-run-script>. Output is line-prefixed with the box name.
# Every box's run lands in its own detached tmux session (see multi-vm-tpu-run.sh DETACH), so
# killing this process does NOT kill the runs.
#
# RUN_START_TIME — the reason training + an eval box can launch in PARALLEL. A run's identity is
# its run-dir, <run_name>-<YYYY-MM-DD>-<HH-MM-SS> (utils.py::run_dir_name), and normally the
# timestamp is minted by train.py at startup — so the eval box cannot know the dir until training
# prints it, forcing a serial launch. Here we mint ONE timestamp and forward it to every box
# (RUN_ENV -> tmux_launch.sh), so training pins it via trainer.run_start_time and the eval box
# composes the same RUN_DIR. Override to re-attach boxes to an existing run.
#
# Exit code: 0 only if every box's script exited 0; otherwise the first non-zero rc.
#
# Why this exists rather than `&`-ing two multi-vm-tpu-run.sh calls yourself: that script tars the
# tree to a FIXED /tmp/memory_layers_sync.tar.gz, so two concurrent invocations race on the same
# file and can ship a half-written tarball. Here each box gets a private TARBALL dir (the basename
# must stay memory_layers_sync.tar.gz — multi-vm-tpu-setup.sh looks for exactly that on the box).
#
# Env: passes through ZONE / PROJECT_ID / GCS_USER_EMAIL / GCLOUD_USERNAME / FOLLOW to each box.
#      FOLLOW=0 => start everything and return immediately.
set -uo pipefail

[ $# -ge 1 ] || {
  echo "usage: $0 <tpu>=<run_script> [<tpu>=<run_script> ...]" >&2
  echo "  e.g. $0 rohun-v6e-8-0=scripts/embed/train_hard_neg_think.sh \\" >&2
  echo "          rohun-v6e-8-1=scripts/embed/hard_neg_eval_box_run.sh" >&2
  exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mlsync.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# UTC so the identity doesn't depend on which machine launched. Format must match
# utils.py::RUN_STAMP_RE (YYYY-MM-DD-HH-MM-SS) or train.py refuses to start.
RUN_START_TIME="${RUN_START_TIME:-$(date -u +%Y-%m-%d-%H-%M-%S)}"
export RUN_START_TIME
echo "[multi] RUN_START_TIME=$RUN_START_TIME  (one identity for all boxes)"

declare -a NAMES=() PIDS=() RCFILES=()

for spec in "$@"; do
  case "$spec" in
    *=*) ;;
    *) echo "ERROR: '$spec' is not <tpu>=<run_script>" >&2; exit 2 ;;
  esac
  tpu="${spec%%=*}"
  script="${spec#*=}"
  [ -n "$tpu" ] && [ -n "$script" ] || { echo "ERROR: bad pair '$spec'" >&2; exit 2; }

  # Private tarball dir per box; basename fixed (the box-side setup script matches on it).
  box_tar="$STAGE/$tpu/memory_layers_sync.tar.gz"
  mkdir -p "$(dirname "$box_tar")"
  rc_file="$STAGE/$tpu.rc"

  echo "[multi] $tpu <- $script"
  (
    # RUN_ENV carries the shared identity to the box (ssh forwards no env of its own); any
    # caller-supplied RUN_ENV is preserved alongside it.
    TPU_NAME="$tpu" RUN_SCRIPT_PATH="$script" TARBALL="$box_tar" \
    TMUX_SESSION="run-$(basename "${script%.sh}")" \
    RUN_ENV="RUN_START_TIME=$RUN_START_TIME ${RUN_ENV:-}" \
      bash "$HERE/multi-vm-tpu-run.sh" 2>&1 | sed -u "s/^/[$tpu] /"
    # PIPESTATUS[0] = multi-vm-tpu-run.sh's rc, not sed's.
    echo "${PIPESTATUS[0]}" > "$rc_file"
  ) &
  PIDS+=("$!"); NAMES+=("$tpu"); RCFILES+=("$rc_file")
done

echo "[multi] ${#PIDS[@]} box(es) launching in parallel; waiting..."
for pid in "${PIDS[@]}"; do wait "$pid"; done

FAIL=0
for i in "${!NAMES[@]}"; do
  rc="$(cat "${RCFILES[$i]}" 2>/dev/null || echo "?")"
  echo "[multi] ${NAMES[$i]}: rc=$rc"
  [ "$rc" = "0" ] || { [ "$FAIL" = "0" ] && FAIL="${rc:-1}"; }
done
[ "$FAIL" = "0" ] && echo "[multi] all boxes OK" || echo "[multi] at least one box failed" >&2
exit "$FAIL"
