#!/usr/bin/env bash
set -euo pipefail
readonly SOURCE_ROOT=/environment/starter
readonly PROGRESS_ROOT=/logs/artifacts/progress
readonly CACHE_ROOT=/tmp/optifine-checkpoint-cache
readonly KEYFRAME_INTERVAL=50
readonly EXCLUDED_PATHS='/assets
/runs
/optifine_public_tests
/hotpot_benchmark
/.venv
/venv
/env
/.cache
/__pycache__
/.pytest_cache
/.ruff_cache'
usage() {
    cat <<'EOF'
Usage:
  checkpoint.sh --label LABEL [--note TEXT] [--source DIRECTORY]
  checkpoint.sh --restore CHECKPOINT --destination DIRECTORY
Save a submission checkpoint under /logs/artifacts/progress. The first
snapshot and every 50th snapshot are full zstd keyframes; intervening snapshots
are zstd patches against their immediate parent. Restore depth is at most 49.
Completed checkpoints are read-only; treat the progress directory as managed data.
Task-excluded assets are listed in excluded-paths.txt, not saved in the snapshot.
Examples:
  checkpoint.sh --label faster-kernel --note "Smoke test passes"
  checkpoint.sh --restore 20260831T120000Z-faster-kernel-123 --destination final_app/
EOF
}
fail() {
    printf 'checkpoint.sh: %s\n' "$1" >&2
    exit 2
}
require_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}
metadata_value() {
    local key=$1
    local path=$2
    sed -n "s/^${key}=//p" "$path" | tail -1
}
sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}
file_size() {
    if stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        stat -f %z "$1"
    fi
}
temporary=""
cleanup() {
    if [[ -n "${temporary:-}" && -e "$temporary" ]]; then
        chmod -R u+w "$temporary" 2>/dev/null || true
        rm -rf -- "$temporary"
    fi
}
verify_snapshot() {
    local checkpoint=$1
    local snapshot=$2
    local expected
    local actual
    expected=$(metadata_value snapshot_sha256 "$checkpoint/metadata.txt")
    actual=$(sha256_file "$snapshot")
    [[ -n "$expected" && "$actual" == "$expected" ]] \
        || fail "snapshot checksum mismatch for ${checkpoint##*/}"
}
reconstruct_tar() {
    local checkpoint=$1
    local output=$2
    local work=$3
    local current=$checkpoint
    local checkpoint_root=${checkpoint%/*}
    local kind
    local parent
    local next
    local index
    local -a chain=()
    local chain_length=0
    while :; do
        [[ -f "$current/metadata.txt" ]] || fail "checkpoint metadata is missing: $current"
        if ((chain_length)); then
            chain=("$current" "${chain[@]}")
        else
            chain=("$current")
        fi
        chain_length=$((chain_length + 1))
        kind=$(metadata_value kind "$current/metadata.txt")
        [[ "$kind" == keyframe || "$kind" == patch ]] \
            || fail "invalid checkpoint kind: $kind"
        [[ "$kind" == keyframe ]] && break
        ((chain_length < KEYFRAME_INTERVAL)) \
            || fail "checkpoint chain exceeds $((KEYFRAME_INTERVAL - 1)) patches"
        parent=$(metadata_value parent_checkpoint_id "$current/metadata.txt")
        [[ -n "$parent" ]] || fail "patch checkpoint has no parent"
        current="$checkpoint_root/$parent"
    done
    current="$work/reconstructed-0.tar"
    zstd -q -d -f --long=31 "${chain[0]}/snapshot.tar.zst" -o "$current"
    verify_snapshot "${chain[0]}" "$current"
    for ((index = 1; index < chain_length; index++)); do
        next="$work/reconstructed-$index.tar"
        zstd -q -d -f --long=31 --patch-from="$current" \
            "${chain[$index]}/snapshot.patch.zst" -o "$next"
        verify_snapshot "${chain[$index]}" "$next"
        rm -f -- "$current"
        current=$next
    done
    mv -- "$current" "$output"
}
restore_checkpoint() {
    local requested=$1
    local output=$2
    local checkpoint
    local checkpoint_root
    local temporary
    local snapshot
    if [[ -d "$requested" ]]; then
        checkpoint=$(cd "$requested" && pwd -P)
    else
        [[ -d "$PROGRESS_ROOT/$requested" ]] || fail "checkpoint does not exist: $requested"
        checkpoint=$(cd "$PROGRESS_ROOT/$requested" && pwd -P)
    fi
    checkpoint_root=${checkpoint%/*}
    if [[ -e "$output" ]]; then
        [[ -d "$output" ]] || fail "restore destination is not a directory: $output"
        [[ -z "$(find "$output" -mindepth 1 -print -quit)" ]] \
            || fail "restore destination must be empty: $output"
    else
        mkdir -p "$output"
    fi
    temporary=$(mktemp -d "$checkpoint_root/.restore.XXXXXX")
    trap cleanup EXIT
    snapshot="$temporary/snapshot.tar"
    reconstruct_tar "$checkpoint" "$snapshot" "$temporary"
    tar -xf "$snapshot" -C "$output"
    cleanup
    temporary=""
    trap - EXIT
    touch -a "$checkpoint_root" "$checkpoint"
    printf 'Checkpoint restored: %s\n' "$checkpoint"
    printf 'Complete submission: %s\n' "$(cd "$output" && pwd -P)"
    if [[ -s "$checkpoint/excluded-paths.txt" ]]; then
        printf 'For local testing, restore only the omitted assets from the original task environment; see %s/excluded-paths.txt\n' "$checkpoint"
    fi
}
label=""
note=""
source_override=""
restore=""
restore_destination=""
while (($#)); do
    case "$1" in
        --label) require_value "$@"; label=$2; shift 2 ;;
        --note) require_value "$@"; note=$2; shift 2 ;;
        --source) require_value "$@"; source_override=$2; shift 2 ;;
        --restore) require_value "$@"; restore=$2; shift 2 ;;
        --destination) require_value "$@"; restore_destination=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done
command -v tar >/dev/null 2>&1 || fail "tar is required"
command -v zstd >/dev/null 2>&1 || fail "zstd is required"
if [[ -n "$restore" ]]; then
    [[ -z "$label" && -z "$note" && -z "$source_override" ]] || fail "restore cannot be combined with save options"
    [[ -n "$restore_destination" ]] || fail "--destination is required with --restore"
    [[ -d "$restore" || -d "$PROGRESS_ROOT" ]] \
        || fail "checkpoint root does not exist: $PROGRESS_ROOT"
    restore_checkpoint "$restore" "$restore_destination"
    exit 0
fi
[[ -z "$restore_destination" ]] || fail "--destination requires --restore"
snapshot_source=${source_override:-$SOURCE_ROOT}
[[ -d "$snapshot_source" ]] || fail "source root does not exist: $snapshot_source"
[[ -n "$label" ]] || fail "--label is required"
command -v rsync >/dev/null 2>&1 || fail "rsync is required"
mkdir -p "$PROGRESS_ROOT" "$CACHE_ROOT"
script_source=$0
[[ "$script_source" == */* ]] || script_source=$(command -v "$script_source")
[[ -f "$script_source" ]] || fail "cannot locate checkpoint.sh for restore"
temporary=$(mktemp "$PROGRESS_ROOT/.checkpoint-script.XXXXXX")
trap cleanup EXIT
cp -- "$script_source" "$temporary"
chmod 0555 "$temporary"
mv -f -- "$temporary" "$PROGRESS_ROOT/checkpoint.sh"
temporary=""
trap - EXIT
safe_label=$(printf '%s' "$label" | LC_ALL=C tr -cs 'A-Za-z0-9._-' '-' | cut -c1-64)
safe_label=${safe_label#-}
safe_label=${safe_label%-}
safe_label=${safe_label:-checkpoint}
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
checkpoint_id="$(date -u +%Y%m%dT%H%M%SZ)-$safe_label-$$"
destination="$PROGRESS_ROOT/$checkpoint_id"
[[ ! -e "$destination" ]] || fail "checkpoint already exists: $destination"
previous=""
previous_sequence=-1
for candidate in "$PROGRESS_ROOT"/*; do
    [[ -f "$candidate/metadata.txt" ]] || continue
    sequence=$(metadata_value sequence "$candidate/metadata.txt")
    [[ "$sequence" =~ ^[0-9]+$ ]] || continue
    if ((sequence > previous_sequence)); then
        previous=$candidate
        previous_sequence=$sequence
    fi
done
sequence=$((previous_sequence + 1))
kind=patch
((sequence % KEYFRAME_INTERVAL)) || kind=keyframe
temporary=$(mktemp -d "$PROGRESS_ROOT/.checkpoint.XXXXXX")
trap cleanup EXIT
snapshot="$temporary/final_app"
staged_checkpoint="$temporary/checkpoint"
current_tar="$temporary/current.tar"
mkdir "$snapshot" "$staged_checkpoint"
rsync --archive --hard-links \
    --exclude=.git --exclude=build --exclude=__pycache__ \
    --exclude='*.pyc' --exclude='*.so' --exclude=.pytest_cache \
    --exclude=.ruff_cache --exclude='*.egg-info' \
    --exclude-from=<(printf '%s\n' "$EXCLUDED_PATHS") \
    -- "$snapshot_source/" "$snapshot/"
printf '%s' "$EXCLUDED_PATHS" > "$staged_checkpoint/excluded-paths.txt"
tar -cf "$current_tar" -C "$snapshot" .
snapshot_sha256=$(sha256_file "$current_tar")
snapshot_bytes=$(file_size "$current_tar")
parent_id=""
parent_sha256=""
if [[ -n "$previous" ]]; then
    parent_id=${previous##*/}
fi
if [[ "$kind" == keyframe ]]; then
    payload=snapshot.tar.zst
    zstd -q -3 -T0 -f "$current_tar" -o "$staged_checkpoint/$payload"
else
    parent_sha256=$(metadata_value snapshot_sha256 "$previous/metadata.txt")
    previous_tar="$CACHE_ROOT/latest.tar"
    cached_id=$(test -f "$CACHE_ROOT/checkpoint_id" && cat "$CACHE_ROOT/checkpoint_id" || true)
    if [[ "$cached_id" != "$parent_id" ]] \
        || [[ ! -f "$previous_tar" ]] \
        || [[ "$(sha256_file "$previous_tar")" != "$parent_sha256" ]]; then
        reconstruct_tar "$previous" "$previous_tar" "$temporary"
    fi
    payload=snapshot.patch.zst
    zstd -q -3 -T0 -f --patch-from="$previous_tar" \
        "$current_tar" -o "$staged_checkpoint/$payload"
fi
{
    printf 'schema_version=4\n'
    printf 'checkpoint_id=%s\n' "$checkpoint_id"
    printf 'sequence=%s\n' "$sequence"
    printf 'created_at=%s\n' "$created_at"
    printf 'label=%s\n' "$safe_label"
    printf 'kind=%s\n' "$kind"
    printf 'payload=%s\n' "$payload"
    printf 'parent_checkpoint_id=%s\n' "$parent_id"
    printf 'parent_snapshot_sha256=%s\n' "$parent_sha256"
    printf 'snapshot_sha256=%s\n' "$snapshot_sha256"
    printf 'snapshot_bytes=%s\n' "$snapshot_bytes"
    printf 'artifact_destination=final_app\n'
    printf 'storage=zstd-patch-chain-50\n'
} > "$staged_checkpoint/metadata.txt"
printf '%s\n' "$note" > "$staged_checkpoint/notes.md"
printf '[{"source":"%s","destination":"artifacts/final_app","type":"%s","status":"ok","service":null}]\n' \
    "$payload" "zstd-$kind" > "$staged_checkpoint/manifest.json"
mv "$staged_checkpoint" "$destination"
chmod -R a-w "$destination"
mv -f -- "$current_tar" "$CACHE_ROOT/latest.tar"
printf '%s\n' "$checkpoint_id" > "$CACHE_ROOT/checkpoint_id.tmp"
mv -f -- "$CACHE_ROOT/checkpoint_id.tmp" "$CACHE_ROOT/checkpoint_id"
cleanup
temporary=""
trap - EXIT
touch -a "$PROGRESS_ROOT" "$destination"
printf 'Checkpoint saved: %s\n' "$destination"
printf 'Restore with: %s/checkpoint.sh --restore %s --destination final_app\n' \
    "$PROGRESS_ROOT" "$checkpoint_id"
