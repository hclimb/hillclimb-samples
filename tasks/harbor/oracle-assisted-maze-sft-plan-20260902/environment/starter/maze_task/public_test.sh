#!/bin/sh
set -eu
mode=quick
mode_value_pending=false
for argument in "$@"; do
    if [ "$mode_value_pending" = true ]; then
        mode=$argument
        mode_value_pending=false
    fi
    case "$argument" in
        --help|-h)
            exec python -I -B /environment/starter/optifine_public_tests/verifiers/run_public.py "$@"
            ;;
        --m|--mo|--mod|--mode) mode_value_pending=true ;;
        --m=*|--mo=*|--mod=*|--mode=*) mode=${argument#*=} ;;
    esac
done
if [ "$mode" != full ]; then
    exec python -I -B /environment/starter/optifine_public_tests/verifiers/run_public.py "$@"
fi

# Test the saved input, not a workspace that another process can still edit.
checkpoint_output=$(checkpoint.sh --label public-test --note "Public test input: $*")
printf '%s\n' "$checkpoint_output" >&2
checkpoint=$(printf '%s\n' "$checkpoint_output" | sed -n 's/^Checkpoint saved: //p')
result_dir="${checkpoint%/*}/../public-tests/${checkpoint##*/}"
mkdir -p "$result_dir"
cp "$checkpoint/metadata.txt" "$result_dir/checkpoint.txt"
for argument in "$@"; do
    printf '%s\n' "$argument"
done > "$result_dir/arguments.txt"

test_root=$(mktemp -d /tmp/maze-public-test.XXXXXX)
trap 'rm -rf -- "$test_root"' EXIT
checkpoint.sh --restore "$checkpoint" --destination "$test_root" >&2
printf 'Public test records: %s\n' "$result_dir" >&2
test_status=0
STARTER_ROOT="$test_root" python -I -B \
    /environment/starter/optifine_public_tests/verifiers/run_public.py "$@" \
    > "$result_dir/result.json" 2> "$result_dir/stderr.log" || test_status=$?
printf '%s\n' "$test_status" > "$result_dir/exit-code.txt"
cat "$result_dir/stderr.log" >&2
cat "$result_dir/result.json"
exit "$test_status"
