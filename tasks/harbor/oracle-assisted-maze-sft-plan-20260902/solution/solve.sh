#!/bin/sh
set -eu
if [ -n "${WERM_MAZE_FROZEN_CONTROL:-}" ]; then
    case "$WERM_MAZE_FROZEN_CONTROL" in
        sol|fable) ;;
        *) echo "Unknown frozen control: $WERM_MAZE_FROZEN_CONTROL" >&2; exit 1 ;;
    esac
    cp "/solution/frozen/$WERM_MAZE_FROZEN_CONTROL.py" /environment/starter/maze_task/candidate.py
    exit 0
fi
cat > /environment/starter/maze_task/candidate.py <<'PY'
"""Reference method adding one lightly weighted contrastive error per maze."""

def build_examples(train_groups):
    selections = []
    for group in train_groups:
        selections.extend((item["id"], 1.0, False) for item in group["catalog"]
                          if item["kind"] == "canonical")
        error = next((item for item in group["catalog"]
                      if item["kind"] == "failure_error"), None)
        if error is not None:
            selections.append((error["id"], 0.1, True))
    return selections
PY
