"""Build compact, decision-focused supervision for the fixed maze policy.

The rollout code already masks walls and visited cells.  Consequently, a label
at a state with only one legal successor cannot affect decoding.  Alternative
paths are most useful at genuine branch points, where they expose states that
are absent from the single canonical path.  We retain an alternative label
only when it follows a shortest route to the goal without revisiting cells.
"""

from collections import deque

import torch


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_ALTERNATIVE_PATHS = 4


def _distance_map(grid, goal, blocked):
    """Return goal distances while treating previously visited cells as walls."""
    if goal in blocked:
        return {}

    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for row_delta, col_delta in _DIRECTIONS:
            neighbor = row + row_delta, col + col_delta
            if (not grid[neighbor[0]][neighbor[1]]
                    and neighbor not in blocked
                    and neighbor not in distances):
                distances[neighbor] = distances[(row, col)] + 1
                queue.append(neighbor)
    return distances


def _is_shortest_branch_action(record, group):
    """Check that a positive label matters to decoding and makes BFS progress."""
    state = record["state"]
    point = tuple(torch.nonzero(state[2], as_tuple=False)[0].tolist())
    visited = {tuple(cell.tolist())
               for cell in torch.nonzero(state[3], as_tuple=False)}

    legal_count = 0
    for row_delta, col_delta in _DIRECTIONS:
        neighbor = point[0] + row_delta, point[1] + col_delta
        if not group["grid"][neighbor[0]][neighbor[1]] and neighbor not in visited:
            legal_count += 1
    if legal_count < 2:
        return False

    distances = _distance_map(group["grid"], group["goal"], visited - {point})
    row_delta, col_delta = _DIRECTIONS[record["action"]]
    successor = point[0] + row_delta, point[1] + col_delta
    return (point in distances
            and successor in distances
            and distances[successor] < distances[point])


def _state_key(record):
    return record["state"].numpy().tobytes()


def build_examples(train_groups):
    """Select canonical, diverse shortest-progress, and targeted recovery labels."""
    selections = []
    for group in train_groups:
        # Alternatives are ordered by path length.  Four gives useful state
        # diversity while allowing several passes within the fixed step budget.
        alternative_cutoff = sum(
            map(len, group["alternatives"][:_ALTERNATIVE_PATHS]))

        # A negative record marks a branch chosen incorrectly by the frozen
        # policy.  Train its evaluator-provided recovery action positively;
        # direct negative losses proved less stable for this small policy.
        mistake_states = {
            _state_key(record)
            for record in group["catalog"]
            if record["kind"] == "failure_error"
        }

        for record in group["catalog"]:
            kind = record["kind"]
            selected = (
                kind == "canonical"
                or (
                    kind == "alternative"
                    and int(record["id"].rsplit(":", 1)[1]) < alternative_cutoff
                    and _is_shortest_branch_action(record, group)
                )
                or (kind == "recovery" and _state_key(record) in mistake_states)
            )
            if selected:
                selections.append((record["id"], 1.0, False))

    return selections
