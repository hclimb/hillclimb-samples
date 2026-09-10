"""Decision-aware maze SFT example selection.

Selection strategy (see README for the catalog contract):

* Every training example is a complete-grid state plus an action.  Because
  decoding masks walls and revisited cells, the only decisions that influence
  a rollout are taken at *junctions* (states with two or more legal moves).
  Evaluation failures are dominated by "fatal" junction choices: entering a
  pocket from which the goal is unreachable without a revisit (timeout, quality
  zero).  Longer-but-safe branches only cost ~1/32 quality per extra step.

* We therefore keep the canonical shortest-path examples (they carry the bulk
  of the "head toward the goal" signal, and the forced corridor states help
  the small policy learn quickly) and add the oracle recovery examples, which
  supply optimal actions on the states actually visited by the initial policy
  when it went wrong.  Both kinds are re-weighted by what the decision is
  worth:

    - forced (single legal move):            weight 1.0
    - tie (all legal moves are optimal):     weight 0.5  (label is arbitrary)
    - regret (a longer safe branch exists):  weight 1.0
    - fatal (a legal move loses the goal):   3 copies at weight 1.0

  Fatal-junction states are emphasised by duplication rather than by a x3
  weight: the per-batch weight normalisation then never lets a single hard
  example dominate a batch, which gave smoother late-training quality.

* Alternative (longer) sampled paths and `failure_error` negatives are not
  used: in the fixed 600-step budget they dilute learning on the canonical-like
  states that the trained policy actually visits (measured with a
  teacher-forced junction diagnostic on held-out panels).

Only catalog IDs from the supplied groups are returned, treatments are never
changed, and the total is capped below the evaluator's example limit.
"""
from collections import deque

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SIZE = 17
MAX_TOTAL = 69_000

WEIGHTS = {
    "canonical": 1.0,   # base multiplier per record kind (0 disables the kind)
    "recovery": 1.0,
    "correct_prefix": 0.0,
    "failure_error": 0.0,
    "alternative": 0.0,
}
DECISION_WEIGHTS = {"forced": 1.0, "tie": 0.5, "regret": 1.0, "fatal": 1.0}
DECISION_DUPS = {"forced": 1, "tie": 1, "regret": 1, "fatal": 3}
DEDUPE = True          # drop repeated (state, action) pairs within a maze


def _as_array(state):
    if hasattr(state, "detach"):
        return state.detach().cpu().numpy()
    return np.asarray(state)


def _position(state):
    index = int(np.argmax(state[2].reshape(-1)))
    return divmod(index, SIZE)


def _distances(grid, goal, blocked):
    """BFS distances to the goal over open cells that are not blocked."""
    dist = np.full((SIZE, SIZE), -1, dtype=np.int32)
    if grid[goal[0], goal[1]] or blocked[goal[0], goal[1]]:
        return dist
    dist[goal[0], goal[1]] = 0
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        step = dist[row, col] + 1
        for dr, dc in ACTIONS:
            nr, nc = row + dr, col + dc
            if (0 <= nr < SIZE and 0 <= nc < SIZE and not grid[nr, nc]
                    and not blocked[nr, nc] and dist[nr, nc] < 0):
                dist[nr, nc] = step
                queue.append((nr, nc))
    return dist


def classify(goal, state):
    """Classify a state as forced / tie / regret / fatal / dead.

    Returns (category, optimal_actions).  Optimal actions lead along a shortest
    remaining route to the goal that never revisits a cell (the same rule the
    oracle recovery labels follow)."""
    grid = state[0] > 0.5
    visited = state[3] > 0.5
    pos = _position(state)
    legal = [a for a, (dr, dc) in enumerate(ACTIONS)
             if not grid[pos[0] + dr, pos[1] + dc] and not visited[pos[0] + dr, pos[1] + dc]]
    if not legal:
        return "dead", []
    if len(legal) == 1:
        return "forced", legal
    blocked = visited.copy()
    blocked[pos[0], pos[1]] = True
    dist = _distances(grid, goal, blocked)
    remaining = {a: int(dist[pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1]]) for a in legal}
    reachable = [a for a in legal if remaining[a] >= 0]
    if not reachable:
        return "dead", []
    best = min(remaining[a] for a in reachable)
    optimal = [a for a in reachable if remaining[a] == best]
    if len(reachable) < len(legal):
        return "fatal", optimal
    if len(optimal) == len(legal):
        return "tie", optimal
    return "regret", optimal


def build_examples(train_groups):
    """Return (catalog_id, weight, is_negative) selections."""
    selections = []
    for group in train_groups:
        goal = tuple(group["goal"])
        seen = set()
        for record in group["catalog"]:
            kind = record["kind"]
            base = WEIGHTS.get(kind, 0.0)
            if base <= 0:
                continue
            negative = bool(record["is_negative"])
            if negative:
                selections.append((record["id"], float(base), True))
                continue
            state = _as_array(record["state"])
            category, optimal = classify(goal, state)
            if category == "dead" or (category != "forced" and record["action"] not in optimal):
                continue  # only teach optimal moves
            weight = base * DECISION_WEIGHTS[category]
            if weight <= 0:
                continue
            if DEDUPE:
                key = (state.tobytes(), int(record["action"]))
                if key in seen:
                    continue
                seen.add(key)
            selections.extend([(record["id"], float(weight), False)] * DECISION_DUPS[category])
    if len(selections) > MAX_TOTAL:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(selections), MAX_TOTAL, replace=False))
        selections = [selections[i] for i in keep]
    return selections
