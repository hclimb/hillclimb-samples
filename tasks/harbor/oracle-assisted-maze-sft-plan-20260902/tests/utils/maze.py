"""Offline real-maze fixtures, trajectory derivation, and scoring."""

from collections import deque
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import torch

SIZE = 17
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DATASET_TO_MODEL = {0: 0, 1: 2, 2: 3, 3: 1}
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INITIAL_CHECKPOINT_SHA256 = "1e1fa297299b788176b302cd108a4e201cff11f9bd57fe312aeeb1cf757afbe4"


def distance_map(grid, goal, blocked=frozenset()):
    if goal in blocked:
        return {}
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ACTIONS:
            point = row + dr, col + dc
            if not grid[point[0]][point[1]] and point not in blocked and point not in distances:
                distances[point] = distances[(row, col)] + 1
                queue.append(point)
    return distances


def shortest_actions(maze, start=None, tie_order=(0, 1, 2, 3)):
    point = maze["start"] if start is None else start
    distances = distance_map(maze["grid"], maze["goal"])
    actions = []
    while point != maze["goal"]:
        options = []
        for action in tie_order:
            dr, dc = ACTIONS[action]
            nxt = point[0] + dr, point[1] + dc
            if nxt in distances and distances[nxt] < distances[point]:
                options.append((distances[nxt], action, nxt))
        _, action, point = min(options)
        actions.append(action)
    return actions


def encode(maze, point, visited):
    value = torch.zeros(5, SIZE, SIZE, dtype=torch.float32)
    value[0] = torch.tensor(maze["grid"], dtype=torch.float32)
    value[1, maze["goal"][0], maze["goal"][1]] = 1
    value[2, point[0], point[1]] = 1
    for row, col in visited:
        value[3, row, col] = 1
    value[4, maze["start"][0], maze["start"][1]] = 1
    return value


def states_for(maze, actions):
    point, visited, examples = maze["start"], {maze["start"]}, []
    for action in actions:
        examples.append((encode(maze, point, visited), action))
        dr, dc = ACTIONS[action]
        point = point[0] + dr, point[1] + dc
        visited.add(point)
    return examples


def failed_examples(maze, actions):
    point, visited, examples = maze["start"], {maze["start"]}, []
    for action in actions:
        examples.append((encode(maze, point, visited), action))
        dr, dc = ACTIONS[action]
        nxt = point[0] + dr, point[1] + dc
        if maze["grid"][nxt[0]][nxt[1]]:
            break
        point = nxt
        visited.add(point)
    return examples


def frozen_failure(maze, policy):
    point, visited, actions = maze["start"], {maze["start"]}, []
    for _ in range(60):
        state = encode(maze, point, visited).unsqueeze(0)
        logits = policy(state)[0].clone()
        for action, (dr, dc) in enumerate(ACTIONS):
            nxt = point[0] + dr, point[1] + dc
            if maze["grid"][nxt[0]][nxt[1]] or nxt in visited:
                logits[action] = -torch.inf
        if not torch.isfinite(logits).any():
            break
        action = int(logits.argmax().item())
        actions.append(action)
        dr, dc = ACTIONS[action]
        nxt = point[0] + dr, point[1] + dc
        if maze["grid"][nxt[0]][nxt[1]]:
            break
        point = nxt
        visited.add(point)
        if point == maze["goal"]:
            break
    return actions


def recovery_examples(maze, failed_actions):
    point, visited, examples = maze["start"], {maze["start"]}, []
    for action in failed_actions:
        distances = distance_map(maze["grid"], maze["goal"], visited - {point})
        for target, (dr, dc) in enumerate(ACTIONS):
            nxt = point[0] + dr, point[1] + dc
            if point in distances and nxt in distances and distances[nxt] < distances[point]:
                examples.append((encode(maze, point, visited), target))
        dr, dc = ACTIONS[action]
        nxt = point[0] + dr, point[1] + dc
        if maze["grid"][nxt[0]][nxt[1]]:
            break
        point = nxt
        visited.add(point)
    return examples


def _validate_path(maze, actions):
    point = maze["start"]
    seen = {point}
    for action in actions:
        if action not in range(4):
            raise RuntimeError("fixture contains an invalid action")
        dr, dc = ACTIONS[action]
        point = point[0] + dr, point[1] + dc
        if maze["grid"][point[0]][point[1]] or point in seen:
            raise RuntimeError("fixture contains an invalid supplied path")
        seen.add(point)
    if point != maze["goal"]:
        raise RuntimeError("fixture path does not reach the goal")


def _group(row, failure=None, *, include_catalog=True):
    required = {"prompt_id", "grid", "L_star", "ub", "n_samples", "samples"}
    if not required <= row.keys() or len(row["grid"]) != SIZE or any(len(line) != SIZE for line in row["grid"]):
        raise RuntimeError("real-data fixture row violates the upstream schema")
    maze = {**row, "start": (1, 1), "goal": (15, 15), "seed": row["prompt_id"]}
    canonical = shortest_actions(maze)
    if len(canonical) != row["L_star"] or row["ub"] != 60 or row["n_samples"] != len(row["samples"]):
        raise RuntimeError("real-data fixture metadata disagrees with evaluator BFS")
    alternatives = []
    for sample in row["samples"]:
        actions = [DATASET_TO_MODEL[action] for action in sample["actions"]]
        if len(actions) != sample["L"]:
            raise RuntimeError("fixture sample length is inconsistent")
        _validate_path(maze, actions)
        alternatives.append(actions)
    failure = [] if failure is None else failure
    group = {**maze, "canonical": canonical, "alternatives": alternatives,
             "failure": failure, "canonical_examples": states_for(maze, canonical)}
    if include_catalog:
        group.update(failure_examples=failed_examples(maze, failure),
                     recovery_examples=recovery_examples(maze, failure),
                     alternative_examples=sum((states_for(maze, path) for path in alternatives), []))
    return group


@lru_cache(maxsize=1)
def fixture_manifest():
    return json.loads((FIXTURES / "manifest.json").read_text())


@lru_cache(maxsize=1)
def frozen_failures():
    metadata = fixture_manifest()["frozen_failures"]
    payload = (FIXTURES / metadata["file"]).read_bytes()
    if len(payload) != metadata["compressed_bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise RuntimeError("frozen baseline failures failed authentication")
    raw = gzip.decompress(payload)
    if len(raw) != metadata["uncompressed_bytes"] or hashlib.sha256(raw).hexdigest() != metadata["uncompressed_sha256"]:
        raise RuntimeError("frozen baseline failure content failed authentication")
    data = json.loads(raw)
    if data["checkpoint_sha256"] != INITIAL_CHECKPOINT_SHA256:
        raise RuntimeError("frozen failures do not match the fixed baseline checkpoint")
    actions = {int(prompt_id): value for prompt_id, value in data["actions_by_prompt_id"].items()}
    expected = set(partition_ids("public_train")) | set(partition_ids("private_train"))
    if set(actions) != expected or len(actions) != metadata["rows"]:
        raise RuntimeError("frozen baseline failure prompt manifest disagrees")
    return actions


@lru_cache(maxsize=None)
def _rows(partition):
    metadata = fixture_manifest()["partitions"][partition]
    payload = (FIXTURES / metadata["file"]).read_bytes()
    if len(payload) != metadata["compressed_bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise RuntimeError(f"real-data fixture {partition} failed authentication")
    raw = gzip.decompress(payload)
    if len(raw) != metadata["uncompressed_bytes"] or hashlib.sha256(raw).hexdigest() != metadata["uncompressed_sha256"]:
        raise RuntimeError(f"real-data fixture {partition} content failed authentication")
    rows = tuple(json.loads(line) for line in raw.splitlines())
    if len(rows) != metadata["rows"] or [row["prompt_id"] for row in rows] != metadata["prompt_ids"]:
        raise RuntimeError(f"real-data fixture {partition} selection manifest disagrees")
    return {row["prompt_id"]: row for row in rows}


def partition_ids(partition):
    return tuple(fixture_manifest()["partitions"][partition]["prompt_ids"])


def build_groups(prompt_ids, failure_policy=None, *, include_catalog=True):
    """Load validated mazes; omit candidate-only tensors when not needed."""
    wanted = set(prompt_ids)
    matches = {}
    for partition in ("initialization", "public_train", "public_eval", "private_train", "private_eval"):
        if wanted & set(partition_ids(partition)):
            matches.update(_rows(partition))
    if not wanted <= matches.keys():
        raise RuntimeError("requested prompt ID is outside the packaged real-data partitions")
    packaged = frozen_failures() if wanted & set(frozen_failures().keys()) else {}
    groups = [_group(matches[prompt_id], packaged.get(prompt_id), include_catalog=include_catalog)
              for prompt_id in prompt_ids]
    if failure_policy is not None:
        for group in groups:
            failure = frozen_failure(group, failure_policy)
            group["failure"] = failure
            if include_catalog:
                group.update(failure_examples=failed_examples(group, failure),
                             recovery_examples=recovery_examples(group, failure))
    return groups


def score_path(maze, actions, upper_bound=None):
    point = maze["start"]
    for action in actions:
        if action not in range(4):
            return 0.0, "malformed"
        dr, dc = ACTIONS[action]
        point = point[0] + dr, point[1] + dc
        if maze["grid"][point[0]][point[1]]:
            return 0.0, "collision"
        if point == maze["goal"]:
            bound = maze["ub"] if upper_bound is None else upper_bound
            denominator = bound - maze["L_star"]
            return max(0.0, min(1.0, (bound - len(actions)) / denominator)), "success"
    return 0.0, "timeout"
