"""Evaluator-owned example catalog and safe array interchange."""

import json
import math
import os
from pathlib import Path
import stat

import numpy as np
import torch

STATE_SHAPE = (5, 17, 17)
MAX_EXAMPLES = 70_000
MAX_BYTES = 430_000_000
MAX_SELECTION_BYTES = 8_000_000


class CandidateError(Exception):
    """A submission-caused contract or execution failure."""


def read_regular_file(path):
    """Read bounded data, not a symlink, pipe, device, or Python object."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise CandidateError("candidate file must be a readable regular file") from error
    with os.fdopen(fd, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_SELECTION_BYTES:
            raise CandidateError("candidate file violates file or size limits")
        payload = stream.read(MAX_SELECTION_BYTES + 1)
    if len(payload) > MAX_SELECTION_BYTES:
        raise CandidateError("candidate file exceeds the byte limit")
    return payload


def read_selections(path):
    payload = read_regular_file(path)
    try:
        return json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CandidateError("candidate selections must be valid JSON") from error


def _state_key(state):
    return state.numpy().tobytes()


def build_catalog(groups):
    """Build immutable lookup data and solver-facing grouped catalog records."""
    lookup = {}
    public_groups = []
    for group_index, group in enumerate(groups):
        recovery = {}
        for state, action in group["recovery_examples"]:
            recovery.setdefault(_state_key(state), set()).add(action)
        sources = [
            ("canonical", group["canonical_examples"], False),
            ("alternative", group["alternative_examples"], False),
            ("recovery", group["recovery_examples"], False),
        ]
        failure_prefix, failure_error = [], []
        for example in group["failure_examples"]:
            targets = recovery.get(_state_key(example[0]))
            # Once the goal is unreachable, no next action can correct the failure.
            if not targets:
                continue
            target = failure_prefix if example[1] in targets else failure_error
            target.append(example)
        sources.extend((("correct_prefix", failure_prefix, False),
                        ("failure_error", failure_error, True)))
        records = []
        for kind, examples, negative in sources:
            for item_index, (state, action) in enumerate(examples):
                catalog_id = f"g{group_index}:{kind}:{item_index}"
                frozen_state = state.detach().cpu().numpy().astype(np.float32, copy=True)
                lookup[catalog_id] = (frozen_state, action, negative)
                records.append({"id": catalog_id, "kind": kind, "state": state.clone(),
                                "action": action, "is_negative": negative})
        public_groups.append({key: value for key, value in group.items()
                              if not key.endswith("_examples")}
                             | {"catalog": records})
    return lookup, public_groups


def validate_selections(selections, catalog):
    if not isinstance(selections, (list, tuple)) or not selections:
        raise CandidateError("build_examples must return a nonempty list or tuple")
    if len(selections) > MAX_EXAMPLES:
        raise CandidateError(f"candidate returned more than {MAX_EXAMPLES} selections")
    states, actions, weights, negatives = [], [], [], []
    normalizer = 0.0
    for index, selection in enumerate(selections):
        if not isinstance(selection, (list, tuple)) or len(selection) != 3:
            raise CandidateError(f"selection {index} must be (catalog_id, weight, is_negative)")
        catalog_id, weight, negative = selection
        if type(catalog_id) is not str or catalog_id not in catalog:
            raise CandidateError(f"selection {index} has an unknown catalog ID")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise CandidateError(f"selection {index} weight must be numeric")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise CandidateError(f"selection {index} weight must be finite and nonnegative")
        if type(negative) is not bool:
            raise CandidateError(f"selection {index} negative flag must be Boolean")
        state, action, permitted_negative = catalog[catalog_id]
        if negative != permitted_negative:
            raise CandidateError(f"selection {index} uses an impermissible treatment")
        states.append(state)
        actions.append(action)
        weights.append(weight)
        negatives.append(negative)
        normalizer += weight
    if normalizer <= 0:
        raise CandidateError("training example weight normalizer must be positive")
    arrays = (np.stack(states), np.asarray(actions, dtype=np.int64),
              np.asarray(weights, dtype=np.float32), np.asarray(negatives, dtype=np.bool_))
    if sum(array.nbytes for array in arrays) > MAX_BYTES:
        raise CandidateError(f"candidate arrays exceed the {MAX_BYTES}-byte limit")
    return arrays


def save_selections(path, selections, catalog):
    states, actions, weights, negatives = validate_selections(selections, catalog)
    np.savez(path, states=states, actions=actions, weights=weights, negatives=negatives)


def load_examples(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = tuple(archive[name] for name in ("states", "actions", "weights", "negatives"))
    states, actions, weights, negatives = arrays
    count = len(actions)
    if (states.shape != (count, *STATE_SHAPE) or states.dtype != np.float32
            or actions.shape != (count,) or actions.dtype != np.int64
            or weights.shape != (count,) or weights.dtype != np.float32
            or negatives.shape != (count,) or negatives.dtype != np.bool_):
        raise CandidateError("serialized candidate arrays violate the schema")
    if count == 0 or count > MAX_EXAMPLES or sum(array.nbytes for array in arrays) > MAX_BYTES:
        raise CandidateError("serialized candidate arrays violate size limits")
    if not np.isfinite(states).all() or not np.isfinite(weights).all():
        raise CandidateError("serialized candidate arrays contain non-finite values")
    if (actions < 0).any() or (actions > 3).any() or (weights < 0).any() or weights.sum() <= 0:
        raise CandidateError("serialized candidate targets or weights are invalid")
    return arrays
