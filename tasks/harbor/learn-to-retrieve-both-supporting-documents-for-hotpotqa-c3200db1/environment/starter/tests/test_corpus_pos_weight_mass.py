"""Standalone test for the corpus-level mem_pos_weight_mass in evals/gen_large_mem.py.

The metric is the corpus analog of train/mem_pos_weight_mass (losses/mem_telemetry.py):
    mass = Σ(softmax weight on positive-doc slots) / Σ(softmax weight on valid slots)
measured over the answer span. Here we pin the numpy math against hand-computed values.

Run:  uv run python tests/test_corpus_pos_weight_mass.py
"""
import os
import sys

# `evals` is not an installed package (pyproject: packages = ["models", "data"]), so importing it
# from tests/ needs the repo root on sys.path — same preamble as tests/test_doc_access_acc.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from evals.gen_large_mem import (
    _numpy_pos_weight_mass,
    _numpy_pos_weight_mass_per_example,
    _numpy_pos_weight_mass_layers,
    _numpy_pos_weight_mass_per_example_layers,
)

FAILS = []


def check(name, got, want, tol=1e-6):
    # tol is 1e-6, not ~0: the probs are float32 (that's what mem_top_k_probs is on device), so
    # e.g. 0.7 is really 0.699999988..., and the ratio inherits ~1e-8 error. Anything tighter
    # tests float32's representation, not the metric.
    ok = (got is None and want is None) or (
        got is not None and want is not None and abs(got - want) < tol
    )
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        FAILS.append(name)


# One batch item, one head, two answer positions, top-k=2 over a 4-slot bank.
# Slots 0,1 belong to the positive doc; 2,3 are distractors.
IDX = np.array([[[[0, 2], [1, 3]]]], dtype=np.int64)        # (B=1,H=1,S=2,K=2)
PROB = np.array([[[[0.7, 0.3], [0.6, 0.4]]]], dtype=np.float32)
POS = [{0, 1}]
ALL_VALID = np.ones(4, dtype=np.int32)
BOTH_POS = np.ones((1, 2), dtype=np.float32)

# Both positions active, all slots valid:
#   num = 0.7 (slot0, pos) + 0.6 (slot1, pos) = 1.3 ; den = 0.7+0.3+0.6+0.4 = 2.0
check("base", _numpy_pos_weight_mass(IDX, PROB, POS, BOTH_POS, ALL_VALID), 1.3 / 2.0)

# Answer-span mask must exclude position 1: num = 0.7 ; den = 1.0
check(
    "loss_mask excludes position",
    _numpy_pos_weight_mass(IDX, PROB, POS, np.array([[1.0, 0.0]], dtype=np.float32), ALL_VALID),
    0.7,
)

# Invalid bank slot 3 drops out of BOTH numerator and denominator:
#   num = 0.7+0.6 = 1.3 ; den = 0.7+0.3+0.6 = 1.6
check(
    "invalid slot excluded from denominator",
    _numpy_pos_weight_mass(IDX, PROB, POS, BOTH_POS, np.array([1, 1, 1, 0], dtype=np.int32)),
    1.3 / 1.6,
)

# No positive slots -> 0.0 mass (defined, not None: weight was retrieved, none of it positive).
check("no positives -> 0.0", _numpy_pos_weight_mass(IDX, PROB, [set()], BOTH_POS, ALL_VALID), 0.0)

# Nothing active -> denominator 0 -> undefined (None), so it is dropped from the mean
# rather than poisoning it with a 0.
check(
    "all masked -> None",
    _numpy_pos_weight_mass(IDX, PROB, POS, np.zeros((1, 2), dtype=np.float32), ALL_VALID),
    None,
)

# All-positive retrieval saturates at 1.0 (sanity on the ratio's upper bound).
check(
    "all-positive -> 1.0",
    _numpy_pos_weight_mass(
        np.array([[[[0, 1], [0, 1]]]], dtype=np.int64), PROB, POS, BOTH_POS, ALL_VALID
    ),
    1.0,
)

# --- pooling: two examples pool as Σnum/Σden, NOT the mean of per-example ratios ---
# ex0: num=1.3 den=2.0 ; ex1: num=0.0 den=2.0  -> pooled = 1.3/4.0 = 0.325
IDX2 = np.concatenate([IDX, np.array([[[[2, 3], [2, 3]]]], dtype=np.int64)], axis=0)
PROB2 = np.concatenate([PROB, PROB], axis=0)
POS2 = [{0, 1}, {0, 1}]
MASK2 = np.ones((2, 2), dtype=np.float32)
check("pooled over batch", _numpy_pos_weight_mass(IDX2, PROB2, POS2, MASK2, ALL_VALID), 1.3 / 4.0)

per_ex = _numpy_pos_weight_mass_per_example(IDX2, PROB2, POS2, MASK2, ALL_VALID)
check("per-example [0]", per_ex[0], 1.3 / 2.0)
check("per-example [1]", per_ex[1], 0.0)

per_ex_none = _numpy_pos_weight_mass_per_example(
    IDX, PROB, POS, np.zeros((1, 2), dtype=np.float32), ALL_VALID
)
check("per-example undefined -> None", per_ex_none[0], None)

# --- layer aggregation: mean over layers (matches the train-time metric) ---
# layer A = base (0.65); layer B = all-positive (1.0) -> mean 0.825
IDX_B = np.array([[[[0, 1], [0, 1]]]], dtype=np.int64)
check(
    "layer-mean",
    _numpy_pos_weight_mass_layers([IDX, IDX_B], [PROB, PROB], POS, BOTH_POS, ALL_VALID),
    (1.3 / 2.0 + 1.0) / 2.0,
)
check(
    "layer-mean per-example",
    _numpy_pos_weight_mass_per_example_layers(
        [IDX, IDX_B], [PROB, PROB], POS, BOTH_POS, ALL_VALID
    )[0],
    (1.3 / 2.0 + 1.0) / 2.0,
)

# A layer that is undefined must not drag the layer-mean toward 0.
check(
    "layer-mean skips undefined layer",
    _numpy_pos_weight_mass_layers(
        [IDX, IDX_B], [PROB, PROB], POS, np.array([[0.0, 0.0]], dtype=np.float32), ALL_VALID
    ),
    None,
)

# Single memory layer (qwen3_mem_embed has mem_layers=[14]) -> layer-mean == that layer.
check(
    "single layer == base",
    _numpy_pos_weight_mass_layers([IDX], [PROB], POS, BOTH_POS, ALL_VALID),
    1.3 / 2.0,
)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} -> {FAILS}")
    sys.exit(1)
print("ALL PASS")
