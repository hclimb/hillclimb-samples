"""
Task-specific evaluation modules.

Each module provides a compute_accuracy function that evaluates
a model on that specific task.

Attribute access is lazy (PEP 562): importing this package used to eagerly pull
in `samsum`, which loads the ROUGE metric — an expensive, network-touching
side effect for callers that only wanted, say, the IFEval verifier during data
preparation. The public names below are unchanged.
"""

import importlib
from typing import Any

_LAZY_ATTRS = {
    "compute_samsum_accuracy": ("samsum", "compute_accuracy"),
    "compute_tydiqa_accuracy": ("tydiqa", "compute_accuracy"),
    "compute_nq_open_accuracy": ("nq_open", "compute_accuracy"),
    "compute_squad_accuracy": ("squad", "compute_accuracy"),
    "compute_ifeval_accuracy": ("ifeval", "compute_accuracy"),
    "compute_ifbench_accuracy": ("ifbench", "compute_accuracy"),
    "compute_math500_accuracy": ("math500", "compute_accuracy"),
    "compute_mbpp_plus_accuracy": ("mbpp_plus", "compute_accuracy"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
