"""Deterministic, model-independent Dolci-Instruct 32K data artifacts.

Heavy builder/conversion imports are lazy so profile discovery and launcher
validation do not require the training stack (notably torch).
"""

from .profile import PROFILE_NAME, PROFILE_VERSION, SETTINGS

__all__ = [
    "Dolci32KBuilder",
    "PROFILE_NAME",
    "PROFILE_VERSION",
    "SETTINGS",
    "artifact_path",
    "candidate_order_path",
    "reaudit_build",
    "resolve_current_build",
]


def __getattr__(name):
    if name in {"Dolci32KBuilder", "reaudit_build"}:
        from .builder import Dolci32KBuilder, reaudit_build
        return {"Dolci32KBuilder": Dolci32KBuilder, "reaudit_build": reaudit_build}[name]
    if name in {"artifact_path", "candidate_order_path", "resolve_current_build"}:
        from .artifacts import artifact_path, candidate_order_path, resolve_current_build
        return {
            "artifact_path": artifact_path,
            "candidate_order_path": candidate_order_path,
            "resolve_current_build": resolve_current_build,
        }[name]
    raise AttributeError(name)
