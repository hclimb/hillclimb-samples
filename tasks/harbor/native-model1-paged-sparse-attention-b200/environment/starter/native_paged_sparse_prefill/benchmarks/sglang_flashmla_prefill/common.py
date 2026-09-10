from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[2]
SGLANG_ROOT = ROOT / "upstreams" / "sglang"
FLASHMLA_ROOT = ROOT / "upstreams" / "FlashMLA"
VALIDATION_SCRIPT = ROOT / "validation" / "verify_sglang_flashmla_prefill_structure.py"


@dataclass(frozen=True)
class Workload:
    prompt_length: int
    batch_size: int
    topk: int = 512
    swa_window: int = 128
    repetitions: int = 3

    @property
    def num_prompts(self) -> int:
        return self.batch_size * self.repetitions

    def to_json(self) -> dict:
        return asdict(self) | {"num_prompts": self.num_prompts}


DEFAULT_WORKLOADS: List[Workload] = [
    Workload(prompt_length=512, batch_size=1),
    Workload(prompt_length=4096, batch_size=4),
    Workload(prompt_length=16384, batch_size=8),
    Workload(prompt_length=65536, batch_size=8),
]


def parse_row_tiles(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    allowed = {1, 2, 4, 8}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"row tiles must be in {sorted(allowed)}, got {invalid}")
    return values


def parse_prompt_lengths(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    known = {workload.prompt_length: workload for workload in DEFAULT_WORKLOADS}
    missing = [value for value in values if value not in known]
    if missing:
        raise ValueError(f"unknown prompt lengths {missing}; known={sorted(known)}")
    return values


def select_workloads(prompt_lengths: Iterable[int]) -> List[Workload]:
    selected = set(prompt_lengths)
    return [
        workload
        for workload in DEFAULT_WORKLOADS
        if workload.prompt_length in selected
    ]
