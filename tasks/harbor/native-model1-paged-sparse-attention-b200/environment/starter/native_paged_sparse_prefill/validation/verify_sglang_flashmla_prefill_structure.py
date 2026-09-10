#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SGLANG_BACKEND = (
    ROOT
    / "upstreams"
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "deepseek_v4_backend.py"
)
FLASHMLA_SETUP = ROOT / "upstreams" / "FlashMLA" / "setup.py"
FLASHMLA_API = ROOT / "upstreams" / "FlashMLA" / "csrc" / "api" / "api.cpp"
BENCH_DIR = ROOT / "benchmarks" / "sglang_flashmla_prefill"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing expected file: {path}")
    return path.read_text()


def main() -> None:
    sglang_backend = _read(SGLANG_BACKEND)
    flashmla_setup = _read(FLASHMLA_SETUP)
    flashmla_api = _read(FLASHMLA_API)

    required_sglang_markers = [
        "SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_M",
        "_pack_flashmla_prefill_rows",
        "q_flashmla",
        "packed_swa_topk_lengths",
        "flash_mla.flash_mla_with_kvcache",
    ]
    for marker in required_sglang_markers:
        if marker not in sglang_backend:
            raise AssertionError(f"SGLang backend missing marker: {marker}")

    removed_native_markers = [
        "flash_mla_paged_sparse_prefill",
        "paged_sparse_prefill_fwd",
        "model1_paged_sparse_prefill",
        "SGLANG_DSV4_NATIVE_PAGED_PREFILL",
        "VLLM_DSV4_NATIVE_PAGED_PREFILL",
    ]
    combined = "\n".join([sglang_backend, flashmla_setup, flashmla_api])
    for marker in removed_native_markers:
        if marker in combined:
            raise AssertionError(f"stale native paged-prefill marker present: {marker}")

    required_benchmark_files = [
        BENCH_DIR / "common.py",
        BENCH_DIR / "run_prefill_matrix.py",
        BENCH_DIR / "compare_results.py",
    ]
    for path in required_benchmark_files:
        _read(path)

    common = _read(BENCH_DIR / "common.py")
    runner = _read(BENCH_DIR / "run_prefill_matrix.py")
    for marker in [
        "Workload(prompt_length=512, batch_size=1)",
        "Workload(prompt_length=4096, batch_size=4)",
        "Workload(prompt_length=16384, batch_size=8)",
        "Workload(prompt_length=65536, batch_size=8)",
        "SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_M",
        "SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_REQUIRED",
    ]:
        if marker not in common + runner:
            raise AssertionError(f"benchmark harness missing marker: {marker}")

    print("SGLang FlashMLA prefill structure ok")


if __name__ == "__main__":
    main()
