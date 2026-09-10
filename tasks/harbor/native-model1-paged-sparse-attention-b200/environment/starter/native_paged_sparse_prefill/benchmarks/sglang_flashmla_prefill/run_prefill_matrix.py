#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from common import DEFAULT_WORKLOADS, FLASHMLA_ROOT, ROOT, SGLANG_ROOT
from common import parse_prompt_lengths, parse_row_tiles, select_workloads


def _split_extra(raw: str) -> List[str]:
    return [item for item in raw.split() if item]


def _run(
    cmd: List[str],
    *,
    env: dict,
    cwd: Path,
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed with rc={proc.returncode}: {' '.join(cmd)}\n"
            f"log: {log_path}"
        )
    return proc


def _pythonpath(env: dict) -> str:
    paths = [
        str(SGLANG_ROOT / "python"),
        str(SGLANG_ROOT / "sgl-kernel" / "python"),
        str(FLASHMLA_ROOT),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def _preflight(args: argparse.Namespace, env: dict, output_dir: Path) -> None:
    preflight_dir = output_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [args.python, str(ROOT / "validation" / "verify_sglang_flashmla_prefill_structure.py")],
        env=env,
        cwd=ROOT,
        log_path=preflight_dir / "structure.log",
    )
    _run(
        [args.python, "-m", "py_compile", str(SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "attention" / "deepseek_v4_backend.py")],
        env=env,
        cwd=ROOT,
        log_path=preflight_dir / "py_compile.log",
    )
    _run(
        ["nvidia-smi", "-L"],
        env=env,
        cwd=ROOT,
        log_path=preflight_dir / "nvidia_smi_L.log",
    )
    cuda_probe = (
        "import json, torch\n"
        "info=[]\n"
        "for i in range(torch.cuda.device_count()):\n"
        "    p=torch.cuda.get_device_properties(i)\n"
        "    info.append({'idx':i,'name':p.name,'capability':torch.cuda.get_device_capability(i),'total_memory':p.total_memory})\n"
        "print(json.dumps(info, indent=2))\n"
        f"assert torch.cuda.device_count() >= {args.tensor_parallel_size}\n"
    )
    _run(
        [args.python, "-c", cuda_probe],
        env=env,
        cwd=ROOT,
        log_path=preflight_dir / "torch_cuda.json",
    )


def _wait_for_health(base_url: str, proc: subprocess.Popen[str], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[str] = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with rc={proc.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"server did not become healthy at {base_url}: {last_error}")


def _terminate(proc: subprocess.Popen[str], timeout_s: int = 60) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _launch_server(
    args: argparse.Namespace,
    *,
    row_tile_m: int,
    env: dict,
    output_dir: Path,
) -> subprocess.Popen[str]:
    server_log = output_dir / f"server_m{row_tile_m}.log"
    server_env = env.copy()
    server_env["SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_M"] = str(row_tile_m)
    if row_tile_m != 1 and not args.allow_row_pack_fallback:
        server_env["SGLANG_DSV4_FLASHMLA_PREFILL_ROW_TILE_REQUIRED"] = "1"

    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--ep-size",
        str(args.ep_size),
        "--trust-remote-code",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ] + _split_extra(args.server_args)

    server_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = server_log.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=SGLANG_ROOT,
        env=server_env,
        text=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._sglang_log_fh = log_fh  # type: ignore[attr-defined]
    try:
        _wait_for_health(f"http://{args.host}:{args.port}", proc, args.server_timeout_s)
    except Exception:
        _terminate(proc)
        log_fh.close()
        raise
    return proc


def _close_server(proc: subprocess.Popen[str]) -> None:
    try:
        _terminate(proc)
    finally:
        log_fh = getattr(proc, "_sglang_log_fh", None)
        if log_fh is not None:
            log_fh.close()


def _run_workload(
    args: argparse.Namespace,
    *,
    row_tile_m: int,
    workload,
    env: dict,
    output_dir: Path,
) -> dict:
    result_file = output_dir / f"bench_m{row_tile_m}_{workload.prompt_length}.jsonl"
    log_file = output_dir / f"bench_m{row_tile_m}_{workload.prompt_length}.log"
    cmd = [
        args.python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model_path,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.prompt_length),
        "--random-output-len",
        str(args.random_output_len),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(workload.num_prompts),
        "--max-concurrency",
        str(workload.batch_size),
        "--disable-tqdm",
        "--output-file",
        str(result_file),
    ] + _split_extra(args.bench_args)
    _run(cmd, env=env, cwd=SGLANG_ROOT, log_path=log_file)
    rows = [json.loads(line) for line in result_file.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"benchmark produced no JSON rows: {result_file}")
    row = rows[-1]
    row["row_tile_m"] = row_tile_m
    row["workload"] = workload.to_json()
    row["result_file"] = str(result_file)
    row["log_file"] = str(log_file)
    return row


def _write_summary(rows: Iterable[dict], output_dir: Path) -> None:
    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--chunked-prefill-size", type=int, default=32768)
    parser.add_argument("--max-prefill-tokens", type=int, default=65536)
    parser.add_argument("--context-length", type=int, default=66000)
    parser.add_argument("--mem-fraction-static", default="0.85")
    parser.add_argument("--random-output-len", type=int, default=1)
    parser.add_argument("--row-tiles", default="1,2,4,8")
    parser.add_argument(
        "--prompt-lengths",
        default=",".join(str(w.prompt_length) for w in DEFAULT_WORKLOADS),
    )
    parser.add_argument("--server-timeout-s", type=int, default=1800)
    parser.add_argument("--server-args", default="")
    parser.add_argument("--bench-args", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--allow-row-pack-fallback",
        action="store_true",
        help="Do not fail if a row-packed server batch is not packable.",
    )
    args = parser.parse_args()

    row_tiles = parse_row_tiles(args.row_tiles)
    prompt_lengths = parse_prompt_lengths(args.prompt_lengths)
    workloads = select_workloads(prompt_lengths)
    output_dir = Path(args.output_dir or ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(env)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest = {
        "model_path": args.model_path,
        "row_tiles": row_tiles,
        "workloads": [workload.to_json() for workload in workloads],
        "args": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not args.skip_preflight:
        _preflight(args, env, output_dir)

    rows: List[dict] = []
    for row_tile_m in row_tiles:
        proc = _launch_server(args, row_tile_m=row_tile_m, env=env, output_dir=output_dir)
        try:
            for workload in workloads:
                rows.append(
                    _run_workload(
                        args,
                        row_tile_m=row_tile_m,
                        workload=workload,
                        env=env,
                        output_dir=output_dir,
                    )
                )
                _write_summary(rows, output_dir)
        finally:
            _close_server(proc)

    _write_summary(rows, output_dir)
    print(f"Wrote benchmark summary to {output_dir / 'summary.jsonl'}")


if __name__ == "__main__":
    main()
