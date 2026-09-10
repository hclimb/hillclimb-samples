"""MBPP+ generation and sandboxed official EvalPlus pass@1 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from SFT.eval.evalplus_contract import (
    EVALPLUS_CONTAINER_DISTRIBUTION_VERSION,
    EVALPLUS_DATASET_VERSION,
    EVALPLUS_HOST_DISTRIBUTION_VERSION,
    EVALPLUS_IMAGE,
    EVALPLUS_RELEASE_VERSION,
    EVALPLUS_RUNTIME_CACHE_CONTAINER_PATH,
    MBPP_PLUS_CACHE_FILENAME,
    MBPP_PLUS_CONTAINER_PATH,
    MBPP_PLUS_DATASET_SHA256,
)

from SFT.eval.tasks.bench_data import load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    require_distribution_version,
    result_provenance,
    single_source_revision,
)
from ..utils import generate_completions, get_eos_token_ids


DEFAULT_MAX_NEW_TOKENS = 2048
DATASET_REPOSITORY = "evalplus/mbppplus"

_FENCED_PYTHON_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE
)
_FENCED_ANY_RE = re.compile(r"```\s*\n(.*?)(?:```|\Z)", re.DOTALL)
_BEGIN_DONE_RE = re.compile(r"\[BEGIN\](.*?)(?:\[DONE\]|\Z)", re.DOTALL)
_CHAT_MARKERS = ("<|im_end|>", "<|im_start|>", "<|user|>", "<|assistant|>", "<|system|>")


def extract_code(text: str) -> str:
    """Recover a Python program from a free-form generation.

    Tries fenced blocks first (what the target format teaches), then the
    ``[BEGIN]/[DONE]`` convention some MBPP prompts elicit, then falls back to
    the raw text so a model that emits bare code is not penalized for format.
    """
    if not text:
        return ""
    cleaned = text
    for marker in _CHAT_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index]

    for pattern in (_FENCED_PYTHON_RE, _FENCED_ANY_RE, _BEGIN_DONE_RE):
        match = pattern.search(cleaned)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return cleaned.strip()
EVALPLUS_VERSION = EVALPLUS_HOST_DISTRIBUTION_VERSION
MBPP_PLUS_DATASET_VERSION = EVALPLUS_DATASET_VERSION
DEFAULT_EVALPLUS_IMAGE = EVALPLUS_IMAGE
EVALPLUS_IMAGE_DIGEST = DEFAULT_EVALPLUS_IMAGE.rsplit("@", 1)[-1]

_PROMPT_TEMPLATE = """Complete this MBPP+ task. Return one complete, self-contained Python solution including the required function signature. Do not include prose outside the code.

{prompt}"""
_FUNCTION_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(")


def task_id_and_prompt(record: Mapping[str, Any]) -> Tuple[str, str]:
    metadata = record.get("metadata", {})
    task_id = metadata.get("task_id") or record.get("id")
    prompt = metadata.get("prompt")
    if not task_id or not prompt:
        raise ValueError(f"Malformed MBPP+ materialized record: {record.get('id')}")
    return str(task_id), str(prompt)

def select_task_rows(
    records: Sequence[Mapping[str, Any]],
    registry_task_ids: Sequence[str],
    requested_n: int,
) -> Tuple[List[Tuple[str, str]], str]:
    """Validate the complete artifact, then select a deterministic smoke prefix."""
    full_rows = [task_id_and_prompt(record) for record in records]
    full_task_ids = [task_id for task_id, _ in full_rows]
    if len(set(full_task_ids)) != len(full_task_ids):
        raise ValueError("Materialized MBPP+ benchmark contains duplicate task IDs")
    registry_ids = {str(task_id) for task_id in registry_task_ids}
    materialized_ids = set(full_task_ids)
    if materialized_ids != registry_ids:
        missing = sorted(registry_ids - materialized_ids)
        unexpected = sorted(materialized_ids - registry_ids)
        raise RuntimeError(
            "Materialized MBPP+ artifact does not match get_mbpp_plus(): "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    if 0 < requested_n < len(full_rows):
        return full_rows[:requested_n], "limited"
    return full_rows, "full"


def resolve_evalplus_image(explicit: Optional[str] = None) -> str:
    """Resolve the immutable campaign image and reject unpinned overrides."""
    image = explicit or os.environ.get("DRPT_EVALPLUS_IMAGE") or DEFAULT_EVALPLUS_IMAGE
    if image != DEFAULT_EVALPLUS_IMAGE:
        raise RuntimeError(
            "MBPP+ requires the Dolci32k profile's exact EvalPlus OCI image pin; "
            f"expected {DEFAULT_EVALPLUS_IMAGE!r}, got {image!r}"
        )
    return image


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evalplus_dataset_path(
    explicit: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve and verify the immutable MBPP+ JSONL used by the OCI runtime.

    EvalPlus does not bundle MBPP+ in its image. The evaluator therefore
    materializes the official dataset once on the host, verifies its content
    hash, and mounts that one file read-only while networking is disabled.
    """
    candidate = (
        explicit
        or os.environ.get("DRPT_EVALPLUS_DATASET_PATH")
        or os.path.join(
            os.path.expanduser("~"), ".cache", "evalplus", MBPP_PLUS_CACHE_FILENAME
        )
    )
    path = os.path.realpath(os.path.abspath(os.path.expanduser(candidate)))
    if not os.path.isfile(path):
        raise RuntimeError(
            "The pinned MBPP+ dataset cache is missing. Materialize "
            f"get_mbpp_plus(version={MBPP_PLUS_DATASET_VERSION!r}) on the host "
            "before submitting downstream evaluation, or set "
            "DRPT_EVALPLUS_DATASET_PATH to the verified JSONL. "
            f"Expected path: {path}"
        )
    if any(character in path for character in (":", ",", "\n")):
        raise RuntimeError(f"MBPP+ dataset path is unsafe for an Apptainer bind: {path!r}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != MBPP_PLUS_DATASET_SHA256:
        raise RuntimeError(
            "Pinned MBPP+ dataset hash mismatch: "
            f"expected {MBPP_PLUS_DATASET_SHA256}, got {actual_sha256} for {path}"
        )
    return path, actual_sha256


def resolve_evalplus_runtime_cache(
    sandbox_dir: str, explicit: Optional[str] = None
) -> str:
    """Return a cell-local cache outside containall's 64MB temporary HOME."""
    sandbox_root = os.path.realpath(os.path.abspath(sandbox_dir))
    environment_override = os.environ.get("DRPT_EVALPLUS_RUNTIME_CACHE")
    if explicit or environment_override:
        raise RuntimeError(
            "EvalPlus runtime-cache overrides are disabled: every evaluation "
            "must use its fresh, cell-local disposable sandbox"
        )
    candidate = os.path.join(sandbox_root, "runtime_cache")
    path = os.path.realpath(os.path.abspath(os.path.expanduser(candidate)))
    try:
        inside_sandbox = os.path.commonpath((sandbox_root, path)) == sandbox_root
    except ValueError:
        inside_sandbox = False
    if not inside_sandbox or path == sandbox_root:
        raise RuntimeError(
            "EvalPlus runtime cache must be an isolated child of its evaluation "
            f"sandbox: sandbox={sandbox_root!r}, cache={path!r}"
        )
    if any(character in path for character in (":", ",", "\n")):
        raise RuntimeError(
            f"EvalPlus runtime cache path is unsafe for an Apptainer bind: {path!r}"
        )
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create EvalPlus runtime cache: {path}") from exc
    if not os.path.isdir(path) or not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError(f"EvalPlus runtime cache is not usable: {path}")
    return path


def _evalplus_container_prefix(
    *,
    runner: str,
    image: str,
    dataset_path: str,
    runtime_cache_path: str,
    extra_binds: Sequence[str] = (),
) -> List[str]:
    sandbox_path = os.path.realpath(os.path.dirname(runtime_cache_path))
    if os.path.basename(os.path.realpath(runtime_cache_path)) != "runtime_cache":
        raise RuntimeError(
            "EvalPlus runtime cache must map exactly to /workspace/runtime_cache"
        )
    command = [
        runner,
        "exec",
        "--containall",
        "--cleanenv",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{dataset_path}:{MBPP_PLUS_CONTAINER_PATH}:ro",
        "--bind",
        f"{sandbox_path}:/workspace",
    ]
    for bind in extra_binds:
        command.extend(("--bind", bind))
    command.extend(
        [
        "--env",
        f"MBPP_OVERRIDE_PATH={MBPP_PLUS_CONTAINER_PATH}",
        "--env",
        f"XDG_CACHE_HOME={EVALPLUS_RUNTIME_CACHE_CONTAINER_PATH}",
        image,
        ]
    )
    return command


def build_evalplus_probe_command(
    *, runner: str, image: str, dataset_path: str, runtime_cache_path: str
) -> List[str]:
    """Build a network-off runtime/version/dataset preflight command."""
    script = (
        "import hashlib,json,os;"
        "from importlib.metadata import version;"
        "from evalplus.data import get_mbpp_plus;"
        "p=os.environ['MBPP_OVERRIDE_PATH'];"
        "d=hashlib.sha256(open(p,'rb').read()).hexdigest();"
        "c=os.path.join(os.environ['XDG_CACHE_HOME'],'evalplus');"
        "os.makedirs(c,exist_ok=True);"
        "w=os.path.join(c,'.drpt-write-probe');"
        "open(w,'w').write('ok');os.unlink(w);"
        f"t=get_mbpp_plus(version={MBPP_PLUS_DATASET_VERSION!r});"
        "print(json.dumps({'distribution_version':version('evalplus'),"
        "'dataset_sha256':d,'task_count':len(t),"
        "'runtime_cache_writable':True}))"
    )
    return _evalplus_container_prefix(
        runner=runner,
        image=image,
        dataset_path=dataset_path,
        runtime_cache_path=runtime_cache_path,
    ) + ["python", "-c", script]


def inspect_evalplus_runtime(
    *,
    runner: str,
    image: str,
    dataset_path: str,
    runtime_cache_path: str,
    expected_task_count: int,
) -> Dict[str, Any]:
    """Execute and validate the immutable container contract before scoring."""
    command = build_evalplus_probe_command(
        runner=runner,
        image=image,
        dataset_path=dataset_path,
        runtime_cache_path=runtime_cache_path,
    )
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"EvalPlus container preflight failed: {' '.join(command)}"
        ) from exc
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"EvalPlus container preflight returned malformed output: {completed.stdout!r}"
        ) from exc
    expected = {
        "distribution_version": EVALPLUS_CONTAINER_DISTRIBUTION_VERSION,
        "dataset_sha256": MBPP_PLUS_DATASET_SHA256,
        "task_count": expected_task_count,
        "runtime_cache_writable": True,
    }
    if payload != expected:
        raise RuntimeError(
            f"EvalPlus container contract mismatch: expected {expected}, got {payload}"
        )
    return payload



def assemble_solution(problem_prompt: str, generation: str) -> str:
    """Produce EvalPlus's self-contained ``solution`` field."""
    code = extract_code(clean_model_response(generation))
    if _FUNCTION_RE.search(code):
        return code
    # EvalPlus prompts end immediately before the function body. A model that
    # emitted only the body is still a valid completion when appended verbatim.
    return problem_prompt.rstrip() + "\n" + code


def write_samples(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def archive_existing_result(path: str) -> Optional[str]:
    """Move a prior EvalPlus intermediate aside so it cannot be reused."""
    if not os.path.isfile(path):
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = f"{path}.previous-{timestamp}"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}.previous-{timestamp}-{suffix}"
        suffix += 1
    os.replace(path, candidate)
    return candidate


def atomic_copy_file(source: str, destination: str) -> None:
    """Copy a sandbox result into the run directory with atomic visibility."""
    destination_dir = os.path.dirname(os.path.abspath(destination))
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.tmp-", dir=destination_dir
    )
    try:
        with open(source, "rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def resolve_sandbox_runner(explicit: Optional[str] = None) -> str:
    if explicit:
        resolved = shutil.which(explicit) if os.path.basename(explicit) == explicit else explicit
        if resolved and os.path.isfile(resolved):
            return resolved
        raise RuntimeError(f"EvalPlus sandbox runner is not executable: {explicit}")
    for candidate in ("apptainer", "singularity"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "MBPP+ requires Apptainer or Singularity; unsafe host execution is intentionally disabled"
    )


def build_evalplus_command(
    *,
    runner: str,
    image: str,
    output_dir: str,
    dataset_path: str,
    runtime_cache_path: str,
    samples_filename: str = "mbpp_plus_samples.jsonl",
    dataset_version: str = MBPP_PLUS_DATASET_VERSION,
) -> List[str]:
    """Build a no-shell command for the pinned official EvalPlus image."""
    output_dir = os.path.abspath(output_dir)
    if os.path.realpath(os.path.dirname(runtime_cache_path)) != os.path.realpath(
        output_dir
    ):
        raise RuntimeError(
            "EvalPlus runtime cache must belong to the bound disposable sandbox"
        )
    sample_host_path = os.path.join(output_dir, samples_filename)
    return _evalplus_container_prefix(
        runner=runner,
        image=image,
        dataset_path=dataset_path,
        runtime_cache_path=runtime_cache_path,
        extra_binds=(
            f"{sample_host_path}:/workspace/{samples_filename}:ro",
        ),
    ) + [
        "evalplus.evaluate",
        "--dataset",
        "mbpp",
        "--samples",
        f"/workspace/{samples_filename}",
        "--version",
        dataset_version,
    ]


def evalplus_result_filename(samples_filename: str) -> str:
    """Return the result name dictated by EvalPlus 0.3.1 ``evaluate``."""
    if not samples_filename.endswith(".jsonl"):
        raise ValueError("EvalPlus samples filename must end in .jsonl")
    return samples_filename.removesuffix(".jsonl") + "_eval_results.json"


def parse_evalplus_results(
    payload: Mapping[str, Any],
    expected_task_ids: Sequence[str],
    *,
    allow_extra_tasks: bool = False,
) -> Dict[str, Any]:
    expected = set(expected_task_ids)
    evaluated = set(payload.get("eval", {}))
    if not expected.issubset(evaluated) or (
        not allow_extra_tasks and evaluated != expected
    ):
        missing = sorted(expected - evaluated)
        unexpected = sorted(evaluated - expected)
        raise RuntimeError(
            "EvalPlus task coverage mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    # EvalPlus 0.3.1 prints aggregate pass@k but persists only per-completion
    # statuses. This campaign generates exactly one greedy completion per task,
    # so pass@1 is the mean of those official statuses.
    rows = []
    for task_id in expected_task_ids:
        task_rows = payload.get("eval", {}).get(task_id)
        if not isinstance(task_rows, list) or len(task_rows) != 1:
            raise RuntimeError(
                f"EvalPlus expected exactly one completion for {task_id}, got {task_rows!r}"
            )
        rows.append(task_rows[0])
    base = sum(row.get("base_status") == "pass" for row in rows) / len(rows)
    plus = sum(
        row.get("base_status") == "pass" and row.get("plus_status") == "pass"
        for row in rows
    ) / len(rows)
    return {
        "base_pass_at_1": base * 100.0,
        "plus_pass_at_1": plus * 100.0,
        # EvalPlus ``plus`` executes the base tests and the additional tests.
        "base_plus_extra_pass_at_1": plus * 100.0,
        "n_test": len(expected),
        "dataset_hash": payload.get("hash"),
    }


def official_problem_registry() -> Tuple[Dict[str, Mapping[str, Any]], str]:
    """Resolve the full MBPP+ registry from the pinned host package."""
    version = require_distribution_version("evalplus", EVALPLUS_VERSION)
    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:  # pragma: no cover - guarded by the version check
        raise RuntimeError("evalplus is installed but get_mbpp_plus is unavailable") from exc
    problems = get_mbpp_plus(version=MBPP_PLUS_DATASET_VERSION)
    if not isinstance(problems, Mapping) or not problems:
        raise RuntimeError("EvalPlus get_mbpp_plus() returned an empty/malformed registry")
    return {str(task_id): problem for task_id, problem in problems.items()}, version


def official_task_ids() -> Tuple[List[str], str]:
    """Resolve dynamic task IDs without hard-coding benchmark cardinality."""
    problems, version = official_problem_registry()
    return sorted(problems), version


def add_official_smoke_fillers(
    samples: Sequence[Mapping[str, Any]],
    problems: Mapping[str, Mapping[str, Any]],
    selected_task_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Fill unselected tasks so EvalPlus can execute a limited generation smoke.

    EvalPlus 0.3.1 requires one completion for every registry task. Metrics are
    still computed only for ``selected_task_ids``; official canonical solutions
    merely satisfy the runner's completeness assertion and never enter reports.
    """
    selected = set(selected_task_ids)
    sample_task_ids = [str(sample.get("task_id")) for sample in samples]
    if (
        len(sample_task_ids) != len(set(sample_task_ids))
        or set(sample_task_ids) != selected
        or not selected.issubset(problems)
    ):
        raise RuntimeError(
            "MBPP+ limited-smoke samples/selected IDs do not match the registry"
        )
    output = [dict(sample) for sample in samples]
    for task_id in sorted(problems):
        if task_id in selected:
            continue
        problem = problems[task_id]
        prompt = problem.get("prompt")
        canonical = problem.get("canonical_solution")
        if not isinstance(prompt, str) or not isinstance(canonical, str):
            raise RuntimeError(f"Malformed official MBPP+ problem for {task_id}")
        output.append({"task_id": task_id, "solution": prompt + canonical})
    if len(output) != len(problems):
        raise RuntimeError(
            "MBPP+ limited-smoke filler construction did not produce full coverage"
        )
    return output


def validate_official_registry_coverage(
    payload: Mapping[str, Any], registry_task_ids: Sequence[str]
) -> None:
    """Require the raw official result to cover the dynamic full registry."""
    evaluated = payload.get("eval")
    actual = set(evaluated) if isinstance(evaluated, Mapping) else set()
    expected = set(registry_task_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            "EvalPlus canonical-filler coverage mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    all_records = load_bench_records(
        getattr(args, "data_dir", "./data"), "mbpp_plus", k=-1
    )
    requested_n = getattr(args, "n_test", -1)
    problem_registry, evaluator_version = official_problem_registry()
    registry_task_ids = sorted(problem_registry)
    task_rows, evaluation_scope = select_task_rows(
        all_records, registry_task_ids, requested_n
    )
    records = all_records[: len(task_rows)]
    task_ids = [task_id for task_id, _ in task_rows]

    prompts = [
        render_generation_chat(
            tokenizer,
            _PROMPT_TEMPLATE.format(prompt=problem_prompt),
            enable_thinking=True,
        )
        for _, problem_prompt in task_rows
    ]
    generations = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False,
        disable_tqdm=False,
    )
    samples = []
    generation_rows = []
    for (task_id, problem_prompt), generation in zip(task_rows, generations):
        scored_response = clean_model_response(generation)
        solution = assemble_solution(problem_prompt, generation)
        samples.append({"task_id": task_id, "solution": solution})
        generation_rows.append(
            {
                "task_id": task_id,
                "raw_generation": generation,
                "scored_response": scored_response,
                "solution": solution,
            }
        )

    configured_output_dir = getattr(args, "output_dir", None)
    if not configured_output_dir:
        raise ValueError("MBPP+ requires args.output_dir for samples and sandbox results")
    output_dir = os.path.abspath(configured_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sandbox_dir = tempfile.mkdtemp(prefix=".mbpp-plus-sandbox-", dir=output_dir)
    os.chmod(sandbox_dir, 0o700)
    samples_filename = "mbpp_plus_samples.jsonl"
    results_filename = evalplus_result_filename(samples_filename)
    samples_path = os.path.join(sandbox_dir, samples_filename)
    results_path = os.path.join(sandbox_dir, results_filename)
    archive_existing_result(results_path)
    evaluation_samples = (
        add_official_smoke_fillers(samples, problem_registry, task_ids)
        if evaluation_scope == "limited"
        else samples
    )
    write_samples(samples_path, evaluation_samples)
    write_samples(os.path.join(output_dir, "mbpp_plus_generations.jsonl"), generation_rows)

    runner = resolve_sandbox_runner(getattr(args, "evalplus_runner", None))
    image = resolve_evalplus_image(getattr(args, "evalplus_image", None))
    dataset_path, dataset_sha256 = resolve_evalplus_dataset_path(
        getattr(args, "evalplus_dataset_path", None)
    )
    runtime_cache_path = resolve_evalplus_runtime_cache(
        sandbox_dir,
        getattr(args, "evalplus_runtime_cache", None)
    )
    container_runtime = inspect_evalplus_runtime(
        runner=runner,
        image=image,
        dataset_path=dataset_path,
        runtime_cache_path=runtime_cache_path,
        expected_task_count=len(registry_task_ids),
    )
    command = build_evalplus_command(
        runner=runner,
        image=image,
        output_dir=sandbox_dir,
        dataset_path=dataset_path,
        runtime_cache_path=runtime_cache_path,
        samples_filename=samples_filename,
    )
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Sandboxed EvalPlus evaluation failed: {' '.join(command)}") from exc
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            official_payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"EvalPlus did not write a readable result: {results_path}") from exc
    if evaluation_scope == "limited":
        validate_official_registry_coverage(official_payload, registry_task_ids)
    official_archive_filename = (
        "mbpp_plus_limited_smoke_with_canonical_fillers_official_eval_results.json"
        if evaluation_scope == "limited"
        else "mbpp_plus_official_eval_results.json"
    )
    official_archive_path = os.path.join(
        output_dir, official_archive_filename
    )
    archive_existing_result(official_archive_path)
    atomic_copy_file(results_path, official_archive_path)

    scores = parse_evalplus_results(
        official_payload,
        task_ids,
        allow_extra_tasks=evaluation_scope == "limited",
    )
    scores["evaluation_scope"] = evaluation_scope
    scores["provenance"] = result_provenance(
        dataset_repository=DATASET_REPOSITORY,
        dataset_revision=single_source_revision(records),
        dataset_split="test",
        evaluator={
            "package": "evalplus",
            "version": evaluator_version,
            "release": f"v{EVALPLUS_RELEASE_VERSION}",
            "container_distribution_version": container_runtime[
                "distribution_version"
            ],
            "dataset_version": MBPP_PLUS_DATASET_VERSION,
            "dataset_jsonl_sha256": dataset_sha256,
            "dataset_task_count": container_runtime["task_count"],
            "canonical_smoke_fillers": len(evaluation_samples) - len(samples),
            "image": image,
            "image_digest": EVALPLUS_IMAGE_DIGEST,
            "dataset_hash": scores.get("dataset_hash"),
            "primary_metric": "base_plus_extra_pass_at_1",
        },
        n_tasks=len(records),
        max_new_tokens=max_new_tokens,
        thinking=True,
    )
    shutil.rmtree(sandbox_dir)
    return scores
