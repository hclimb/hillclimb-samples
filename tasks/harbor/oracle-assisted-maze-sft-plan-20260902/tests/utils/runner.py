"""Isolated candidate construction, fixed-budget training, and comparison."""

import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .contract import (CandidateError, build_catalog, load_examples, read_regular_file,
                       read_selections, save_selections)
from .maze import build_groups, fixture_manifest, partition_ids, score_path
from .model import MazePolicy, collate, initialized_model, rollout_batched

SEEDS = (17, 29, 43)
REPEAT_SEED = 101
PUBLIC_FOLDS = (0, 1)
PRIVATE_PANELS = 6
CONFIGS = {
    "quick": {"train_count": 48, "eval_count": 24, "steps": 120, "batch_size": 32},
    "full": {"train_count": 192, "eval_count": 96, "steps": 600, "batch_size": 64},
}
CHILD_TIMEOUT = 900


def run_schedule(mode, panel_count=len(SEEDS)):
    return tuple((panel, seed) for panel in range(panel_count)
                 for seed in ((SEEDS[panel % len(SEEDS)], REPEAT_SEED) if mode == "full"
                              else (SEEDS[panel % len(SEEDS)],)))


def canonical_arrays(groups):
    examples = [(state.numpy(), action) for group in groups
                for state, action in group["canonical_examples"]]
    return (np.stack([item[0] for item in examples]),
            np.asarray([item[1] for item in examples], dtype=np.int64),
            np.ones(len(examples), dtype=np.float32),
            np.zeros(len(examples), dtype=np.bool_))


def contrastive_losses(logits, targets, negative):
    selected = logits.softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
    return torch.where(negative, -torch.log1p(-selected.clamp(max=1 - 1e-6)),
                       -torch.log(selected.clamp(min=1e-6)))


def train_model(model, arrays, config, device):
    states, actions, weights, negatives = arrays
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(config["seed"])
    order = torch.randperm(len(actions), generator=generator).tolist()
    for step in range(config["steps"]):
        offset = step * config["batch_size"]
        indexes = [order[(offset + index) % len(order)] for index in range(config["batch_size"])]
        inputs = torch.from_numpy(states[indexes]).to(device)
        targets = torch.from_numpy(actions[indexes]).to(device)
        batch_weights = torch.from_numpy(weights[indexes]).to(device)
        negative = torch.from_numpy(negatives[indexes]).to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = contrastive_losses(model(inputs), targets, negative)
        loss = (losses * batch_weights).sum() / batch_weights.sum().clamp(min=1e-6)
        if not torch.isfinite(loss):
            raise CandidateError("training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def validate_model(model, initial_keys):
    if type(model) is not MazePolicy or tuple(model.state_dict()) != initial_keys:
        raise CandidateError("training must preserve the fixed MazePolicy architecture")
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise CandidateError("candidate checkpoint contains non-finite values")


@torch.inference_mode()
def evaluate(model, mazes, device):
    scores, lengths, nll_sum, nll_count = [], [], 0.0, 0
    outcomes = {"success": 0, "collision": 0, "timeout": 0, "malformed": 0}
    model.eval()
    for maze, actions in zip(mazes, rollout_batched(model, mazes, device)):
        score, outcome = score_path(maze, actions)
        scores.append(score)
        lengths.append(len(actions))
        outcomes[outcome] += 1
    examples = [example for maze in mazes for example in maze["canonical_examples"]]
    for start in range(0, len(examples), 256):
        batch = examples[start:start + 256]
        states, targets = collate(batch, device)
        nll_sum += float(torch.nn.functional.cross_entropy(
            model(states), targets, reduction="sum").item())
        nll_count += len(batch)
    count = len(mazes)
    return {"quality": sum(scores) / count,
            "success_rate": outcomes["success"] / count,
            "collision_rate": outcomes["collision"] / count,
            "mean_path_length": sum(lengths) / count,
            "validation_nll": nll_sum / nll_count,
            "outcomes": outcomes}


def seed_sets(mode, split_offset, private=False, public_fold=None):
    settings = CONFIGS[mode]
    scope = "private" if private or split_offset >= 2_000_000 else "public"
    if public_fold is not None and (
            scope != "public" or mode != "full" or public_fold not in PUBLIC_FOLDS):
        raise ValueError("public folds require full mode and fold 0 or 1")
    train_ids = partition_ids(f"{scope}_train")
    eval_ids = partition_ids(f"{scope}_eval")
    panel_count = len(PUBLIC_FOLDS) * len(SEEDS) if scope == "public" else PRIVATE_PANELS
    full = CONFIGS["full"]
    if (len(train_ids) != panel_count * full["train_count"]
            or len(eval_ids) != panel_count * full["eval_count"]):
        raise ValueError(f"{scope} fixture counts do not match the fixed-budget panels")
    training, evaluation = [], []
    runs = PRIVATE_PANELS if scope == "private" and mode == "full" else len(SEEDS)
    for run_index in range(runs):
        # Both paths pair each seed with a distinct, equally sized maze panel.
        panel = (public_fold or 0) * len(SEEDS) + run_index if scope == "public" else run_index
        train_start = panel * full["train_count"]
        eval_start = panel * full["eval_count"]
        training.append(list(train_ids[train_start:train_start + settings["train_count"]]))
        evaluation.append(list(eval_ids[eval_start:eval_start + settings["eval_count"]]))
    return training, evaluation


def _run(command, *, untrusted=False, cwd=None, timeout=None):
    timeout = CHILD_TIMEOUT if timeout is None else timeout
    environment = {**os.environ, "PYTHONHASHSEED": "0"}
    if untrusted:
        environment = {key: os.environ[key] for key in ("LANG", "PATH")
                       if key in os.environ}
        environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
                            "PYTHONNOUSERSITE": "1", "TMPDIR": str(cwd)})
    process = None
    output_files = None
    try:
        if untrusted:
            output_files = tuple(tempfile.TemporaryFile(mode="w+", errors="replace") for _ in range(2))
            process = subprocess.Popen(command, text=True, stdout=output_files[0],
                                       stderr=output_files[1], env=environment, cwd=cwd,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            process.wait(timeout=timeout)
            stdout, stderr = "", ""
        else:
            process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=environment, cwd=cwd)
            stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if untrusted:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        excerpt = ((stdout or "") + "\n" + (stderr or ""))[-8000:]
        failure = CandidateError if untrusted else RuntimeError
        kind = "candidate" if untrusted else "trusted"
        raise failure(f"{kind} worker timed out; output: {excerpt}") from error
    finally:
        if untrusted and process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for output_file in output_files or ():
                output_file.seek(0)
            if output_files:
                for output_file in output_files:
                    output_file.seek(max(0, os.fstat(output_file.fileno()).st_size - 8000))
                stdout, stderr = (output_file.read() for output_file in output_files)
                for output_file in output_files:
                    output_file.close()
    return process.returncode, (stdout + "\n" + stderr)[-8000:]


def training_groups(prompt_ids):
    allowed = set().union(*(set(partition_ids(name)) for name in fixture_manifest()["partitions"]
                            if name.endswith("_train")))
    if not set(prompt_ids) <= allowed:
        raise ValueError("candidate inputs must use training partitions, never evaluation mazes")
    return build_groups(prompt_ids)


def construct_examples(starter, prompt_ids, artifact):
    # Keep an independent catalog; the child is not an OS security boundary.
    catalog, groups = build_catalog(training_groups(prompt_ids))
    with tempfile.TemporaryDirectory(prefix="maze-candidate-") as temporary:
        directory = Path(temporary)
        inputs, output = directory / "input", directory / "output"
        inputs.mkdir()
        output.mkdir()
        worker = Path(__file__).resolve().parents[1] / "verifiers/build_candidate.py"
        (inputs / "worker.py").write_bytes(worker.read_bytes())
        command = [sys.executable, "-I", "-B", str(inputs / "worker.py"),
                   str(inputs), str(output)]
        # Check worker dependencies and resource setup before candidate execution.
        code, excerpt = _run([*command, "--check"])
        if code != 0 or excerpt.strip() != "builder-ready":
            raise RuntimeError(f"evaluator builder preflight failed: {excerpt}")
        source = Path(starter) / "maze_task/candidate.py"
        (inputs / "candidate.py").write_bytes(read_regular_file(source))
        torch.save(groups, inputs / "groups.pt")
        code, excerpt = _run(command, untrusted=True, cwd=directory)
        if code != 0:
            raise CandidateError(f"candidate construction failed; output: {excerpt}")
        selections = read_selections(output / "selections.json")
        save_selections(artifact, selections, catalog)
        load_examples(artifact)


def aggregate(results, mode, private=False):
    expected = CONFIGS[mode]["eval_count"]
    schedule = run_schedule(mode, PRIVATE_PANELS if private and mode == "full" else len(SEEDS))
    for method in ("baseline", "candidate"):
        runs = results[method]
        if len(runs) != len(schedule):
            raise RuntimeError(f"{method} worker omitted a scored panel or seed")
        for (panel, seed), run in zip(schedule, runs):
            if (run.get("panel") != panel or run["seed"] != seed
                    or sum(run["outcomes"].values()) != expected
                    or not math.isfinite(run["quality"]) or not 0 <= run["quality"] <= 1):
                raise RuntimeError(f"{method} worker returned incomplete or invalid metrics")
    baseline = statistics.mean(run["quality"] for run in results["baseline"])
    candidate = statistics.mean(run["quality"] for run in results["candidate"])
    if baseline <= 0:
        raise RuntimeError("ordinary-gold baseline quality is nonpositive")
    return {"reward": candidate, "baseline": baseline, "candidate": candidate,
            "ratio": candidate / baseline,
            "baseline_runs": results["baseline"], "candidate_runs": results["candidate"],
            "evaluation_version": "quality-v1",
            "aggregation": "mean_over_panel_seed_pairs" if mode == "full" else "mean_over_disjoint_panels"}


def compare(starter, mode, split_offset, device, private=False, public_fold=None):
    tests = Path(__file__).resolve().parents[1]
    private = private or split_offset >= 2_000_000
    training_seeds, evaluation_seeds = seed_sets(mode, split_offset, private=private,
                                               public_fold=public_fold)
    with tempfile.TemporaryDirectory(prefix="maze-eval-") as directory_name:
        directory = Path(directory_name)
        artifacts = []
        for run_index, seeds in enumerate(training_seeds):
            artifact = directory / f"examples-{run_index}.npz"
            construct_examples(starter, seeds, artifact)
            artifacts.append(str(artifact))
        seed_path = directory / "training.json"
        seed_path.write_text(json.dumps(training_seeds))
        evaluation_path = directory / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation_seeds))
        results = {}
        for method in ("baseline", "candidate"):
            output = directory / f"{method}.json"
            command = [sys.executable, "-I", "-B", str(tests / "verifiers" / "run_training.py"), method,
                       mode, str(seed_path), str(output), str(evaluation_path), str(device)]
            if method == "candidate":
                command.extend(artifacts)
            timeout = CHILD_TIMEOUT * (2 if private and mode == "full" else 1)
            code, child_excerpt = _run(command, timeout=timeout)
            if code != 0:
                message = f"{method} worker failed; output: {child_excerpt}"
                if method == "candidate" and "CANDIDATE_ERROR:" in child_excerpt:
                    raise CandidateError(message)
                raise RuntimeError(message)
            results[method] = json.loads(output.read_text())
    return aggregate(results, mode, private=private)
