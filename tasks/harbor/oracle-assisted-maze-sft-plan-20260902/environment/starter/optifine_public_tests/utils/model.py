"""Evaluator-owned fixed maze policy architecture and greedy generation."""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from .maze import ACTIONS, INITIAL_CHECKPOINT_SHA256, encode


class MazePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Conv2d(5, 64, 3, padding=1), torch.nn.GELU()]
        for _ in range(7):
            layers += [torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.GELU()]
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(64 * 17 * 17, 128),
            torch.nn.GELU(), torch.nn.Linear(128, 4))

    def forward(self, inputs):
        return self.head(self.body(inputs))


INITIAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "fixtures" / "maze_policy_init.pt"
INITIAL_SHA256 = INITIAL_CHECKPOINT_SHA256


@lru_cache(maxsize=1)
def _initial_state():
    payload = INITIAL_CHECKPOINT.read_bytes()
    if hashlib.sha256(payload).hexdigest() != INITIAL_SHA256:
        raise RuntimeError("evaluator-owned maze policy checkpoint failed authentication")
    state = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        expected = MazePolicy().state_dict()
    if tuple(state) != tuple(expected) or any(
            state[key].shape != value.shape or not torch.isfinite(state[key]).all().item()
            for key, value in expected.items()):
        raise RuntimeError("evaluator-owned maze policy checkpoint is invalid")
    return state


def initialized_model():
    with torch.random.fork_rng(devices=[]):
        model = MazePolicy()
    model.load_state_dict(_initial_state(), strict=True)
    return model


@torch.inference_mode()
def rollout(model, maze, device, upper_bound=60):
    model.eval()
    point, visited, actions = maze["start"], {maze["start"]}, []
    for _ in range(upper_bound):
        inputs = encode(maze, point, visited).unsqueeze(0).to(device)
        logits = model(inputs)[0]
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


@torch.inference_mode()
def rollout_batched(model, mazes, device, upper_bound=60):
    model.eval()
    points = [maze["start"] for maze in mazes]
    visited = [{point} for point in points]
    paths = [[] for maze in mazes]
    active = list(range(len(mazes)))
    for _ in range(upper_bound):
        if not active:
            break
        inputs = torch.stack([encode(mazes[index], points[index], visited[index])
                              for index in active]).to(device)
        logits = model(inputs)
        allowed = []
        for index in active:
            row, col = points[index]
            allowed.append([not mazes[index]["grid"][row + dr][col + dc]
                            and (row + dr, col + dc) not in visited[index]
                            for dr, dc in ACTIONS])
        allowed = torch.tensor(allowed, dtype=torch.bool, device=device)
        logits.masked_fill_(~allowed, -torch.inf)
        choices = logits.argmax(dim=1)
        choices[~torch.isfinite(logits).any(dim=1)] = -1
        # Batch kernels can perturb close logits, especially with TF32 convolutions.
        # Recheck legal choices within 1% of their scale using the original input shape.
        top = logits.topk(2, dim=1).values
        close = (torch.isfinite(top).all(dim=1)
                 & (top[:, 0] - top[:, 1] <= 0.01 * top.abs().amax(dim=1).clamp_min(1)))
        choices[close] = -2
        following = []
        for batch_index, (index, action) in enumerate(zip(active, choices.cpu().tolist())):
            if action == -2:
                single = model(inputs[batch_index:batch_index + 1])[0]
                single.masked_fill_(~allowed[batch_index], -torch.inf)
                action = int(single.argmax().item()) if torch.isfinite(single).any() else -1
            if action < 0:
                continue
            paths[index].append(action)
            dr, dc = ACTIONS[action]
            row, col = points[index]
            points[index] = (row + dr, col + dc)
            visited[index].add(points[index])
            if points[index] != mazes[index]["goal"]:
                following.append(index)
        active = following
    return paths


def collate(examples, device):
    return (torch.stack([item[0] for item in examples]).to(device),
            torch.tensor([item[1] for item in examples], device=device))
