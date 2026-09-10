# Improve Oracle-Assisted Maze Supervised Fine-Tuning

Hardware: one H100. Agent budget: six hours (21,600 seconds). Verifier budget: one hour, run afterwards in a separate offline sandbox.

## Task

A small maze-solving policy is fine-tuned with a fixed architecture, optimizer, seed schedule, and 600-update budget. The agent decides which training examples the model sees and how much weight each one gets, by editing one file: `/environment/starter/maze_task/candidate.py`. Nothing else may change.

For each training maze the evaluator supplies a catalog of records: the canonical shortest path, valid alternative detours, deterministic failed rollouts, shortest-recovery actions from failure states, and failure prefixes. Each record carries a fixed positive or contrastive treatment. The candidate selects records from this catalog and assigns finite nonnegative weights; the evaluator owns the model, losses, and training resources.

## Why it is hard

Lowering teacher-forced loss is not enough. The model is evaluated by free-running greedy decoding on unseen 17x17 mazes, where a single wrong legal branch can lead to a dead end or a longer path. Decoding masks walls and revisited cells but performs no search, so the learned policy has to choose branches that reach the goal. Alternative trajectories trade state coverage against path efficiency, and long failed rollouts can crowd out gold supervision if weighted carelessly.

## Public testing

`/environment/starter/maze_task/public_test.sh` supports:

- `--mode contract`: CPU-only check on four mazes that the candidate returns valid selections.
- `--mode quick`: reduced-budget training and scoring for fast feedback.
- `--mode full`: three public maze groups, 192 training and 96 evaluation mazes each, two seeds per group.
- `--mode full --validation-fold 1`: the same on three held-out public groups.

Full-mode checks save a submission checkpoint under `/logs/artifacts/progress` before testing it. `/environment/starter/maze_task/README.md` documents the submission API, budgets, and scoring.

## Scoring

Private verification trains on six private maze panels with three seed pairs each, twelve candidate runs in total. For every evaluation maze, a goal-reaching path of length L scores `q = clamp((60 - L) / (60 - L*), 0, 1)`, where L* is the shortest path length; malformed, colliding, or incomplete paths score 0. The reward is the mean q over all runs, between 0 and 1. A reward of 1 requires shortest paths on every evaluation maze in every run. The starter's uniformly weighted canonical-path fine-tuning scores about 0.48.

Invalid candidate selections, out-of-catalog IDs, changed treatments, or candidate execution failures yield reward 0.
