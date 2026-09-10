# CLAUDE.md

Memory Layers — a JAX/TPU research framework for augmenting frozen LLMs with learnable
external memory. **[`wiki/`](wiki/README.md) is the source of truth** for how the system
works; the root `README.md` is the human-facing overview. Entry points (`train.py`, `eval.py`,
`rag_eval.py`) run **on a TPU box** through `uv` — **there is no local accelerator** (see
*Start here*).

## Start here — how work happens

This is a **remote-TPU research framework: there is no local accelerator.** `train.py` /
`eval.py` run on TRC TPU VMs through `uv`, launched with
`scripts/infrastructure/multi-vm-tpu-run.sh` (it syncs your tree — a git worktree included — and
runs a script on the box). **Before running or benchmarking anything, read the launch runbook:**
[`wiki/infrastructure/experiment-launch-instructions.md`](wiki/infrastructure/experiment-launch-instructions.md).
Don't try to run locally, and don't hand-roll ssh/scp/sync — use the launcher.

A task usually flows:
1. **Orient** — skim the relevant `wiki/` page (it's the source of truth) before touching code.
2. **Run / measure on a box** — via the launcher. Benchmarks and experiments are **standalone
   scripts** under `scripts/…` that import and reuse existing code; **don't instrument
   `train.py` / `trainer.py` / `eval.py`** just to take a measurement.
3. **Document** — every change/experiment ships its write-up (rule below), and any reusable
   gotcha or tooling fix earns a short wiki note even when it isn't tied to a code change.

## The documentation rule

**Every substantive change ships its docs in the same commit/PR.** Undocumented work is
unfinished work. There are two genres, and they live in different places:

| The change is a… | You must… |
|------------------|-----------|
| **Code / feature / codebase change** | **(a)** Update the affected **reference** page(s) under `wiki/{architecture,data,evaluation,training,infrastructure}/` so "how it works now" stays true, **and (b)** add an **implementation note** in [`wiki/implementations/`](wiki/implementations/README.md): motivation, options + tradeoffs, the approach chosen, how it was built & integrated, and the test record. See [`wiki/implementations/GUIDELINES.md`](wiki/implementations/GUIDELINES.md). |
| **Experiment** | Add a write-up in [`wiki/experiments/`](wiki/experiments/README.md): hypothesis/motivation → setup + reproducibility → **full results** with pointers to plots/tables/artifacts → a one-line conclusion. See [`wiki/experiments/GUIDELINES.md`](wiki/experiments/GUIDELINES.md). |

Then **link the new page from its section README** (the index table) — an unlinked page is a
lost page.

## Golden rules

- **Reference pages are living; logs are append-only.** If your change makes a `wiki/`
  reference page wrong, fix it *in the same change* — do not leave it stale and do not
  write a competing "how I implemented X" doc (that belongs in the reference page). Implementation
  notes and experiment logs, by contrast, are dated and never rewritten — supersede, don't edit.
- **Results must be git-friendly.** Small results (metrics, tables) go **in the doc** as
  Markdown tables or as committed CSV/JSON. Large artifacts — checkpoints, raw result dumps,
  plots — are **never committed as binaries**; record a *stable pointer* instead: a `gs://`
  path, a wandb run URL/ID, or a `results/…` path. Commit the `.png` only when it's small and
  central to the conclusion.
- **Every experiment records a repro block** so a colleague can rerun it blind: exact
  `uv run …` command + Hydra overrides, **commit SHA**, checkpoint `gs://` path(s), **wandb**
  run URL/ID, and **TPU type** (e.g. `v4-8`). The fully-resolved config is also saved in the
  checkpoint's `.hydra/config.yaml`.
- **A colleague must be able to tell, at a glance, what was done and what the conclusion is.**
  Lead with the answer, not the narrative.

## Tests

Tests in `tests/` are **standalone scripts** (no pytest dependency): run the relevant one with
`uv run python tests/test_<foo>.py`. A feature's implementation note must name the test file(s), the
command, and paste the pass/fail output.

## Where things are

- Reference wiki: `architecture/`, `data/`, `evaluation/`, `training/` each have a README index;
  `infrastructure/` (checkpointing, launch) is indexed by [`wiki/README.md`](wiki/README.md).
- Launching runs on TPUs: [`wiki/infrastructure/experiment-launch-instructions.md`](wiki/infrastructure/experiment-launch-instructions.md).
- Configs are Hydra (`configs/`); checkpoints and large results live in GCS (`gs://`).
