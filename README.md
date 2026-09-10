# Hillclimb samples

Three Harbor tasks for open-ended ML and ML-systems optimization, the job
configuration that runs them, and one completed six-hour job with all thirty
trial directories.

## Tasks

| Task | Hardware | Objective |
|---|---|---|
| [`learn-to-retrieve-both-supporting-documents-for-hotpotqa-c3200db1`](tasks/harbor/learn-to-retrieve-both-supporting-documents-for-hotpotqa-c3200db1/) | 2 H200 | Improve the training recipe for a BERT-Tiny retriever so it ranks both HotpotQA supporting paragraphs higher, within a 900-second training budget. Scored by nDCG@10 over 3,072 questions and 60,000 paragraphs. |
| [`native-model1-paged-sparse-attention-b200`](tasks/harbor/native-model1-paged-sparse-attention-b200/) | 1 B200 | Speed up a FlashMLA FP8 paged sparse-attention kernel on one fixed workload while all 22 original correctness cases keep passing. Scored as throughput relative to an estimated dense-BF16 roofline, with the untouched starter at 0 and the roofline at 1. |
| [`oracle-assisted-maze-sft-plan-20260902`](tasks/harbor/oracle-assisted-maze-sft-plan-20260902/) | 1 H100 | Choose and weight the training examples for fixed-budget supervised fine-tuning of a small 17x17 maze-solving model by editing one file, `candidate.py`. Scored by path correctness and efficiency on unseen mazes. |

Each task gives the agent a six-hour working window. Every task README
describes its verifier, metric, and score in full.

## Layout

```
hillclimb-samples/
  README.md
  configs/harbor-mini-swe.yaml             job configuration for the three tasks
  tasks/harbor/<task>/                     one directory per task
    instruction.md                         solver-facing objective
    task.toml                              runtime, hardware, network, artifact contract
    README.md                              task, verifier, and scoring overview
    environment/                           agent Dockerfile and starter workspace
    tests/                                 verifier, protected workloads, scoring
    solution/                              reference implementation
  jobs/six-hour-inference-only-20260910-0304/   the completed job
    config.json                            resolved job configuration
    result.json                            job-level result
    job.log                                job log
    metrics/summary.txt                    per-agent summary
    metrics/trials.csv                     one row per trial
    task-snapshots/<checksum>/             exact task copies used by the job
    <task>__<id>/                          one directory per trial (30 total)
      result.json                          trial reward, timing, cost, errors
      trial.log                            trial log
      agent/                               agent trajectory and session logs
      verifier/                            reward.json, diagnostics, verifier stdout
      artifacts/final_app/                 the submitted workspace
      artifacts/logs/                      checkpoints and public-test output
```

## Cloning

Model weights, the HotpotQA corpus, compiled kernels, wheels, and trial
checkpoints are stored with Git LFS (about 3.8 GB). Install Git LFS before
cloning, or the clone contains small pointer files in their place and the task
images will not build:

```bash
git lfs install
```

```bash
git clone https://github.com/hclimb/hillclimb-samples.git
```

If the repository was cloned before Git LFS was installed, fetch the objects
afterwards:

```bash
git lfs pull
```

## The completed job

`jobs/six-hour-inference-only-20260910-0304` ran all three tasks with two agent
lanes, five attempts each, thirty trials in total, on Modal. Each agent had the
full six-hour window; verification ran afterwards in a separate offline sandbox.

| Lane | Agent | Model |
|---|---|---|
| 1 | Claude Code | Claude Fable 5.1 through Amazon Bedrock |
| 2 | Codex | GPT-6 Astra through Azure OpenAI |

This job was run with Werm, Hillclimb's internal fork of Harbor, against the
Werm copies of the same three tasks, so its `config.json` lists agent names and
model providers that differ from the packaged Harbor configuration. The task
contents, budgets, verifiers, and scoring are identical to the copies under
`tasks/harbor/`. Start with `metrics/summary.txt` and `metrics/trials.csv`, then
open a trial directory for its trajectory, verifier output, and submitted
workspace.

## Running the configuration

`configs/harbor-mini-swe.yaml` runs the three tasks on Modal with two
mini-SWE-agent lanes, Claude Fable 5.1 and GPT-6 Astra, both through OpenRouter,
three attempts each. Agent commands may run for up to 900 seconds. During the
solver phase the sandbox can reach only `openrouter.ai`; the verifier sandbox is
fully offline.

Install Harbor with the Modal extra and authenticate to your Modal workspace:

```bash
uv tool install 'harbor[modal]==0.22.0' --with-executables-from modal
```

```bash
modal setup
```

Put your OpenRouter key in a `.env` file at the root of this directory:

```
OPENROUTER_API_KEY=...
```

Print the resolved configuration first and check the tasks, lanes, attempts,
and concurrency:

```bash
harbor run --config configs/harbor-mini-swe.yaml --env-file .env --print-config
```

Then run it from the root of this directory:

```bash
harbor run --config configs/harbor-mini-swe.yaml --env-file .env
```

Results land under `jobs/harbor-mini-swe/` in the same layout as the completed
job above. Give a rerun a new name with `--job-name`. The first run builds each
task image on Modal, which can take a while on a cold cache; sandboxes are
deleted automatically when each trial finishes.

Keep this directory private. It contains verifier logic, held-out task
construction details, reference solutions, and complete agent trajectories.
