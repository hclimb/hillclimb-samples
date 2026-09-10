# Datagen TPU Orchestrator

This folder contains the complete robust TPU execution pipeline for offline synthetic data generation. It handles preemption and internet disconnects across Google Cloud QueuedResources seamlessly.

## Abstraction
The Orchestrator is **fully language/script agnostic**. It does not care if you want to run `answer_all_questions.py` or some future script `generate_qwen_qas.py`. 

It simply provisions a Google QueuedResource, waits until the SSH daemon is responsive, runs `--setup-script` ONCE to install dependencies, and then iterates through your `--parquets` list by interpolating `{CHUNKS}` into your `--run-command`.

## Target Script Requirements (The Contract)
For **any future Python script** to magically correctly resume and scale under this Orchestrator, it MUST obey these three strict rules:

1. **Accept the `--parquets` parameter**: The Orchestrator controls the queue mapping by interpolating `{CHUNKS}` into your command. Your script must read that string (e.g., `5,6,10`) and strictly process exactly those files/sections.
2. **Be Idempotent (Preemption Save-States)**: The Orchestrator assumes your script occasionally gets murdered violently via hardware preemption (Exit Code 255). 
   - **Bad**: Your script processes 5 hours of a chunk, gets killed, and blind-restarts doing the exact same 5 hours again when the Orchestrator feeds it the chunk.
   - **Good**: Your script regularly pushes small "I finished up to row X" files (like our `state_chunk_{uuid}.json`) to a central location (HuggingFace, GCS). Upon boot, it checks that location for its `{CHUNKS}` ID, fast-forwards its inner loop past the work already saved, and *only* computes the remaining remainder.
3. **Exit `0` on Success**: The Orchestrator permanently removes `{CHUNKS}` from `pending` -> `completed` ONLY when your script cleanly terminates with exit code `0`. If your script crashes internally for non-hardware reasons (Code 1), the Orchestrator safely marks it `failed` and will attempt moving on.

## Setup Memory Layers

You should edit the injected initialization script in `setup_memory_layers.sh` to control what `git clone` or `apt-get` commands are run when a *brand new TPU* is spawned. The Orchestrator safely marks the TPU as `.initialized` so this only runs once per machine!

## How To Run

Execute the entrypoint on your local machine targeting a single TPU instance. 
Run it via `tmux` or `nohup` if you want it to run indefinitely over a 3-day job.

```bash
uv run orchestrator.py \
  --project YOUR-PROJECT \
  --zone us-east5-c \
  --tpu-name my-datagen-session-1 \
  --tpu-type v6e-8 \
  --parquets "0-100" \
  --chunk-size 1 \
  --state-file session-1-queue.json \
  --setup-script setup_memory_layers.sh \
  --run-command "uv run datagen/retry_setup/answer_all_questions.py --parquet-numbers '{CHUNKS}'"
```

**Note:** The setup script automatically creates `.env` on the worker chip using your local `.env` definition via `--env-file`.
