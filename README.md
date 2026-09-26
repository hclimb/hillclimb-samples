# Hillclimb samples

Two Harbor tasks for open-ended ML and ML-systems optimization, the job
configuration that runs them, and per-trial metrics from our 1- and 3-hour
rollouts across seven agents.

## Tasks

| Task | Hardware | Objective |
|---|---|---|
| [`learn-to-retrieve-both-supporting-documents-for-hotpotqa-c3200db1`](tasks/harbor/learn-to-retrieve-both-supporting-documents-for-hotpotqa-c3200db1/) | 2 H200 | Improve the training recipe for a BERT-Tiny retriever so it ranks both HotpotQA supporting paragraphs higher, within a 900-second training budget. Measured by nDCG@10 over 3,072 held-out questions and 60,000 paragraphs. |
| [`native-model1-paged-sparse-attention-b200`](tasks/harbor/native-model1-paged-sparse-attention-b200/) | 1 B200 | Speed up a FlashMLA FP8 paged sparse-attention kernel on one fixed workload while all original correctness cases keep passing. Measured by the scored call's throughput relative to the unmodified starter. |

## Cloning

Model weights, the HotpotQA corpus, compiled kernels, and wheels are stored with
Git LFS (about 0.5 GB). Install Git LFS before cloning, or the clone contains
small pointer files in their place and the task images will not build:

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

Install Harbor with the Modal extra and authenticate to your Modal workspace:

```bash
uv tool install 'harbor[modal]==0.23.0' --with-executables-from modal
```

```bash
modal setup
```

Put your OpenRouter key in a `.env` file at the root of this directory:

```
OPENROUTER_API_KEY=...
```

Then run it from the root of this directory:

```bash
harbor run --config configs/harbor-mini-swe.yaml --env-file .env
```
