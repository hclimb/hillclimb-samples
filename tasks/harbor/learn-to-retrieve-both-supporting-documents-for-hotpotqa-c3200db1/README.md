# Learn HotpotQA Retrieval with BERT-Tiny

## Task

Improve the training recipe in `/environment/starter/train_retriever.sh` and `training.py` so one supplied BERT-Tiny ranks both supporting paragraphs of each HotpotQA question higher. Train offline on two H200 GPUs using 16,384 training questions and a 200,000-paragraph corpus. Write `model.safetensors` to `--output_dir` within 900 seconds, including source snapshotting, tokenization and negative mining.

Trusted retrieval has 600 seconds to rank a 60,000-paragraph panel for 3,072 questions. Architecture, tokenization, pooling and ranking are fixed; the submitted recipe changes the encoder weights. The solver and private verifier each have a 3,600-second limit.

## Background

BERT-Tiny has two layers and width 128. It encodes queries truncated at 64 tokens and paragraphs at 256. Masked mean pooling and L2 normalization produce unit vectors; cosine ranks the full panel, breaking ties by corpus row ID.

Each question has two Wikipedia supports. Bridge questions require evidence not necessarily named in the question; comparison questions concern two entities. Binary nDCG@10 discounts each support by `1/log2(rank + 1)` and divides total gain by `1 + 1/log2(3)`.

## Difficulty

The small encoder must recover evidence from tens of thousands of paragraphs. The starter takes roughly two minutes; the winning recipe uses most of the training budget. Training efficiency, negative selection and the objective can all matter.

Training and evaluation corpora overlap: 24,389 exact paragraphs with the public panel and 24,117 with the private panel. This is not a test of unseen-document generalization. A weights-only inference boundary does not prove learning or prevent memorization. Capacity, solver difficulty and research headroom still require qualification; historical repeated-seed gains do not establish them.

## Optimization directions

- Mine hard negatives with the model being trained, while controlling false negatives that share supporting evidence.
- Improve the multi-positive loss, shared-support handling, or the balance between direct and bridge evidence.
- Tune learning rate, temperature, batch size and the training schedule together with the mining schedule.
- Improve tokenization and two-GPU throughput within the same total training budget.

Edit the launcher, training code and ordinary helper source files. Public labels support diagnosis and model selection, but the final recipe trains only on the designated training split. Other pretrained models, teacher predictions and retrieval-trained starting weights are not allowed.

## Solution

`solution/solve.sh` installs the exact `training.py` from the winning Fable/Claude Code checkpoint `gLoDw6S`. It uses periodic corpus hard-negative mining and multi-positive contrastive training over a longer schedule. The archive contains source only, with no trained output weights; hashes are recorded in `solution/source-hashes.json`.

The recorded private nDCG@10 of 0.715285038072705 sets the fixed upper anchor. Before the margin was added, a fresh reference rerun achieved 0.713960, corresponding to a score of 0.994790. The 1% endpoint margin maps that measurement to 1 without changing its raw nDCG.

Fresh private-verifier controls with the 1% endpoint margin:

| Submission | nDCG@10 | Reward | Valid |
|---|---:|---:|---|
| Unchanged starter | 0.461002 | 0.000000 | Yes |
| Winning reference (`solve.sh`) | 0.713939 | 1.000000 | Yes |

Raw metrics, paired baselines, trial paths and tested task hashes are recorded in [the calibration record](tests/calibration/timing.json). The upper anchor remains fixed to the historical winning submission; it is not refit to these checks.

Earlier calibration measured public nDCG of 0.187 before training and 0.447 for the starter. An older reference added self-mined negatives, trained for roughly two minutes and measured private nDCG of 0.515294 and 0.515868. These are historical raw metrics. Full one-hour solver runs and automated release qualification under this scoring remain pending.

## Verification

The evaluator snapshots the submitted source, checks packaged assets and two CUDA GPUs within 30 seconds, then trains the frozen starter and candidate offline as an unprivileged process. Training and retrieval caches do not carry between runs. Each result is scored by a fresh trusted interpreter that loads the tensor state and never imports submitted Python; the evaluator still executes the submitted recipe during training.

Run the public paired comparison from the starter:

```sh
python /environment/starter/optifine_public_tests/run.py \
  --submission /environment/starter/train_retriever.sh \
  --output /environment/starter/runs/public
```

Every quality run retrains the starter; `--paired` remains accepted. `--smoke` trains two steps under 60-second phase limits without a quality score. Public runs retain source, tested weights and hashed reports under `/logs/artifacts/public-verifier/<run-id>`, copying results to the requested output. Private evaluation retrains submitted source and writes `/logs/verifier/reward.json`.

See [the training interface and public commands](environment/starter/RETRIEVAL.md) for the complete contract.

Executable checks:

- **Valid weights:** training must finish within its budget and produce a regular, non-symlinked `model.safetensors` no larger than 32,000,000 bytes. Tensor keys, shapes and floating dtypes must match the fixed encoder; values and trusted inference outputs must be finite. Invalid submissions score 0; infrastructure failures abort without a score.
- **Retrieval quality:** trusted inference computes mean nDCG@10 against the two annotated supports. Both-supports@2 and bridge/comparison slices remain diagnostics.

## Scoring

```text
C = candidate mean nDCG@10
B = starter mean nDCG@10, freshly measured on the same panel
T = 0.715285038072705
progress = (C - B) / (T - B)
reward = clip((progress - 0.01) / 0.98, 0, 1)
```

The margin is 1% of the starter-to-best gap at each end. Progress up to 0.01 scores 0; progress from 0.99 scores 1. The midpoint still scores 0.5. Raw nDCG remains available, and fresh training can still vary; the margin is a tolerance, not a guarantee against every fluctuation. There is no Elo or KDE transform.

The frozen starter is unchanged. Its freshly measured nDCG supplies B.
