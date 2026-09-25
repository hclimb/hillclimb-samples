# Train a small HotpotQA retriever

Use only the supplied plain pretrained Google BERT-Tiny and training data.
`train_retriever.sh` receives absolute `--train_dir`, `--models_dir`,
`--output_dir` paths and `--seed`. Write `model.safetensors` in the output
directory. The file contains the exact `BertModel(add_pooling_layer=False)`
state dict, without a `bert.` prefix, pooler, classifier, or extra tensors.
All encoder weights may change. Float32, float16 and bfloat16 tensors are
accepted; the fixed architecture, tokenizer and ranking implementation do not change.
The initial checkpoint is pretrained, not retrieval-trained.

## Local commands

From `/environment/starter`:

```
python /environment/starter/optifine_public_tests/run.py --submission /environment/starter/train_retriever.sh --output /environment/starter/runs/public
python /environment/starter/optifine_public_tests/run.py --submission /environment/starter/train_retriever.sh --output /environment/starter/runs/smoke --smoke
python /environment/starter/optifine_public_tests/run.py --check-weights /environment/starter/runs/manual/model.safetensors --output /environment/starter/runs/validity
python /environment/starter/optifine_public_tests/run.py --submission /environment/starter/train_retriever.sh --output /environment/starter/runs/paired --paired
```

The first command freshly trains the frozen starter and your current recipe on the complete public panel.
The smoke command passes `--steps 2` and gives training 60 seconds, then checks
tensor loading and a CPU forward pass. Support this optional flag in your script,
or use the weights check on a checkpoint you trained manually. These checks report
no quality score. Every full quality run is paired; `--paired` remains accepted.
The full paths give each recipe 900 seconds for training and 600 for retrieval,
including preparation, tokenization, loading and sorting. Two H200 GPUs are available;
the starter trains with distributed data parallelism (synchronized gradient updates).
The small fixed inference path uses one GPU; the second remains available, not required.
No network is needed or available during these runs. All candidate-specific mining
and tokenization count toward training time. `--seed` supports repeatability studies;
the standard seed is 1729. Source is snapshotted when each test starts.

Every public command automatically saves a source checkpoint and retains the exact
`model.safetensors` evaluated by trusted inference, when training produces one.
The weights check retains the supplied model. Each run has a unique directory in
`/logs/artifacts/public-verifier/<run-id>` containing weights, hashes and reports.
`checkpoint.json` in your output directory links that archive and the normal
`/logs/artifacts/progress` source checkpoint. Reports are copied to your output
when execution finishes; reusing that path preserves earlier archives and failures.
Archival I/O is outside training/retrieval limits; existing preparation still counts.
The private run retrains the submitted recipe as before.

## Data and fixed retrieval

`optifine_public_tests/assets/hotpotqa/train` contains 16,384 questions and
200,000 paragraphs. Each question has two `pos_doc_ids` and ten `context_doc_ids`;
IDs index rows of `corpus.jsonl`. All ten context paragraphs remain available for
training. Background sampling excludes grouped evaluation questions and support
pairs. The public benchmark searches all 60,000 paragraphs for each of its 3,072
questions, not just the ten original contexts. All public questions are hard:
2,467 bridge questions and 605 comparison questions.

HotpotQA comes from `hotpotqa/hotpot_qa` at
`1908d6afbbead072334abe2965f91bd2709910ab`, CC-BY-SA-4.0
(https://creativecommons.org/licenses/by-sa/4.0/).
Paragraph text is `title + ': ' + ' '.join(sentences)`. Only identical text is
deduplicated. Manifests pin selected IDs, independent shuffling seeds and file hashes.
Training/public corpora overlap by 24,389 exact paragraphs. This is not a test of
unseen-document generalization. Public labels support diagnosis and model selection;
train the final recipe only on the designated training split.

The only model asset is `google/bert_uncased_L-2_H-128_A-2` at
`30b0a37ccaaa32f332884b96992754e246e48c5f` (Apache-2.0).
The original config and vocabulary are pinned. Trusted `BertTokenizerFast` uses
uncased WordPiece, right padding and normal BERT special tokens, with truncation
at 64 query tokens and 256 paragraph tokens. The encoder has two layers, width
128, two attention heads and intermediate width 512. Inference uses bfloat16
autocast, float32 masked mean pooling including special tokens, and L2 normalization.
There is no trainable head, lexical ranker, title graph or second-hop pass.
All document/query cosine scores are computed in float32; descending stable sorting
breaks ties by corpus row ID. The ten best documents form the output. All corpus
encoding is fresh for every run. Tokenization, pooling and ranking are not editable.

## Interpreting results

Binary nDCG@10 (normalized discounted cumulative gain) discounts a support at rank
`r` by `1/log2(r+1)` and divides total gain by `1 + 1/log2(3)`.
Both supports at ranks one and two give exactly 1. The tool reports mean nDCG,
both-supports@2, bridge/comparison slices, per-question ranks and actual phase times.
For a valid submission, progress = (C - B) / (0.715285038072705 - B), where C is candidate mean nDCG@10 and B is the frozen starter remeasured on the same panel. Reward = clip((progress - 0.01) / 0.98, 0, 1). The 1% margin is a fraction of the starter-to-best gap at each end: progress up to 0.01 scores 0 and progress from 0.99 scores 1. Invalid submissions score 0. Every non-smoke quality run measures the starter; --paired remains accepted.
Raw nDCG remains available. Infrastructure failures abort without a score. The untouched pretrained encoder remains a separate no-training control.

The starter uses multi-positive contrastive loss: both supports are encouraged,
with context negatives plus other questions' documents as negatives. Improve its
loss, sampling, mining, curriculum, optimization or implementation. Do not use other
pretrained models, teacher predictions or retrieval-trained weights. Training code
is retained, but environments, assets, benchmark tools and run outputs are excluded
from source transfer. Keep every required helper in ordinary source directories.
Only the resulting tensor state is used by fresh retrieval; submitted Python and
predictions are never imported by the retrieval process. This boundary does not
prove learning or prevent memorization; robust learning must be established empirically.
