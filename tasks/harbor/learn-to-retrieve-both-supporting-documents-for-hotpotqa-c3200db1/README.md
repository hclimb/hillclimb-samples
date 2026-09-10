# Learn HotpotQA Retrieval with BERT-Tiny

Hardware: two H200. Agent budget: six hours (21,600 seconds). Verifier budget: one hour, run afterwards in a separate offline sandbox.

## Task

Train a plain pretrained BERT-Tiny (two layers, width 128, about 4.4 million parameters) so that it ranks both supporting paragraphs of a HotpotQA question near the top of a 60,000-paragraph corpus. The agent improves the training recipe in `/environment/starter/train_retriever.sh` and `/environment/starter/training.py`, subject to a 900-second training budget on the two GPUs, and saves the encoder weights as `model.safetensors`.

Tokenization, mean pooling, and cosine retrieval are fixed. Only the encoder weights are submitted; the trusted evaluator reloads them and ranks the whole corpus itself.

## Why it is hard

Multi-hop questions need evidence from two documents, including bridge paragraphs that the question never names. The small encoder must learn entity and complementary-evidence matching across the full corpus, not just the question's original distractor bundle. Hard-negative mining, false-negative handling, loss design, and efficient two-GPU training all have to fit inside the fixed 15-minute budget.

The starter uses distributed multi-positive contrastive training with the two supports, four original context negatives, and in-batch negatives.

## Public testing

```
python /environment/starter/optifine_public_tests/run.py \
  --submission /environment/starter/train_retriever.sh \
  --output /environment/starter/runs/public
```

This trains with the submission, runs retrieval for 3,072 public questions, and reports nDCG@10. `--smoke` trains two steps and validates the output tensors without a quality claim. `/environment/starter/RETRIEVAL.md` documents the training interface and quick checks.

## Scoring

The verifier trains the submitted recipe once in an unprivileged process, loads the resulting encoder into a fresh trusted interpreter, and ranks every paragraph in the private panel by cosine similarity. Retrieval over 3,072 questions must finish within 600 seconds.

For each question, `DCG = sum over ranks r = 1..10 of 1[doc_r is gold] / log2(r + 1)` and `nDCG = DCG / (1 + 1 / log2(3))`. With C the mean nDCG@10, the reward is `(C - 0.446) / (1 - 0.446)`. The untouched starter scores about 0; a perfect ranking scores 1; a candidate below the anchor receives a negative reward. Training that exceeds 900 seconds, emits anything other than the fixed encoder tensors, or produces non-finite values yields reward 0.
