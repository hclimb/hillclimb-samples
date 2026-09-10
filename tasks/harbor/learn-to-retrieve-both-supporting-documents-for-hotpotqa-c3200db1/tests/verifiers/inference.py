import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from utils.model import DOC_LENGTH, QUERY_LENGTH, InvalidWeights, encode, load_weights, rows, tokenizer


def main():
    parser = argparse.ArgumentParser()
    for name in ('models', 'weights', 'corpus', 'queries', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--validity', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    started = time.monotonic()
    model = load_weights(args.models, args.weights).eval()
    if args.validity:
        tokens = tokenizer(args.models)(['Where was the author born?'], return_tensors='pt')
        with torch.inference_mode():
            vector = encode(model, tokens)
        if not torch.isfinite(vector).all() or vector.norm() < 0.99:
            raise InvalidWeights('Invalid encoder output')
        args.output.write_text(json.dumps(dict(valid=1, seconds=time.monotonic() - started)))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Full retrieval requires CUDA')
    model = model.cuda()
    corpus, queries = rows(args.corpus), rows(args.queries)
    if len(corpus) != 60000 or len(queries) != 3072:
        raise ValueError('Wrong full panel size')
    if any(row['id'] != index or set(row) != {'id', 'text'} for index, row in enumerate(corpus)):
        raise ValueError('Invalid corpus rows')
    if any(set(row) != {'id', 'question'} for row in queries):
        raise ValueError('Queries must be label-free')
    tokenize = tokenizer(args.models)

    @torch.inference_mode()
    def vectors(texts, length):
        output = []
        for start in range(0, len(texts), 512):
            tokens = tokenize(texts[start:start + 512], padding=True, truncation=True,
                              max_length=length, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                embeddings = encode(model, tokens)
            if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=-1) < 0.99).any():
                raise InvalidWeights('Nonfinite or zero embeddings')
            output.append(embeddings)
        return torch.cat(output)

    docs = vectors([row['text'] for row in corpus], DOC_LENGTH)
    questions = vectors([row['question'] for row in queries], QUERY_LENGTH)
    results = []
    for start in range(0, len(queries), 128):
        similarities = questions[start:start + 128] @ docs.T
        ranked = torch.argsort(similarities, dim=1, descending=True, stable=True)[:, :10].cpu().tolist()
        for query, indices in zip(queries[start:start + 128], ranked):
            results.append(dict(query_id=query['id'], query=query['question'],
                                retrieved=[dict(doc_index=index) for index in indices]))
    torch.cuda.synchronize()
    args.output.write_text(json.dumps(results))
    print(json.dumps(dict(retrieval_seconds=time.monotonic() - started,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated())), flush=True)


if __name__ == '__main__':
    try:
        main()
    except InvalidWeights:
        traceback.print_exc()
        sys.exit(2)
