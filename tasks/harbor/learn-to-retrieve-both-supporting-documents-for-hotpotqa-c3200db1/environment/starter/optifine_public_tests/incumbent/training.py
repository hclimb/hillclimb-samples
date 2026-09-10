import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from transformers import BertModel, BertTokenizerFast


def read_rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def encode(model, tokens):
    hidden = model(**tokens).last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1)
    return functional.normalize((hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    for name in ('train_dir', 'models_dir', 'output_dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--recipe', choices=['starter', 'control'], default='starter')
    parser.add_argument('--steps', type=int, default=1200)
    args = parser.parse_args()
    started = time.monotonic()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed + rank)
    distributed.init_process_group('nccl')
    model_dir = args.models_dir / 'bert-tiny'
    model = BertModel.from_pretrained(model_dir, add_pooling_layer=False, local_files_only=True).cuda()
    if args.recipe == 'control':
        if rank == 0:
            save_file({name: value.cpu().contiguous() for name, value in model.state_dict().items()},
                      str(args.output_dir / 'model.safetensors'))
        distributed.destroy_process_group()
        return
    tokenize = BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt'), do_lower_case=True)
    corpus = read_rows(args.train_dir / 'corpus.jsonl')
    questions = read_rows(args.train_dir / 'questions.jsonl')
    if args.steps <= 2:
        questions = questions[:128]
        used = sorted({index for row in questions for index in row['context_doc_ids']})
        corpus = [corpus[index] for index in used]
        lookup = {index: local for local, index in enumerate(used)}
        questions = [dict(row, **{key: [lookup[index] for index in row[key]]
                                 for key in ('pos_doc_ids', 'context_doc_ids')}) for row in questions]

    def prepare(texts, length):
        blocks = [tokenize(texts[start:start + 4096], padding='max_length', truncation=True,
                           max_length=length, return_tensors='pt')
                  for start in range(0, len(texts), 4096)]
        return {name: torch.cat([block[name] for block in blocks]).to(device=rank, dtype=torch.int32)
                for name in blocks[0]}

    doc_tokens = prepare([row['text'] for row in corpus], 256)
    query_tokens = prepare([row['question'] for row in questions], 64)
    positives = torch.tensor([row['pos_doc_ids'] for row in questions], device=rank)
    negatives = [[index for index in row['context_doc_ids'] if index not in row['pos_doc_ids']]
                 for row in questions]
    model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator(device=rank).manual_seed(args.seed + rank)
    batch_size, temperature = 64, 0.05

    def take(tokens, indices):
        return {name: values[indices].long() for name, values in tokens.items()}

    for step in range(args.steps):
        selected = torch.randint(len(questions), (batch_size,), device=rank, generator=generator)
        selected_list = selected.tolist()
        negative_ids = []
        for index in selected_list:
            pool = negatives[index]
            negative_ids.append(random.sample(pool, 4))
        doc_ids = torch.cat([positives[selected], torch.tensor(negative_ids, device=rank)], dim=1).flatten()
        positive_mask = (doc_ids[None, :, None] == positives[selected, None, :]).any(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vectors = encode(model, take(doc_tokens, doc_ids))
            query_vectors = encode(model, take(query_tokens, selected))
            logits = (query_vectors @ vectors.T).float() / temperature
            log_probs = functional.log_softmax(logits, dim=1)
            loss = -(log_probs * positive_mask).sum(1).div(positive_mask.sum(1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rate = min((step + 1) / 100, max(0.1, 1 - step / args.steps))
        for group in optimizer.param_groups:
            group['lr'] = 3e-4 * rate
        optimizer.step()
        if rank == 0 and step % 100 == 0:
            print(json.dumps(dict(step=step, loss=loss.item(), seconds=time.monotonic() - started)), flush=True)
    if rank == 0:
        save_file({name: value.detach().float().cpu().contiguous()
                   for name, value in model.module.state_dict().items()}, str(args.output_dir / 'model.safetensors'))
        print(json.dumps(dict(steps=args.steps, seconds=time.monotonic() - started)), flush=True)
    distributed.destroy_process_group()


if __name__ == '__main__':
    main()
