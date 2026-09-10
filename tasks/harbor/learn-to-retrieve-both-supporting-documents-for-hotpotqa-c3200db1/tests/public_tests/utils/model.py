import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file
from transformers import BertConfig, BertModel, BertTokenizerFast


MODEL_REVISION = '30b0a37ccaaa32f332884b96992754e246e48c5f'
DOC_LENGTH = 256
QUERY_LENGTH = 64


class InvalidWeights(ValueError):
    pass


def tokenizer(model_dir):
    return BertTokenizerFast(vocab_file=str(Path(model_dir) / 'vocab.txt'), do_lower_case=True)


def architecture(model_dir):
    config = BertConfig.from_json_file(str(Path(model_dir) / 'config.json'))
    assert (config.vocab_size, config.hidden_size, config.num_hidden_layers,
            config.num_attention_heads, config.intermediate_size) == (30522, 128, 2, 2, 512)
    return BertModel(config, add_pooling_layer=False)


def load_weights(model_dir, checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size > 32_000_000:
        raise InvalidWeights('Expected a regular model.safetensors file below 32 MB')
    model = architecture(model_dir)
    expected = model.state_dict()
    try:
        with safe_open(checkpoint, framework='pt', device='cpu') as handle:
            if set(handle.keys()) != set(expected):
                raise InvalidWeights('Tensor keys differ from the fixed BERT encoder (no pooler)')
            for name, value in expected.items():
                tensor = handle.get_tensor(name)
                if tensor.shape != value.shape or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    raise InvalidWeights(f'Invalid shape or dtype: {name}')
                if not torch.isfinite(tensor).all():
                    raise InvalidWeights(f'Nonfinite weights: {name}')
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
    except SafetensorError as error:
        raise InvalidWeights(str(error)) from error
    return model


def encode(model, tokens):
    hidden = model(**tokens).last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1)
    pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
    return functional.normalize(pooled, dim=-1)


def rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]
