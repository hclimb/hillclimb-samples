"""Shared utilities for evaluators (document embedding and generation helpers)."""

import re
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tqdm import tqdm
from jax.sharding import PartitionSpec as P

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)


def global_device_put(arr, mesh, spec):
    """Place a host-replicated numpy array as a GLOBAL array on a (possibly multi-process) mesh.

    `jax.device_put(jnp.array(x), P(...))` breaks on a multi-host mesh: under the ambient
    jax.set_mesh, jnp.array materializes a mesh-wide (non-addressable) array first, which
    device_put then refuses to reshard ("must be a fully addressable array").
    make_array_from_callback only ever asks for the local shards. Every process must pass
    an identical `arr`.
    """
    arr = np.asarray(arr)
    # Match jnp.array's default canonicalization (x64 disabled) — 64-bit host arrays would
    # otherwise land on TPU as x64 and fail.
    if arr.dtype == np.int64:
        arr = arr.astype(np.int32)
    elif arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def split_thinking(text: str) -> tuple[str, str]:
    """Split a generated string into (thinking, answer).

    Returns:
        thinking: content inside the first <think>...</think> block, or "".
        answer:   everything after that block (stripped), or "" if
                  no think block is present.
    """
    m = _THINK_RE.match(text)
    if m:
        return m.group(1), text[m.end():].strip()
    return "", ""


def tokenize_documents_chunked(tokenizer, texts, chunk_size, max_chunks_per_doc):
    """Tokenize each document, split into chunk_size-token chunks, take up to
    max_chunks_per_doc chunks. Returns right-padded (input_ids, pad_mask) over
    all chunks, shape (total_chunks, chunk_size)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_ids, all_masks = [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="np", truncation=False)["input_ids"][0]
        chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
        chunks = chunks[:max_chunks_per_doc]
        for chunk in chunks:
            pad_len = chunk_size - len(chunk)
            all_ids.append(np.concatenate([chunk, np.full(pad_len, pad_id, dtype=chunk.dtype)]))
            all_masks.append(np.concatenate([np.ones(len(chunk), dtype=np.bool_), np.zeros(pad_len, dtype=np.bool_)]))
    return np.stack(all_ids), np.stack(all_masks)


def embed_documents(embed_fn, tokenizer, doc_texts, doc_seq_len, doc_batch_size, max_chunks_per_doc=4, doc_chunk_size=256):
    """Embed all documents in batches, return concatenated (mem_k, mem_v, mem_mask)."""
    all_k, all_v, all_mask = [], [], []
    n = len(doc_texts)

    for start in tqdm(range(0, n, doc_batch_size), desc="Embedding docs"):
        end = min(start + doc_batch_size, n)
        batch_texts = doc_texts[start:end]

        ids, mask = tokenize_documents_chunked(tokenizer, batch_texts, doc_chunk_size, max_chunks_per_doc)

        fsdp = jax.device_count()
        actual_b = ids.shape[0]
        if actual_b % fsdp != 0:
            pad_b = fsdp - (actual_b % fsdp)
            ids = np.concatenate([ids, np.zeros((pad_b, ids.shape[1]), dtype=ids.dtype)])
            mask = np.concatenate([mask, np.zeros((pad_b, mask.shape[1]), dtype=mask.dtype)])

        ids_jax = jax.device_put(jnp.array(ids), P("data", None))
        mask_jax = jax.device_put(jnp.array(mask), P("data", None))

        mem_k, mem_v, mem_mask_flat, _ = embed_fn(ids_jax, mask_jax)

        mem_k_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_k, tiled=True))
        mem_v_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_v, tiled=True))
        mem_mask_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_mask_flat, tiled=True))

        vecs_per_chunk = mem_k_np.shape[0] // ids.shape[0]
        real_vecs = actual_b * vecs_per_chunk
        all_k.append(mem_k_np[:real_vecs])
        all_v.append(mem_v_np[:real_vecs])
        all_mask.append(mem_mask_np[:real_vecs])

    return np.concatenate(all_k), np.concatenate(all_v), np.concatenate(all_mask)


def embed_documents_tokenized(embed_fn, doc_ids, doc_masks, doc_batch_size, mesh=None):
    """Embed pre-tokenized documents in batches. doc_ids/doc_masks are numpy arrays
    of shape (N, seq_len). Returns concatenated (mem_k, mem_v, mem_mask).

    Pass `mesh` on a multi-host mesh — the default bare-P device_put path is single-host
    only (see global_device_put)."""
    all_k, all_v, all_mask = [], [], []
    n = doc_ids.shape[0]
    fsdp = jax.device_count()

    for start in tqdm(range(0, n, doc_batch_size), desc="Embedding docs"):
        end = min(start + doc_batch_size, n)
        ids = doc_ids[start:end]
        mask = doc_masks[start:end]

        actual_b = ids.shape[0]
        if actual_b % fsdp != 0:
            pad_b = fsdp - (actual_b % fsdp)
            ids = np.concatenate([ids, np.zeros((pad_b, ids.shape[1]), dtype=ids.dtype)])
            mask = np.concatenate([mask, np.zeros((pad_b, mask.shape[1]), dtype=mask.dtype)])

        if mesh is not None:
            ids_jax = global_device_put(ids, mesh, P("data", None))
            mask_jax = global_device_put(mask.astype(np.bool_), mesh, P("data", None))
        else:
            ids_jax = jax.device_put(jnp.array(ids), P("data", None))
            mask_jax = jax.device_put(jnp.array(mask.astype(np.bool_)), P("data", None))

        mem_k, mem_v, mem_mask_flat, _ = embed_fn(ids_jax, mask_jax)

        mem_k_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_k, tiled=True))
        mem_v_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_v, tiled=True))
        mem_mask_np = np.array(jax.experimental.multihost_utils.process_allgather(mem_mask_flat, tiled=True))

        vecs_per_chunk = mem_k_np.shape[0] // ids.shape[0]
        real_vecs = actual_b * vecs_per_chunk
        all_k.append(mem_k_np[:real_vecs])
        all_v.append(mem_v_np[:real_vecs])
        all_mask.append(mem_mask_np[:real_vecs])

    return np.concatenate(all_k), np.concatenate(all_v), np.concatenate(all_mask)


def tokenize_queries_batch(tokenizer, queries, max_len=512):
    """Left-pad a batch of queries with chat template. Returns (input_ids, pad_mask)."""
    encoded = []
    for q in queries:
        msgs = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="np", truncation=True, max_length=max_len)["input_ids"][0]
        encoded.append(ids)

    max_tok_len = max(len(t) for t in encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    padded, masks = [], []
    for ids in encoded:
        pad_len = max_tok_len - len(ids)
        padded.append(np.concatenate([np.full(pad_len, pad_id, dtype=ids.dtype), ids]))
        masks.append(np.concatenate([np.zeros(pad_len, dtype=np.bool_), np.ones(len(ids), dtype=np.bool_)]))

    return np.stack(padded), np.stack(masks)