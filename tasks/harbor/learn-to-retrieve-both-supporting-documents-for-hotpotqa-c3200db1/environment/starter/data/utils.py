import numpy as np


def chunk_text(tokenizer, text, doc_chunk_seq_len):
    """Tokenize text and split into fixed-size chunks of doc_chunk_seq_len."""
    if not text:
        return [np.full((doc_chunk_seq_len,), tokenizer.pad_token_id or 0, dtype=np.int32)]
    full_text = text + (tokenizer.eos_token or "<|im_end|>")
    tokens = tokenizer(full_text, truncation=False, return_tensors="np")["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), doc_chunk_seq_len):
        chunk = tokens[i:i + doc_chunk_seq_len]
        if len(chunk) < doc_chunk_seq_len:
            chunk = np.pad(chunk, (0, doc_chunk_seq_len - len(chunk)),
                           constant_values=tokenizer.pad_token_id or 0)
        chunks.append(chunk)
    return chunks or [np.full((doc_chunk_seq_len,), tokenizer.pad_token_id or 0, dtype=np.int32)]


def pack_docs(tokenizer, pos_docs_raw, neg_docs_raw, num_chunks_per_doc, doc_chunk_seq_len):
    """Chunk and pack pos/neg docs into fixed-shape arrays.

    Returns:
        final_doc_chunks: (num_chunks_per_doc, doc_chunk_seq_len) int32
        final_doc_masks:  (num_chunks_per_doc, doc_chunk_seq_len) float32
        final_pos_mask:   (num_chunks_per_doc,) int32  — 1 for pos, 0 for neg/pad
    """
    doc_chunks, pos_mask = [], []
    for doc in pos_docs_raw:
        if not doc: continue
        for c in chunk_text(tokenizer, doc, doc_chunk_seq_len):
            doc_chunks.append(c); pos_mask.append(1)
    for doc in neg_docs_raw:
        if not doc: continue
        for c in chunk_text(tokenizer, doc, doc_chunk_seq_len):
            doc_chunks.append(c); pos_mask.append(0)

    doc_chunks = doc_chunks[:num_chunks_per_doc]
    pos_mask   = pos_mask[:num_chunks_per_doc]

    pad_val = tokenizer.pad_token_id or 0
    final_doc_chunks = np.full((num_chunks_per_doc, doc_chunk_seq_len), pad_val, dtype=np.int32)
    final_doc_masks  = np.zeros((num_chunks_per_doc, doc_chunk_seq_len), dtype=np.float32)
    final_pos_mask   = np.zeros((num_chunks_per_doc,), dtype=np.int32)

    for i, chunk in enumerate(doc_chunks):
        final_doc_chunks[i] = chunk
        final_doc_masks[i]  = make_attn_mask(chunk, tokenizer.pad_token_id)
        final_pos_mask[i]   = pos_mask[i]

    return final_doc_chunks, final_doc_masks, final_pos_mask


def build_prefix_text(tokenizer, question, prompt_tmpl, chat_template, has_thinking):
    """Build the (unmasked) prefix string for a QA item.

    Handles three modes in priority order:
      1. chat_template  — apply_chat_template with enable_thinking
      2. prompt_tmpl    — format the template string (document="")
      3. plain          — bare "Question: … Answer: " fallback
    """
    if chat_template:
        content = prompt_tmpl.format(document="", question=question).rstrip("\n") if prompt_tmpl else question
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=has_thinking
        )
    elif prompt_tmpl:
        return prompt_tmpl.format(document="", question=question)
    else:
        return f"Question: {question}\nAnswer: "


def make_attn_mask(tokens, pad_token_id):
    """Float32 mask: 1 for real tokens, 0 for pad."""
    mask = np.ones_like(tokens, dtype=np.float32)
    if pad_token_id is not None:
        mask[tokens == pad_token_id] = 0.0
    return mask

def make_hotpotqa_normalizer():
    """Normalizer for hotpotqa/hotpot_qa (distractor config).

    Fields are top-level (not nested in metadata):
      - question: str
      - answer: str
      - context: {"title": [...], "sentences": [[sent, ...], ...]}
      - supporting_facts: {"title": [...], "sent_id": [...]}

    pos_doc = paragraphs whose title is in supporting_facts.title
    neg_doc = remaining paragraphs (distractors)
    """
    def normalize(item):
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context") or {}
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        sf = item.get("supporting_facts") or {}
        gold_titles = set(sf.get("title", []))

        pos_parts, neg_parts = [], []
        for title, sents in zip(titles, sentences_list):
            text = title + ": " + " ".join(sents) if isinstance(sents, list) else title
            if title in gold_titles:
                pos_parts.append(text)
            else:
                neg_parts.append(text)

        result = {
            "question": question,
            "answer": answer,
            "pos_doc": pos_parts,
            "neg_doc": neg_parts,
        }
        if "pos_doc_ids" in item:
            result["pos_doc_ids"] = item["pos_doc_ids"]
        return result
    return normalize


def make_normalizer(field_map, think_field, doc_separator=None, neg_score_threshold=None, min_neg_docs=0, mask_ce=False):
    def normalize(item):
        fm = field_map or {}

        # pos_doc: split on doc_separator if str, else pass list, else wrap scalar
        raw_pos_doc = item.get(fm.get("pos_doc", "pos_doc"), "")
        if doc_separator and isinstance(raw_pos_doc, str):
            pos_doc = [d.strip() for d in raw_pos_doc.split(doc_separator) if d.strip()]
        elif isinstance(raw_pos_doc, list):
            pos_doc = raw_pos_doc
        else:
            pos_doc = [raw_pos_doc] if raw_pos_doc else []

        # neg_doc: same treatment (pipe-separated in hard-neg sources, list elsewhere)
        raw_neg_doc = item.get(fm.get("neg_doc", "neg_doc"), [])
        if doc_separator and isinstance(raw_neg_doc, str):
            neg_doc = [d.strip() for d in raw_neg_doc.split(doc_separator) if d.strip()]
        elif isinstance(raw_neg_doc, list):
            neg_doc = raw_neg_doc
        else:
            neg_doc = [raw_neg_doc] if raw_neg_doc else []

        # neg_scores + threshold filter: only when threshold is set (hard-neg sources)
        if neg_score_threshold is not None:
            raw_scores = item.get("neg_scores", [])
            if doc_separator and isinstance(raw_scores, str):
                neg_scores = [float(s) for s in raw_scores.split(doc_separator) if s.strip()]
            elif isinstance(raw_scores, list):
                neg_scores = [float(s) for s in raw_scores]
            else:
                neg_scores = []
            neg_doc = [d for d, s in zip(neg_doc, neg_scores) if s <= neg_score_threshold]

        result = {
            "question": item.get(fm.get("question", "question"), ""),
            "answer":   item.get(fm.get("answer", "answer"), ""),
            "pos_doc":  pos_doc,
            "neg_doc":  neg_doc,
            "_min_neg_docs": min_neg_docs,   # per-source, read by _StreamingQAFilter
            # _ce_enable=0.0 masks the cross-entropy loss for this row (retrieval-only
            # supervision via doc_access_loss); 1.0 = normal CE. Used for text-similarity
            # sources where reproducing the memorized doc via CE would corrupt the LM.
            "_ce_enable": 0.0 if mask_ce else 1.0,
        }
        if think_field and item.get(think_field):
            result["answer"] = f"<think>\n\n{item[think_field]}</think>\n\n{result['answer']}"
        if "pos_doc_ids" in item:
            result["pos_doc_ids"] = item["pos_doc_ids"]
        return result
    return normalize


def make_flashrag_normalizer(context_key=None):
    """Normalizer for FlashRAG datasets (RUC-NLPIR/FlashRAG_datasets).

    Handles:
    - golden_answers (list) → answer (first non-empty element)
    - context_key="context" → pos_doc from metadata (e.g. hotpotqa)
      HotpotQA context format: [[title, [sent1, sent2, ...]], ...]
    - context_key="passages" → pos_doc from metadata.passages (e.g. msmarco-qa)
      MS MARCO format: metadata.passages = {passage_text: [...], is_selected: [...]}
      Selected passages (is_selected=1) → pos_doc; others → neg_doc
    """
    def normalize(item):
        question = item.get("question", "")
        golden_answers = item.get("golden_answers", [])
        if isinstance(golden_answers, list):
            answer = next((a for a in golden_answers if a), "")
        else:
            answer = str(golden_answers or "")

        pos_doc = []
        neg_doc = []
        if context_key == "passages":
            metadata = item.get("metadata") or {}
            passages_data = metadata.get("passages") or {}
            texts = passages_data.get("passage_text", [])
            selected = passages_data.get("is_selected", [])
            pos_docs = [t for t, s in zip(texts, selected) if s == 1]
            neg_doc = [t for t, s in zip(texts, selected) if s == 0]
            pos_doc = pos_docs if pos_docs else []
        elif context_key:
            metadata = item.get("metadata") or {}
            context = metadata.get(context_key)
            if context and isinstance(context, list):
                parts = []
                for entry in context:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        title, sentences = entry[0], entry[1]
                        text = title + ": " + " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                        parts.append(text)
                    else:
                        parts.append(str(entry))
                pos_doc = parts
            elif context:
                pos_doc = [str(context)]

        result = {
            "question": question,
            "answer": answer,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
        }
        if "pos_doc_ids" in item:
            result["pos_doc_ids"] = item["pos_doc_ids"]
        return result
    return normalize


# Per-process corpus cache, keyed by path. The mapped handle must never be reachable from a
# pickled function: grain ships the map fn to worker processes with cloudpickle, and a pyarrow
# MemoryMappedFile raises "no default __reduce__ due to non-trivial __cinit__". Note that
# module-level alone is NOT enough — cloudpickle pickles a *local* function by value and drags
# in every global it references, cache included. Hence the partial() below over a module-level
# function, which cloudpickle pickles by REFERENCE and so never walks these globals.
_DOCID_CORPUS_CACHE = {}


def _docid_text_column(corpus_path):
    """Open (once per process) the Arrow IPC corpus and return its text column."""
    col = _DOCID_CORPUS_CACHE.get(corpus_path)
    if col is None:
        import os
        import pyarrow as pa
        source = pa.memory_map(os.path.expanduser(corpus_path), "rb")
        col = pa.ipc.open_file(source).read_all().column("text")
        _DOCID_CORPUS_CACHE[corpus_path] = col
        _DOCID_CORPUS_CACHE[(corpus_path, "src")] = source   # keep the mapping alive
    return col


def _docid_normalize(item, corpus_path, think_field, max_neg_docs, min_neg_docs, mask_ce,
                     field_map, cot_field=None):
    """Module-level so functools.partial(...) pickles by reference. See make_docid_normalizer."""
    fm = field_map or {}
    col = _docid_text_column(corpus_path)
    n_docs = len(col)

    def resolve(ids):
        out = []
        for i in ids or []:
            i = int(i)
            if 0 <= i < n_docs:            # skip -1 padding / out-of-range ids
                out.append(col[i].as_py())
        return out

    neg_ids = item.get("neg_doc_ids") or []
    if max_neg_docs is not None:
        neg_ids = neg_ids[:max_neg_docs]

    result = {
        "question": item.get(fm.get("question", "question"), ""),
        "answer":   item.get(fm.get("answer", "answer"), ""),
        "pos_doc":  resolve(item.get("pos_doc_ids")),
        "neg_doc":  resolve(neg_ids),
        "_min_neg_docs": min_neg_docs,
        "_ce_enable": 0.0 if mask_ce else 1.0,
    }
    if think_field and item.get(think_field):
        result["answer"] = f"<think>\n\n{item[think_field]}</think>\n\n{result['answer']}"
    # cot_field is independent of think_field: it doesn't touch the answer/CE target, it just
    # forwards the raw CoT text for qa_transform_item to append as an extra positive doc in the
    # memory bank (see data/qa.py). Keeping the two separate means enabling one never silently
    # changes the other's behavior.
    if cot_field and item.get(cot_field):
        result["cot_doc"] = item[cot_field]
    if "pos_doc_ids" in item:
        result["pos_doc_ids"] = item["pos_doc_ids"]
    return result


def make_docid_normalizer(corpus_path, think_field=None, max_neg_docs=None,
                          min_neg_docs=0, mask_ce=False, field_map=None, cot_field=None):
    """Normalizer for id-based sources: resolve doc ids to text against a corpus file.

    Rows carry `pos_doc_ids` / `neg_doc_ids` (int lists) instead of document text. Storing ids
    is what makes 200 negatives/row tractable at all: as text that is ~187 GB against a 5.4 GB
    source, versus ~1.7 GB for ids + corpus.

    `corpus_path` is an **Arrow IPC file** whose row i is doc_id i (built and order-asserted by
    datagen/download_multihop_hardneg.py). Arrow IPC rather than parquet on purpose: it is
    uncompressed and memory-mappable, so all N grain worker processes share ONE physical copy
    via the OS page cache instead of each decompressing its own ~0.9 GB.

    Returns a functools.partial over a module-level function, NOT a closure: grain cloudpickles
    the map fn for its workers, and a closure would be pickled by value together with the
    globals it touches — including the cached MemoryMappedFile, which is unpicklable.

    max_neg_docs truncates the negative list. Negatives are stored hardest-first, so a prefix is
    the K hardest; None keeps all of them.

    cot_field: if set and present on the row, forwards that field's raw text under the
    `cot_doc` key (untouched otherwise) for qa_transform_item to append as an extra positive
    document in the memory bank. Independent of `think_field` (which instead prepends CoT into
    the answer/CE target) so the two can't accidentally interact.
    """
    import functools
    return functools.partial(
        _docid_normalize,
        corpus_path=corpus_path,
        think_field=think_field,
        max_neg_docs=max_neg_docs,
        min_neg_docs=min_neg_docs,
        mask_ce=mask_ce,
        field_map=field_map,
        cot_field=cot_field,
    )
