"""
  - Wonderwords adj-noun pairs as NIAH keys
  - PG19 as the distractor haystack 
  - Proper CWE (num_cw=10, freq_cw=30, freq_ucw=3, wonderwords vocab, numbered list)
  - Proper FWE (Zipfian alpha=2.0, scipy.special.zeta, top-3 answers)
  - Real SQuAD + HotpotQA 
  - string_match_all scoring for all tasks; string_match_part for QA

Tasks:
  niah_s       Single NIAH: one key-value needle
  niah_mk      Multi-Key NIAH: N needles, query for one
  niah_mv      Multi-Value NIAH: one key, N values
  niah_mq      Multi-Query NIAH: N needles, N questions
  vt           Variable Tracking: chain of assignments
  cwe          Common Words Extraction: numbered word list
  fwe          Frequent Words Extraction: Zipfian coded text
  qa_squad     QA on SQuAD documents
  qa_hotpot    QA on HotpotQA documents
  
"""

import json
import math
import os
import random
import string
import uuid

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils
import numpy as np
from jax.sharding import NamedSharding
from scipy.special import zeta as scipy_zeta
from tqdm import tqdm

from evals.utils import embed_documents_tokenized
from inference import _generate_tokens


_TASK_MAX_TOKENS = {
    t: 1024
    for t in [
        "niah_s",  "niah_sa1", "niah_sa2", "niah_sa3",
        "niah_mk", "niah_mk1", "niah_mk2", "niah_mk3",
        "niah_mv",
        "vt", "cwe", "fwe", "qa_squad", "qa_hotpot",
    ]
}
for _t in ["niah_sa1", "niah_sa2", "niah_sa3",
           "niah_mk1", "niah_mk2", "niah_mk3",
           "niah_mv",  "niah_mq"]:
    _TASK_MAX_TOKENS[_t] = 2048

# Tasks that use PG19 + needle injection.  All others generate context directly.
_HAYSTACK_TASKS = {
    "niah_s", "niah_sa1", "niah_sa2", "niah_sa3",
    "niah_mk", "niah_mk1", "niah_mk2", "niah_mk3",
    "niah_mv", "niah_mq", "vt",
}

_PG19_DEFAULT_MAX_CHARS = 40_000_000

_DOCUMENT_PROMPT = "Document {i}:\n{document}"

# CWE/FWE prompts (from RULER constants.py), adapted for embed path where the
# context is already in memory rather than inline in the prompt.
_CWE_QUESTION   = "What are the 10 most common words in the word list you have memorized?"
_CWE_INSTRUCTION = "List the 10 words, one per line, no extra text."

_FWE_QUESTION    = "In the coded text you have memorized, what are the three most frequently appearing coded words? Do not provide any explanation. Please ignore the dots '....'."
_FWE_INSTRUCTION = "The three most frequently appeared words are:"

_QA_QUESTION_FMT  = "{question}"
_QA_INSTRUCTION   = "Answer with only the exact answer, no extra text."


# ── PG19 helpers ──────────────────────────────────────────────────────────────

_PG19_BIBLE_KEYWORDS = (
    "bible", "king james", "testament", "old testament", "new testament",
    "king james bible", "holy bible",
)

def _is_bible_book(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _PG19_BIBLE_KEYWORDS)


def _load_pg19_text(cache_dir: str = "/tmp/pg19_haystack",
                    max_chars: int = _PG19_DEFAULT_MAX_CHARS) -> str:
    """Stream PG19 books from HuggingFace, concatenate up to max_chars, cache locally.

    Bible books (King James, Old/New Testament) are excluded from the corpus.
    """
    from pathlib import Path
    cache_path = Path(cache_dir) / f"corpus_{max_chars}_no_bible.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RULER] Streaming PG19 from HuggingFace (target {max_chars:,} chars, Bible excluded)...")

    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    parts, total, skipped = [], 0, 0
    for example in ds:
        title = example.get("short_book_title", "")
        if _is_bible_book(title):
            skipped += 1
            continue
        text = example["text"]
        remaining = max_chars - total
        if len(text) >= remaining:
            parts.append(text[:remaining])
            total += remaining
            break
        parts.append(text)
        total += len(text)

    corpus = "\n\n".join(parts)
    cache_path.write_text(corpus, encoding="utf-8")
    print(f"[RULER] Cached PG19 ({len(corpus):,} chars, {skipped} Bible books skipped) → {cache_path}")
    return corpus


def _tokenize_corpus(tokenizer, corpus: str, min_tokens: int) -> list:
    """Tokenize corpus; tile until we have at least min_tokens tokens."""
    ids = tokenizer(corpus, return_tensors="np", add_special_tokens=False)["input_ids"][0].tolist()
    while len(ids) < min_tokens:
        ids = ids + ids
    return ids


# ── Wonderwords vocab (lazy-loaded) ───────────────────────────────────────────

_WW_ADJ_NOUN = None   # adj-noun pairs for NIAH keys
_WW_WORDS    = None   # nouns + adjs + verbs for CWE vocab
_RANDLE_WORDS = None  # english_words.json fallback for CWE


def _get_adj_noun_words():
    global _WW_ADJ_NOUN
    if _WW_ADJ_NOUN is None:
        import wonderwords
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs  = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        _WW_ADJ_NOUN = sorted(list(set(f"{a}-{n}" for a in adjs for n in nouns)))
    return _WW_ADJ_NOUN


def _get_ww_words():
    global _WW_WORDS
    if _WW_WORDS is None:
        import wonderwords
        nouns  = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs   = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        verbs  = wonderwords.random_word._get_words_from_text_file("verblist.txt")
        _WW_WORDS = sorted(list(set(nouns + adjs + verbs)))
    return _WW_WORDS


def _get_randle_words(n: int = 100_000) -> list:
    """Generate a large pool of synthetic 5-char lowercase words as CWE fallback."""
    global _RANDLE_WORDS
    if _RANDLE_WORDS is None or len(_RANDLE_WORDS) < n:
        rng = random.Random(0)
        pool = set()
        while len(pool) < n:
            pool.add("".join(rng.choices(string.ascii_lowercase, k=5)))
        _RANDLE_WORDS = sorted(pool)
    return _RANDLE_WORDS


# ── Haystack token manipulation ────────────────────────────────────────────────

def _build_haystack_tokens(corpus_tokens: list, total_tokens: int,
                            needle_ids_with_depths: list) -> list:
    """Insert needles into a corpus slice at specified depths."""
    total_needle_len = sum(len(ids) for ids, _ in needle_ids_with_depths)
    haystack = corpus_tokens[: total_tokens - total_needle_len]
    H = len(haystack)

    sorted_needles = sorted(needle_ids_with_depths, key=lambda x: x[1])
    full = list(haystack)
    cumulative = 0
    for needle_ids, depth_pct in sorted_needles:
        insert_at = int(H * depth_pct) + cumulative
        full = full[:insert_at] + needle_ids + full[insert_at:]
        cumulative += len(needle_ids)
    return full


def _make_needle_chunks(corpus_tokens: list, tokenizer, total_tokens: int,
                         needles_with_depths: list, chunk_size: int,
                         stride: int, data_parallel: int = 1):
    """Build (docs, mask) arrays with needles injected, sized to data_parallel multiple."""
    needle_ids_with_depths = [
        (tokenizer(n, return_tensors="np", add_special_tokens=False)["input_ids"][0].tolist(), d)
        for n, d in needles_with_depths
    ]
    max_needle_len = max(len(ids) for ids, _ in needle_ids_with_depths)
    assert max_needle_len <= chunk_size, (
        f"needle ({max_needle_len} tokens) > chunk_size ({chunk_size})"
    )
    assert stride <= chunk_size - max_needle_len, (
        f"stride ({stride}) must be <= chunk_size - max_needle_len ({chunk_size - max_needle_len})"
    )

    full = _build_haystack_tokens(corpus_tokens, total_tokens, needle_ids_with_depths)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    starts = list(range(0, len(full), stride))
    n_real = len(starts)
    n_pad  = n_real + (-n_real % data_parallel)

    docs = np.full((n_pad, chunk_size), pad_id, dtype=np.int32)
    mask = np.zeros((n_pad, chunk_size), dtype=np.bool_)
    for i, start in enumerate(starts):
        chunk = full[start : start + chunk_size]
        docs[i, : len(chunk)] = chunk
        mask[i, : len(chunk)] = True
    return docs, mask


# ── Depth helpers ──────────────────────────────────────────────────────────────

def _evenly_spaced_depths(n: int) -> list:
    return [(i + 1) / (n + 1) for i in range(n)]


# ── NIAH / VT task builders ────────────────────────────────────────────────────

def _build_niah_s(rng: np.random.RandomState, cfg: dict):
    """Single NIAH: one adj-noun key → value at midpoint.

    value_type (from cfg):
      "numeric"  SA2 — random 7-digit number (default, original behaviour)
      "word"     SA1 — random word from wonderwords vocab
      "uuid"     SA3 — 32-char UUID hex string
    """
    value_type = cfg.get("value_type", "numeric")
    words = _get_adj_noun_words()
    key   = words[int(rng.randint(0, len(words)))]

    if value_type == "word":
        vocab = _get_ww_words()
        value = vocab[int(rng.randint(0, len(vocab)))]
        needle   = f"\nThe special magic word for {key} is: {value}.\n"
        question = f"What is the special magic word for {key}?"
    elif value_type == "uuid":
        # Use rng seed so results are reproducible; uuid4 is random by nature,
        # so we seed Python's random, generate, then restore state.
        py_state = random.getstate()
        random.seed(int(rng.randint(0, 2**31)))
        value = uuid.UUID(int=random.getrandbits(128), version=4).hex  # 32-char hex
        random.setstate(py_state)
        needle   = f"\nThe special magic code for {key} is: {value}.\n"
        question = f"What is the special magic code for {key}?"
    else:  # "numeric"
        value    = str(int(rng.randint(1_000_000, 9_999_999)))
        needle   = f"\nOne of the special magic numbers for {key} is: {value}.\n"
        question = f"What is the special magic number for {key}?"

    return [(needle, 0.5)], question, [value]


def _build_niah_sa1(rng: np.random.RandomState, cfg: dict):
    """SA1 — word value."""
    return _build_niah_s(rng, {**cfg, "value_type": "word"})


def _build_niah_sa2(rng: np.random.RandomState, cfg: dict):
    """SA2 — 7-digit numeric value."""
    return _build_niah_s(rng, {**cfg, "value_type": "numeric"})


def _build_niah_sa3(rng: np.random.RandomState, cfg: dict):
    """SA3 — 32-char UUID value."""
    return _build_niah_s(rng, {**cfg, "value_type": "uuid"})


def _build_niah_mk(rng: np.random.RandomState, cfg: dict):
    """Multi-Key NIAH: N distinct adj-noun keys at evenly-spaced depths."""
    num_needles = int(cfg.get("num_needles", 4))
    words   = _get_adj_noun_words()
    depths  = _evenly_spaced_depths(num_needles)
    indices = rng.choice(len(words), size=num_needles, replace=False).tolist()
    keys    = [words[i] for i in indices]
    values  = [str(int(rng.randint(1_000_000, 9_999_999))) for _ in range(num_needles)]
    target  = int(rng.randint(0, num_needles))

    needles_with_depths = [
        (f"\nOne of the special magic numbers for {keys[i]} is: {values[i]}.\n", depths[i])
        for i in range(num_needles)
    ]
    question = f"What is the special magic number for {keys[target]}?"
    return needles_with_depths, question, [values[target]]


def _build_niah_mk1(rng: np.random.RandomState, cfg: dict):
    """MK1 — 3 needles total (1 target + 2 distractors)."""
    return _build_niah_mk(rng, {**cfg, "num_needles": 3})


def _build_niah_mk2(rng: np.random.RandomState, cfg: dict):
    """MK2 — 4 needles total (1 target + 3 distractors)."""
    return _build_niah_mk(rng, {**cfg, "num_needles": 4})


def _build_niah_mk3(rng: np.random.RandomState, cfg: dict):
    """MK3 — 5 needles total (1 target + 4 distractors)."""
    return _build_niah_mk(rng, {**cfg, "num_needles": 5})


def _build_niah_mv(rng: np.random.RandomState, cfg: dict):
    """Multi-Value NIAH: one adj-noun key → N distinct values at evenly-spaced depths."""
    num_values = int(cfg.get("num_values", 4))
    words  = _get_adj_noun_words()
    key    = words[int(rng.randint(0, len(words)))]
    values = [str(int(rng.randint(1_000_000, 9_999_999))) for _ in range(num_values)]
    depths = _evenly_spaced_depths(num_values)

    needles_with_depths = [
        (f"\nOne of the special magic numbers for {key} is: {values[i]}.\n", depths[i])
        for i in range(num_values)
    ]
    question = (
        f"There are exactly {num_values} separate special magic numbers for {key}. "
        f"List all {num_values} of those numbers, one per line."
    )
    return needles_with_depths, question, values


def _build_niah_mq(rng: np.random.RandomState, cfg: dict):
    """Multi-Query NIAH: N distinct adj-noun key-value needles, N questions."""
    num_queries = int(cfg.get("num_queries", 4))
    words   = _get_adj_noun_words()
    depths  = _evenly_spaced_depths(num_queries)
    indices = rng.choice(len(words), size=num_queries, replace=False).tolist()
    keys    = [words[i] for i in indices]
    values  = [str(int(rng.randint(1_000_000, 9_999_999))) for _ in range(num_queries)]

    needles_with_depths = [
        (f"\nOne of the special magic numbers for {keys[i]} is: {values[i]}.\n", depths[i])
        for i in range(num_queries)
    ]
    lines = "\n".join(
        f"{i+1}. What is the special magic number for {keys[i]}?"
        for i in range(num_queries)
    )
    question = f"Answer each question, one answer per line:\n{lines}"
    return needles_with_depths, question, values


def _build_vt(rng: np.random.RandomState, cfg: dict):
    """Variable Tracking: chain of assignments inserted at midpoint."""
    num_hops = int(cfg.get("num_hops", 4))

    initial_value = int(rng.randint(100_000, 999_999))
    # 5-char uppercase variable names
    k = 5
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    var_names = []
    seen = set()
    while len(var_names) < num_hops + 1:
        name = "".join(rng.choice(alphabet, size=k).tolist())
        if name not in seen:
            seen.add(name)
            var_names.append(name)

    lines = [f"VAR {var_names[0]} = {initial_value}"]
    for i in range(1, num_hops + 1):
        lines.append(f"VAR {var_names[i]} = VAR {var_names[i-1]}")
    needle = "\n" + "\n".join(lines) + "\n"

    question = (
        f"Track the chain(s) of variable assignment. "
        f"Find all variables that are assigned the value {initial_value}."
    )
    answers = var_names
    return [(needle, 0.5)], question, answers


# ── Binary search helper ───────────────────────────────────────────────────────

def _binary_search_size(gen_text_fn, tokenizer, target_tokens: int,
                         lo: int, hi: int) -> int:
    """Find largest size s.t. tokenized text fits in target_tokens."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        text = gen_text_fn(mid)
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n <= target_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ── CWE task builder ───────────────────────────────────────────────────────────

def _make_cwe_context(num_words: int, rng_seed: int, num_cw: int = 10,
                       freq_cw: int = 30, freq_ucw: int = 3) -> tuple:
    """Build a numbered word list context and return (context_text, common_words)."""
    ww = _get_ww_words()
    randle = _get_randle_words()
    # Use deterministic local rng for word sampling
    local_rng = random.Random(rng_seed)
    if num_words <= len(ww):
        word_list_full = local_rng.sample(ww, num_words)
    else:
        word_list_full = local_rng.sample(randle, min(num_words, len(randle)))

    common   = word_list_full[:num_cw]
    uncommon = word_list_full[num_cw:]
    words    = common * freq_cw + uncommon * freq_ucw
    local_rng.shuffle(words)

    context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(words))
    return context, common


def _build_cwe(rng: np.random.RandomState, tokenizer, context_length: int,
                cfg: dict, _size_cache: dict):
    """CWE: numbered word list that fills context_length tokens."""
    num_cw  = int(cfg.get("num_cw",   10))
    freq_cw = int(cfg.get("freq_cw",  30))
    freq_ucw = int(cfg.get("freq_ucw",  3))
    seed    = int(rng.randint(0, 2**30))

    cache_key = ("cwe", context_length, num_cw, freq_cw, freq_ucw)
    if cache_key not in _size_cache:
        tokens_to_gen = _TASK_MAX_TOKENS["cwe"]
        target = context_length - tokens_to_gen

        def gen_fn(nw):
            ctx, _ = _make_cwe_context(nw, rng_seed=0, num_cw=num_cw,
                                        freq_cw=freq_cw, freq_ucw=freq_ucw)
            return ctx

        lo = num_cw + 1
        hi = max(lo + 1, target // 3)  # rough upper bound: ~3 tokens per "N. word"
        _size_cache[cache_key] = _binary_search_size(gen_fn, tokenizer, target, lo, hi)

    num_words = _size_cache[cache_key]
    context, common = _make_cwe_context(num_words, rng_seed=seed,
                                         num_cw=num_cw, freq_cw=freq_cw, freq_ucw=freq_ucw)
    question = _CWE_QUESTION
    return context, question, common


# ── FWE task builder ───────────────────────────────────────────────────────────

def _make_fwe_context(num_total_words: int, vocab_size: int,
                       coded_wordlen: int, alpha: float, rng_seed: int) -> tuple:
    """Build Zipfian coded text; return (context_text, top3_answers)."""
    local_rng = random.Random(rng_seed)
    vocab = ["".join(local_rng.choices(string.ascii_lowercase, k=coded_wordlen))
             for _ in range(vocab_size * 2)]
    # deduplicate, sort, shuffle, trim
    vocab = sorted(list(set(vocab)))
    local_rng.shuffle(vocab)
    vocab = vocab[:vocab_size]
    vocab[0] = "..."  # top-ranked treated as noise

    np_rng = np.random.RandomState(rng_seed)
    k     = np.arange(1, len(vocab) + 1)
    cnt   = (num_total_words * (k ** -alpha) / scipy_zeta(alpha)).astype(int)
    words = [w for w, c in zip(vocab, cnt) for _ in range(c)]
    local_rng.shuffle(words)

    context = " ".join(words)
    return context, vocab[1:4]


def _build_fwe(rng: np.random.RandomState, tokenizer, context_length: int,
                cfg: dict, _size_cache: dict):
    """FWE: Zipfian coded text that fills context_length tokens."""
    alpha         = float(cfg.get("fwe_alpha",       2.0))
    coded_wordlen = int(cfg.get("coded_wordlen",    6))
    seed          = int(rng.randint(0, 2**30))

    tokens_to_gen = _TASK_MAX_TOKENS["fwe"]
    target        = context_length - tokens_to_gen
    vocab_size    = max(50, target // 50)

    cache_key = ("fwe", context_length, vocab_size, coded_wordlen, alpha)
    if cache_key not in _size_cache:
        def gen_fn(nw):
            ctx, _ = _make_fwe_context(nw, vocab_size, coded_wordlen, alpha, rng_seed=0)
            return ctx

        lo = 100
        hi = max(lo + 1, target // coded_wordlen)
        _size_cache[cache_key] = _binary_search_size(gen_fn, tokenizer, target, lo, hi)

    num_total_words = _size_cache[cache_key]
    context, answers = _make_fwe_context(num_total_words, vocab_size,
                                          coded_wordlen, alpha, rng_seed=seed)
    return context, _FWE_QUESTION, answers


# ── QA data loading ────────────────────────────────────────────────────────────

_SQUAD_DATA    = None
_HOTPOTQA_DATA = None


def _load_squad():
    """Load SQuAD validation via HuggingFace datasets. Returns (qas, docs)."""
    global _SQUAD_DATA
    if _SQUAD_DATA is not None:
        return _SQUAD_DATA

    print("[RULER] Loading SQuAD via HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")

    total_docs = sorted(list(set(ex["context"] for ex in ds)))
    doc_idx    = {c: i for i, c in enumerate(total_docs)}

    total_qas = []
    seen_ctx_per_doc = {}  # paragraph_context -> [qas with other paragraphs in same article]
    for ex in ds:
        total_qas.append({
            "query":   ex["question"],
            "outputs": ex["answers"]["text"],
            "context": [doc_idx[ex["context"]]],
        })

    _SQUAD_DATA = (total_qas, total_docs)
    return _SQUAD_DATA


def _load_hotpotqa():
    """Load HotpotQA distractor validation via HuggingFace datasets. Returns (qas, docs)."""
    global _HOTPOTQA_DATA
    if _HOTPOTQA_DATA is not None:
        return _HOTPOTQA_DATA

    print("[RULER] Loading HotpotQA via HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    all_docs = []
    for ex in ds:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            all_docs.append(title + "\n" + "".join(sents))
    total_docs = sorted(list(set(all_docs)))
    doc_idx    = {d: i for i, d in enumerate(total_docs)}

    total_qas = []
    for ex in ds:
        ctx_docs = [
            doc_idx[title + "\n" + "".join(sents)]
            for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]
        total_qas.append({
            "query":   ex["question"],
            "outputs": [ex["answer"]],
            "context": ctx_docs,
        })

    _HOTPOTQA_DATA = (total_qas, total_docs)
    return _HOTPOTQA_DATA


def _make_qa_context(sample_idx: int, num_docs: int, qas: list, docs: list,
                      rng_seed: int) -> tuple:
    """Build multi-document context for one QA sample."""
    local_rng = random.Random(rng_seed)
    qa    = qas[sample_idx % len(qas)]
    query = qa["query"]
    answers = qa["outputs"]

    required = qa["context"]
    if num_docs >= len(docs):
        repeats = (num_docs + len(docs) - 1) // len(docs)
        all_doc_texts = (docs * repeats)[:num_docs]
    else:
        additional = [i for i in range(len(docs)) if i not in required]
        local_rng.shuffle(additional)
        selected = required + additional[: max(0, num_docs - len(required))]
        all_doc_texts = [docs[i] for i in selected]

    local_rng.shuffle(all_doc_texts)
    context = "\n\n".join(
        _DOCUMENT_PROMPT.format(i=i + 1, document=d)
        for i, d in enumerate(all_doc_texts)
    )
    return context, query, answers


def _build_qa(task: str, sample_idx: int, rng: np.random.RandomState,
               tokenizer, context_length: int, cfg: dict, _size_cache: dict):
    """QA on real SQuAD or HotpotQA documents."""
    qas, docs = _load_squad() if task == "qa_squad" else _load_hotpotqa()
    seed      = int(rng.randint(0, 2**30))

    tokens_to_gen = _TASK_MAX_TOKENS[task]
    target        = context_length - tokens_to_gen

    cache_key = (task, context_length)
    if cache_key not in _size_cache:
        def gen_fn(nd):
            ctx, _, _ = _make_qa_context(0, nd, qas, docs, rng_seed=0)
            return ctx

        lo = len(qas[0]["context"])
        hi = max(lo + 1, target // 200)  # rough: ~200 tokens per doc
        _size_cache[cache_key] = _binary_search_size(gen_fn, tokenizer, target, lo, hi)

    num_docs = _size_cache[cache_key]
    context, query, answers = _make_qa_context(sample_idx, num_docs, qas, docs, rng_seed=seed)
    question = query
    return context, question, answers


# ── Tokenize context into chunks ───────────────────────────────────────────────

def _tokenize_and_chunk(tokenizer, context_text: str, chunk_size: int,
                         stride: int, data_parallel: int = 1):
    """Tokenize context_text into overlapping chunks; pad to data_parallel multiple."""
    pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    ids = tokenizer(context_text, return_tensors="np",
                    add_special_tokens=False)["input_ids"][0].tolist()

    starts = list(range(0, len(ids), stride))
    n_real = len(starts)
    n_pad  = n_real + (-n_real % data_parallel)

    docs = np.full((n_pad, chunk_size), pad_id, dtype=np.int32)
    mask = np.zeros((n_pad, chunk_size), dtype=np.bool_)
    for i, start in enumerate(starts):
        chunk = ids[start : start + chunk_size]
        docs[i, : len(chunk)] = chunk
        mask[i, : len(chunk)] = True
    return docs, mask


# ── Embed infrastructure ───────────────────────────────────────────────────────

def _setup_embed(model):
    """Extract embed weights and mesh from loaded model. Returns (embed_fn, data_parallel, mesh)."""
    from models.qwen3_mem_embed import embed_forward
    from models.utils import split_weights
    from models.retrieval_ops import set_global_mesh

    _, embed_w = split_weights(model.weights, ["main_model", "embed_model"])
    embed_cfg  = model.cfg["embed_model"]
    embed_w    = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, embed_w
    )
    embed_fn = lambda docs, dmask: embed_forward(embed_cfg, docs, embed_w, dmask)

    some_w = next(v for v in model.weights.values()
                  if hasattr(getattr(v, "sharding", None), "mesh"))
    mesh          = some_w.sharding.mesh
    data_parallel = mesh.shape["data"]

    set_global_mesh(mesh)
    return embed_fn, data_parallel, mesh


def _embed_base_haystack(corpus_tokens: list, ctx_len: int, chunk_size: int,
                          stride: int, tokenizer, embed_fn, embed_batch_size: int,
                          data_parallel: int):
    """Pre-embed a bare PG19 haystack (no needles). Returns (k, v, mask, n_real, vpc)."""
    pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)

    starts = list(range(0, ctx_len, stride))
    n_real = len(starts)
    n_pad  = n_real + (-n_real % data_parallel)

    docs = np.full((n_pad, chunk_size), pad_id, dtype=np.int32)
    mask = np.zeros((n_pad, chunk_size), dtype=np.bool_)
    for i, start in enumerate(starts):
        chunk = corpus_tokens[start : start + chunk_size]
        docs[i, : len(chunk)] = chunk
        mask[i, : len(chunk)] = True

    k_np, v_np, mask_np = embed_documents_tokenized(embed_fn, docs, mask, embed_batch_size)
    vecs_per_chunk = k_np.shape[0] // n_pad
    k_np    = k_np   [: n_real * vecs_per_chunk]
    v_np    = v_np   [: n_real * vecs_per_chunk]
    mask_np = mask_np[: n_real * vecs_per_chunk]
    return k_np, v_np, mask_np, n_real, vecs_per_chunk


def _affected_chunk_indices(needle_ids_with_depths: list, total_tokens: int,
                             chunk_size: int, stride: int) -> list:
    """Chunk indices whose content changes when needles are inserted."""
    total_needle_len = sum(len(ids) for ids, _ in needle_ids_with_depths)
    H        = total_tokens - total_needle_len
    n_chunks = math.ceil(total_tokens / stride)

    sorted_needles = sorted(needle_ids_with_depths, key=lambda x: x[1])
    cumulative = 0
    affected   = set()
    for needle_ids, depth_pct in sorted_needles:
        insert_at  = int(H * depth_pct) + cumulative
        needle_len = len(needle_ids)
        i_min = max(0, math.ceil((insert_at - chunk_size + 1) / stride))
        i_max = min(n_chunks - 1, (insert_at + needle_len - 1) // stride)
        for i in range(i_min, i_max + 1):
            affected.add(i)
        cumulative += needle_len
    return sorted(affected)


def _patch_embeddings(base_k, base_v, base_mask, n_real_chunks, vecs_per_chunk,
                       corpus_tokens, needles_with_depths, tokenizer,
                       total_tokens, chunk_size, stride,
                       embed_fn, embed_batch_size, data_parallel):
    """Re-embed only chunks containing needle tokens; patch into base arrays."""
    needle_ids_with_depths = [
        (tokenizer(n, return_tensors="np", add_special_tokens=False)["input_ids"][0].tolist(), d)
        for n, d in needles_with_depths
    ]

    affected = _affected_chunk_indices(needle_ids_with_depths, total_tokens, chunk_size, stride)
    full     = _build_haystack_tokens(corpus_tokens, total_tokens, needle_ids_with_depths)

    pad_id  = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
               else tokenizer.eos_token_id)
    n_aff   = len(affected)
    n_pad   = n_aff + (-n_aff % data_parallel)

    patch_docs = np.full((n_pad, chunk_size), pad_id, dtype=np.int32)
    patch_mask = np.zeros((n_pad, chunk_size), dtype=np.bool_)
    for j, i in enumerate(affected):
        chunk = full[i * stride : i * stride + chunk_size]
        patch_docs[j, : len(chunk)] = chunk
        patch_mask[j, : len(chunk)] = True

    patch_k, patch_v, patch_mask_out = embed_documents_tokenized(
        embed_fn, patch_docs, patch_mask, embed_batch_size
    )

    mem_k    = base_k.copy()
    mem_v    = base_v.copy()
    mem_mask = base_mask.copy()
    for j, i in enumerate(affected):
        src = slice(j * vecs_per_chunk, (j + 1) * vecs_per_chunk)
        dst = slice(i * vecs_per_chunk, (i + 1) * vecs_per_chunk)
        mem_k[dst]    = patch_k[src]
        mem_v[dst]    = patch_v[src]
        mem_mask[dst] = patch_mask_out[src]
    return mem_k, mem_v, mem_mask


# ── Run helper ─────────────────────────────────────────────────────────────────

def _run_embed(model, mem_k_np, mem_v_np, mem_mask_np,
               question: str, instruction: str,
               tokenizer, mesh, data_parallel: int, max_new_tokens: int,
               pad_to_len: int = None) -> tuple:
    """
    Shard mem_k to devices, hold mem_v on CPU, run generation.
    mem_k_np/mem_v_np/mem_mask_np must already be trimmed to real chunks.

    pad_to_len: if given, left-pad the tokenized prompt to this fixed length (same
    convention as inference.generate()). _generate_tokens is jax.jit'd on the prompt's
    shape, so calling it with a different prompt length every sample (the question text
    varies per sample, e.g. NIAH's random adj-noun key) forces a full XLA recompile of the
    whole mem-augmented generate program EVERY sample. Padding every sample in a
    (task, context_length) cell to the same fixed length means the program compiles once
    and is reused for the rest of the cell.

    Returns:
        (answer, thinking_trace)
    """
    from models.retrieval_ops import set_cpu_mem_v

    # Pre-normalize mem_k on CPU (halves peak HBM)
    mem_k_norm_keys = [k for k in model.weights.keys() if k.endswith("mem_k_norm")]
    if mem_k_norm_keys:
        mem_k_norm_w = np.array(model.weights[mem_k_norm_keys[0]])
        rms_eps = model.cfg["main_model"].get("rms_norm_eps", 1e-6)
        sq_mean = np.mean(mem_k_np.astype(np.float32) ** 2, axis=-1, keepdims=True)
        mem_k_np = (
            mem_k_np.astype(np.float32) / np.sqrt(sq_mean + rms_eps) * mem_k_norm_w
        ).astype(np.float32)
        model.cfg["main_model"]["mem_k_prenormed"] = True

    # Pad to multiple of data_parallel
    M = mem_k_np.shape[0]
    rem = M % data_parallel
    if rem != 0:
        pad = data_parallel - rem
        mem_k_np    = np.concatenate([mem_k_np,    np.zeros((pad, mem_k_np.shape[1]),   dtype=mem_k_np.dtype)])
        mem_v_np    = np.concatenate([mem_v_np,    np.zeros((pad, mem_v_np.shape[1]),   dtype=mem_v_np.dtype)])
        mem_mask_np = np.concatenate([mem_mask_np, np.zeros(pad,                        dtype=mem_mask_np.dtype)])

    k_sharding    = NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    mask_sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    def _make_sharded(arr_np, sharding, dtype=None):
        def cb(idx):
            s = arr_np[idx]
            return s.astype(dtype) if dtype is not None else s
        return jax.make_array_from_callback(arr_np.shape, sharding, cb)

    set_cpu_mem_v(mem_v_np.astype(np.float32))
    mem_k_rep    = _make_sharded(mem_k_np,    k_sharding,    dtype=jnp.bfloat16)
    mem_mask_rep = _make_sharded(mem_mask_np, mask_sharding)
    Dv = mem_v_np.shape[-1]
    mem_v_dummy = jax.device_put(
        jnp.zeros((1, Dv), dtype=jnp.bfloat16),
        NamedSharding(mesh, jax.sharding.PartitionSpec(None, None)),
    )

    params = dict(model.weights)
    params["main_model.mem_k"]    = mem_k_rep
    params["main_model.mem_v"]    = mem_v_dummy
    params["main_model.mem_mask"] = mem_mask_rep
    model.cfg["main_model"]["mem_shard_axis"] = "data"
    model.cfg["main_model"]["mem_lookup_chunk_size"] = 8192

    messages = [{"role": "user", "content": f"{question}\n\n{instruction}"}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    single_ids = tokenizer(prompt_text, return_tensors="np")["input_ids"][0]

    if pad_to_len is not None:
        pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_len = pad_to_len - len(single_ids)
        assert pad_len >= 0, f"prompt ({len(single_ids)} tok) exceeds precomputed pad_to_len ({pad_to_len})"
        single_mask = np.concatenate([np.zeros(pad_len, dtype=np.bool_), np.ones(len(single_ids), dtype=np.bool_)])
        single_ids  = np.concatenate([np.full(pad_len, pad_id, dtype=single_ids.dtype), single_ids])
    else:
        single_mask = np.ones_like(single_ids, dtype=np.bool_)

    prompt_ids = np.stack([single_ids]  * data_parallel).astype(np.int32)
    pmask      = np.stack([single_mask] * data_parallel)

    prompt_jax = jax.device_put(
        jnp.array(prompt_ids),
        NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
    )
    pmask_jax = jax.device_put(
        jnp.array(pmask),
        NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
    )

    jax.set_mesh(mesh)
    gen_tokens = _generate_tokens(
        model.forward, model.init_kv, params,
        prompt_jax, max_new_tokens,
        pad_mask=pmask_jax, temperature=0.0,
    )

    gen_np  = np.array(jax.experimental.multihost_utils.process_allgather(gen_tokens, tiled=True))
    eos_pos = np.where(gen_np[0] == tokenizer.eos_token_id)[0]
    gen     = gen_np[0, : eos_pos[0]] if len(eos_pos) > 0 else gen_np[0]

    set_cpu_mem_v(None)
    model.cfg["main_model"].pop("mem_shard_axis",       None)
    model.cfg["main_model"].pop("mem_k_prenormed",     None)
    model.cfg["main_model"].pop("mem_lookup_chunk_size", None)

    full_text = tokenizer.decode(gen, skip_special_tokens=False)

    if "</think>" in full_text:
        think_end   = full_text.index("</think>")
        think_start = full_text.index("<think>") + len("<think>") if "<think>" in full_text else 0
        thinking_trace = full_text[think_start:think_end].strip()
        answer = full_text[think_end + len("</think>"):].strip()
        # Strip any residual special tokens from the answer
        answer = tokenizer.decode(
            tokenizer(answer, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )
    elif "<think>" in full_text:
        # Model was cut off mid-thinking (no </think>); no usable answer.
        thinking_trace = full_text[full_text.index("<think>") + len("<think>"):].strip()
        answer = ""
    else:
        thinking_trace = ""
        answer = tokenizer.decode(gen, skip_special_tokens=True)

    return answer, thinking_trace


# ── Per-task instructions (for embed path prompt) ──────────────────────────────

_TASK_INSTRUCTION = {
    "niah_s":    "Answer with the exact number only.",
    "niah_sa1":  "Answer with the exact word only.",
    "niah_sa2":  "Answer with the exact number only.",
    "niah_sa3":  "Answer with the exact value only.",
    "niah_mk":   "Answer with the exact number only.",
    "niah_mk1":  "Answer with the exact number only.",
    "niah_mk2":  "Answer with the exact number only.",
    "niah_mk3":  "Answer with the exact number only.",
    "niah_mv":   "List all numbers, one per line.",
    "niah_mq":   "Answer each question on a separate numbered line, in the same order as asked.",
    "vt":        "List all variable names assigned this value, one per line.",
    "cwe":       _CWE_INSTRUCTION,
    "fwe":       _FWE_INSTRUCTION,
    "qa_squad":  _QA_INSTRUCTION,
    "qa_hotpot": _QA_INSTRUCTION,
}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_recall(output: str, answers: list) -> float:
    """string_match_all: fraction of answer strings found in output."""
    out = output.lower()
    return sum(1 for a in answers if str(a).lower() in out) / len(answers)


def _score_qa(output: str, answers: list) -> float:
    """string_match_part: 1.0 if any answer found, else 0.0."""
    out = output.lower()
    return 1.0 if any(str(a).lower() in out for a in answers) else 0.0


_TASK_SCORE_FN = {
    "niah_s":    _score_recall,
    "niah_sa1":  _score_recall,
    "niah_sa2":  _score_recall,
    "niah_sa3":  _score_recall,
    "niah_mk":   _score_recall,
    "niah_mk1":  _score_recall,
    "niah_mk2":  _score_recall,
    "niah_mk3":  _score_recall,
    "niah_mv":   _score_recall,
    "niah_mq":   _score_recall,
    "vt":        _score_recall,
    "cwe":       _score_recall,
    "fwe":       _score_recall,
    "qa_squad":  _score_qa,
    "qa_hotpot": _score_qa,
}


# ── Save / heatmap ─────────────────────────────────────────────────────────────

def _save(out_path: str, model_name: str, tasks: list,
          context_lengths: list, all_results: dict):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_name":      model_name,
            "tasks":           tasks,
            "context_lengths": context_lengths,
            "results":         all_results,
        }, f, indent=2)


def _save_heatmap(png_path: str, model_name: str, tasks: list,
                   context_lengths: list, all_results: dict, avg_score: float):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[RULER] matplotlib not installed — skipping heatmap.")
        return

    grid = np.zeros((len(context_lengths), len(tasks)))
    for col, task in enumerate(tasks):
        for r in all_results[task]:
            ci = context_lengths.index(r["context_length"])
            grid[ci, col] = r["score"]

    ctx_labels = [
        f"{c // 1024}K" if c < 1_000_000 else f"{c // 1_000_000}M"
        for c in context_lengths
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(tasks) * 1.5), max(3, len(context_lengths) + 1)))
    fig.suptitle(f"RULER — {model_name}  avg={avg_score:.3f}", fontsize=12)
    im = ax.imshow(grid, vmin=0.0, vmax=1.0, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(context_lengths)))
    ax.set_yticklabels(ctx_labels, fontsize=8)
    ax.set_ylabel("Context length", fontsize=9)
    for ci in range(len(context_lengths)):
        for col in range(len(tasks)):
            ax.text(col, ci, f"{grid[ci, col]:.0%}", ha="center", va="center",
                    fontsize=8, color="black" if 0.2 < grid[ci, col] < 0.8 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main evaluate function ─────────────────────────────────────────────────────

def evaluate(
    model,
    tasks:              list  = None,
    context_lengths:    list  = None,
    num_samples:        int   = 500,
    chunk_size:         int   = 256,
    chunk_stride:       int   = 196,
    vt_chunk_stride:    int   = 128,
    embed_batch_size:   int   = 64,
    output_dir:         str   = "results/ruler",
    haystack_cache_dir: str   = "/tmp/pg19_haystack",
    haystack_max_chars: int   = _PG19_DEFAULT_MAX_CHARS,
    # task-specific
    num_needles:  int = 4,
    num_values:   int = 4,
    num_queries:  int = 4,
    num_hops:     int = 4,
    num_cw:       int = 10,
    freq_cw:      int = 30,
    freq_ucw:     int = 3,
):
    """
    Run RULER benchmark evaluation.

    Args:
        model:            Loaded qwen3_mem_embed model instance.
        tasks:            Task names to run (default: all 9 tasks).
        context_lengths:  Token context lengths to evaluate.
        num_samples:      Samples per (task, context_length) cell.
        chunk_size:       Tokens per doc chunk for embedding.
        chunk_stride:     Chunk stride (tokens). VT uses vt_chunk_stride instead.
        embed_batch_size: Batch size for document embedding.
        output_dir:       Where to write JSON + PNG results.
        haystack_cache_dir: Cache dir for PG19 text.
    """
    if tasks is None:
        tasks = ["niah_s", "niah_mk", "niah_mv", "niah_mq", "vt",
                 "cwe", "fwe", "qa_squad", "qa_hotpot"]
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    cfg = {
        "num_needles": num_needles, "num_values": num_values,
        "num_queries": num_queries, "num_hops":   num_hops,
        "num_cw": num_cw, "freq_cw": freq_cw, "freq_ucw": freq_ucw,
    }

    tokenizer = model.tokenizer

    for t in tasks:
        if t not in _TASK_MAX_TOKENS:
            raise ValueError(f"Unknown task '{t}'. Valid: {list(_TASK_MAX_TOKENS)}")

    if jax.process_index() == 0:
        print(f"[RULER] tasks={tasks}  context_lengths={context_lengths}  "
              f"num_samples={num_samples}")

    # ── Setup embed model ──────────────────────────────────────────────────────
    embed_fn, data_parallel, mesh = _setup_embed(model)

    # Task-specific strides (VT needle can be long, needs smaller stride)
    task_stride = {t: (vt_chunk_stride if t == "vt" else chunk_stride) for t in tasks}

    # ── Load PG19 for haystack tasks ───────────────────────────────────────────
    haystack_tasks_in_run = [t for t in tasks if t in _HAYSTACK_TASKS]
    corpus_tokens = []
    if haystack_tasks_in_run:
        corpus = _load_pg19_text(haystack_cache_dir, haystack_max_chars)
        min_tok = max(context_lengths) + 1024
        corpus_tokens = _tokenize_corpus(tokenizer, corpus, min_tok)

    # ── Pre-embed base haystacks for NIAH/VT tasks ────────────────────────────
    # One base per (ctx_len, stride_val): all samples share the same PG19 slice
    # as background; per-sample randomness comes from the needle position/content.
    unique_strides = sorted(set(task_stride[t] for t in haystack_tasks_in_run)) if haystack_tasks_in_run else []
    base_mem = {}  # (ctx_len, stride_val) -> (k, v, mask, n_real, vpc)

    if haystack_tasks_in_run:
        n_bases = len(context_lengths) * len(unique_strides)
        if jax.process_index() == 0:
            print(f"[RULER] Pre-embedding {n_bases} base haystack(s)...")

        for stride_val in unique_strides:
            for ctx_len in context_lengths:
                key = (ctx_len, stride_val)
                base_mem[key] = _embed_base_haystack(
                    corpus_tokens, ctx_len, chunk_size, stride_val,
                    tokenizer, embed_fn, embed_batch_size, data_parallel,
                )
                if jax.process_index() == 0:
                    n_chunks = base_mem[key][3]
                    print(f"[RULER]   base embedded: ctx={ctx_len} stride={stride_val} "
                          f"chunks={n_chunks}")

    # ── Size cache for CWE/FWE/QA binary search ───────────────────────────────
    _size_cache = {}

    # ── Main evaluation loop ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ruler_results.json")

    # Resume: load any already-completed cells so we can skip them.
    completed = {}  # (task, ctx_len) -> cell dict
    if os.path.exists(out_path):
        try:
            with open(out_path) as _f:
                _existing = json.load(_f)
            for _task, _cells in _existing.get("results", {}).items():
                for _cell in _cells:
                    if len(_cell.get("sample_scores", [])) == num_samples:
                        completed[(_task, _cell["context_length"])] = _cell
            if completed and jax.process_index() == 0:
                print(f"[RULER] Resuming — skipping {len(completed)} already-complete cell(s).")
        except Exception:
            pass

    all_results = {task: [] for task in tasks}
    # Pre-populate with completed cells so they appear in the final JSON.
    for (task, ctx_len), cell in completed.items():
        if task in all_results:
            all_results[task].append(cell)

    skipped_cells = len(completed)
    total_cells = len(tasks) * len(context_lengths)
    pbar = tqdm(total=total_cells, initial=skipped_cells, desc="[RULER]",
                disable=jax.process_index() != 0)

    for task in tasks:
        score_fn    = _TASK_SCORE_FN[task]
        instruction = _TASK_INSTRUCTION[task]
        max_new_tok = _TASK_MAX_TOKENS[task]
        stride_val  = task_stride[task]

        for ctx_len in context_lengths:
            if (task, ctx_len) in completed:
                continue
            sample_scores   = []
            sample_outputs  = []
            sample_thinking = []
            try:
                # Build every sample's needle/context/question up front. This is cheap
                # (RNG + string formatting, no TPU work) compared to what it buys: a single
                # fixed prompt length for the whole cell, so _run_embed's jax.jit'd generate
                # call compiles once instead of recompiling every sample (see _run_embed).
                built = []
                for s_idx in range(num_samples):
                    seed = abs(hash(f"{task}_{ctx_len}_{s_idx}")) % (2 ** 31)
                    rng  = np.random.RandomState(seed)

                    if task in _HAYSTACK_TASKS:
                        needles, question, answers = {
                            "niah_s":   _build_niah_s,
                            "niah_sa1": _build_niah_sa1,
                            "niah_sa2": _build_niah_sa2,
                            "niah_sa3": _build_niah_sa3,
                            "niah_mk":  _build_niah_mk,
                            "niah_mk1": _build_niah_mk1,
                            "niah_mk2": _build_niah_mk2,
                            "niah_mk3": _build_niah_mk3,
                            "niah_mv":  _build_niah_mv,
                            "niah_mq":  _build_niah_mq,
                            "vt":       _build_vt,
                        }[task](rng, cfg)
                        built.append(("haystack", needles, question, answers))
                    else:
                        if task == "cwe":
                            context, question, answers = _build_cwe(
                                rng, tokenizer, ctx_len, cfg, _size_cache)
                        elif task == "fwe":
                            context, question, answers = _build_fwe(
                                rng, tokenizer, ctx_len, cfg, _size_cache)
                        else:  # qa_squad / qa_hotpot
                            context, question, answers = _build_qa(
                                task, s_idx, rng, tokenizer, ctx_len, cfg, _size_cache)
                        built.append(("context", context, question, answers))

                pad_to_len = max(
                    len(tokenizer(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": f"{b[2]}\n\n{instruction}"}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=True,
                        ), return_tensors="np",
                    )["input_ids"][0])
                    for b in built
                )

                for s_idx in range(num_samples):
                    kind, first, question, answers = built[s_idx]

                    if kind == "haystack":
                        needles = first
                        base_k, base_v, base_mask, n_real, vpc = base_mem[(ctx_len, stride_val)]
                        mem_k_np, mem_v_np, mem_mask_np = _patch_embeddings(
                            base_k, base_v, base_mask, n_real, vpc,
                            corpus_tokens, needles, tokenizer,
                            ctx_len, chunk_size, stride_val,
                            embed_fn, embed_batch_size, data_parallel,
                        )

                    else:
                        context = first
                        docs, doc_mask = _tokenize_and_chunk(
                            tokenizer, context, chunk_size, stride_val, data_parallel)
                        n_pad = docs.shape[0]
                        k_np, v_np, mask_np = embed_documents_tokenized(
                            embed_fn, docs, doc_mask, embed_batch_size)
                        # trim pad rows
                        starts = list(range(0, len(
                            tokenizer(context, add_special_tokens=False)["input_ids"]
                        ), stride_val))
                        n_real = len(starts)
                        vpc    = k_np.shape[0] // n_pad
                        mem_k_np    = k_np   [: n_real * vpc]
                        mem_v_np    = v_np   [: n_real * vpc]
                        mem_mask_np = mask_np[: n_real * vpc]

                    output, thinking = _run_embed(
                        model, mem_k_np, mem_v_np, mem_mask_np,
                        question, instruction,
                        tokenizer, mesh, data_parallel, max_new_tok,
                        pad_to_len=pad_to_len,
                    )
                    sample_scores.append(score_fn(output, answers))
                    sample_outputs.append(output)
                    sample_thinking.append(thinking)

            except Exception as e:
                if jax.process_index() == 0:
                    print(f"[RULER] ERROR task={task} ctx={ctx_len}: {e}")
                sample_scores   = sample_scores   or [0.0]
                sample_outputs  = sample_outputs  or [f"ERROR: {e}"]
                sample_thinking = sample_thinking or [""]

            cell = {
                "context_length": ctx_len,
                "score":          float(np.mean(sample_scores)),
                "sample_scores":  sample_scores,
                "outputs":        sample_outputs,
            }
            if any(sample_thinking):
                cell["thinking"] = sample_thinking
            all_results[task].append(cell)
            pbar.update(1)

            if jax.process_index() == 0:
                _save(out_path, "qwen3_mem_embed", tasks, context_lengths, all_results)

    pbar.close()

    from models.retrieval_ops import set_global_mesh
    set_global_mesh(None)

    avg_score = float(np.mean([
        r["score"]
        for task_results in all_results.values()
        for r in task_results
    ]))

    if jax.process_index() == 0:
        _save(out_path, "qwen3_mem_embed", tasks, context_lengths, all_results)
        print(f"[RULER] Results saved → {out_path}")
        png_path = out_path.replace(".json", ".png")
        _save_heatmap(png_path, "qwen3_mem_embed", tasks, context_lengths, all_results, avg_score)
        print(f"[RULER] Heatmap saved  → {png_path}")
        print(f"[RULER] Average score: {avg_score:.3f}")
        for task in tasks:
            task_avg = float(np.mean([r["score"] for r in all_results[task]]))
            print(f"  {task}: {task_avg:.3f}")

    return {
        "ruler_score": avg_score,
        **{f"ruler_{t}": float(np.mean([r["score"] for r in all_results[t]])) for t in tasks},
    }


class RULEREvaluator:
    """Wraps the module-level evaluate() so it plugs into the eval.py / eval_worker pipeline."""

    def __init__(self, cfg, key=None):
        self.cfg = cfg
        self.key = key or "ruler"

    def evaluate(self, model, dataset, step=None, **kwargs):
        # Resolve output dir using the same path logic as other evaluators so
        # eval_worker can find the file when building the W&B artifact manifest.
        base_output_dir = os.environ.get("EVAL_OUTPUT_DIR", os.getcwd())
        if step is not None:
            output_dir = os.path.join(base_output_dir, "eval_results", f"step_{step}", self.key)
        else:
            output_dir = os.path.join(base_output_dir, "eval_results", self.key)
        os.makedirs(output_dir, exist_ok=True)

        cfg = self.cfg

        return evaluate(
            model=model,
            tasks=list(cfg.get("tasks", ["niah_s", "niah_mk", "niah_mv", "niah_mq",
                                          "vt", "cwe", "fwe", "qa_squad", "qa_hotpot"])),
            context_lengths=list(cfg.get("context_lengths", [4096, 8192, 16384, 32768, 65536, 131072])),
            num_samples=cfg.get("num_samples", 500),
            chunk_size=cfg.get("chunk_size", 256),
            chunk_stride=cfg.get("chunk_stride", 196),
            vt_chunk_stride=cfg.get("vt_chunk_stride", 128),
            embed_batch_size=cfg.get("embed_batch_size", 64),
            output_dir=output_dir,
            haystack_cache_dir=cfg.get("haystack_cache_dir", "/tmp/pg19_haystack"),
            haystack_max_chars=cfg.get("haystack_max_chars", _PG19_DEFAULT_MAX_CHARS),
            num_needles=cfg.get("num_needles", 4),
            num_values=cfg.get("num_values", 4),
            num_queries=cfg.get("num_queries", 4),
            num_hops=cfg.get("num_hops", 4),
            num_cw=cfg.get("num_cw", 10),
            freq_cw=cfg.get("freq_cw", 30),
            freq_ucw=cfg.get("freq_ucw", 3),
        )
