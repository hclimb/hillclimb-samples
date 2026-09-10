#!/usr/bin/env python3
"""
Stage 0 of multihop hard-negative mining: build the deduplicated document corpus.

ragrawal36/multihop_qa_sft packs every row's supporting paragraphs into one string
field (`paragraphs`) joined by "<|doc_separator|>" (they are written by
datagen/generate_multihop_sft.py, which flattens a {entity: abstract} dict). All of a
row's paragraphs are POSITIVE docs for that row's question; the dataset ships no
negatives. Before negatives can be mined we need a flat corpus of unique docs with
stable ids, plus each row's positive ids.

Dedup runs an exact pass by default, and an OPTIONAL fuzzy pass (--fuzzy):

  1. exact   — hash of whitespace-collapsed, casefolded text. Always on.
  2. fuzzy   — bucket by the first --prefix-words normalised words, then inside each
               multi-member bucket verify with a 5-word-shingle Jaccard and merge at
               >= --jaccard. OFF by default; see the warning below.

*** Why --fuzzy is OFF by default ***

Near-identical docs in this dataset are NOT Wikipedia revision variants. They are the
same article carrying a DIFFERENT injected bridge fact -- one per hop of the row's
reasoning chain, matching the row's `path` field. Example (row 829): the same "3rd
Strike" article appears twice, once ending

    "Jim Korthe is associated with the band 3rd Strike."     (path hop 1)

and once ending

    "3rd Strike is a rap metal band."                        (path hop 2)

Measured over merge groups in a 3000-row sample: 100% had members with DIFFERENT final
lines, and 0% were pure duplicates. So fuzzy-merging silently discards the bridge fact
of every variant it absorbs -- and the bridge fact is exactly the evidence that makes
the hop answerable.

It also means a same-article variant carrying a different bridge fact is a legitimate
(and usefully hard) negative for a row whose positive carries the needed one, rather
than a mislabelled positive. Enable --fuzzy only if you have re-verified that on your
data.

Runs on CPU; no TPU needed.

Usage:
    python datagen/multihop_extract_and_dedup_docs.py --max-rows 5000   # smoke test
    python datagen/multihop_extract_and_dedup_docs.py                   # full stream
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SOURCE_DATASET = "ragrawal36/multihop_qa_sft"
DOC_SEPARATOR = "<|doc_separator|>"

_WS = re.compile(r"\s+")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=SOURCE_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stop after N source rows (smoke test). Default: whole split.")
    p.add_argument("--fuzzy", action="store_true",
                   help="Enable the fuzzy near-duplicate merge pass. OFF by default: on this "
                        "dataset near-identical docs differ by their injected per-hop bridge "
                        "fact, so merging them destroys hop evidence (see module docstring).")
    p.add_argument("--prefix-words", type=int, default=50,
                   help="Bucket key = first N normalised words. Only used with --fuzzy.")
    p.add_argument("--jaccard", type=float, default=0.8,
                   help="5-word-shingle Jaccard threshold to merge inside a prefix bucket.")
    p.add_argument("--shingle", type=int, default=5, help="Shingle size in words.")
    p.add_argument("--max-bucket", type=int, default=200,
                   help="Prefix buckets larger than this are reported and merged by prefix "
                        "alone (guards a pathological all-pairs blowup).")
    p.add_argument("--out-dir", default="data/multihop_doc_corpus")
    return p


def normalize(text):
    """Whitespace-collapsed, casefolded form used for both dedup passes."""
    return _WS.sub(" ", text).strip().casefold()


def split_docs(paragraphs):
    """Split a packed `paragraphs` field into individual docs.

    Matches data/utils.py::make_normalizer exactly so this corpus stays consistent with
    how data/qa.py parses the same field at training time.
    """
    if not isinstance(paragraphs, str):
        return []
    return [d.strip() for d in paragraphs.split(DOC_SEPARATOR) if d.strip()]


def shingles(normalized_text, k):
    words = normalized_text.split()
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def stream_and_exact_dedup(args):
    """Pass 1. Stream the split, split packed docs, collapse byte-identical duplicates.

    Returns (uniq_texts, rows) where rows carries each source row's question/answer plus
    its doc indices into uniq_texts.

    Keys the dedup map on a 128-bit digest rather than the normalised text: at ~10^6
    unique multi-KB docs, retaining a second normalised copy of every doc costs GBs for
    no benefit. Pass 2 recomputes normalize() over unique docs only, which is cheap.
    """
    from datasets import load_dataset

    log.info(f"streaming {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    uniq_texts = []
    by_norm_hash = {}
    rows = []
    n_instances = 0
    docs_per_row = Counter()

    for i, ex in enumerate(ds):
        if args.max_rows is not None and i >= args.max_rows:
            break

        docs = split_docs(ex.get("paragraphs", ""))
        docs_per_row[len(docs)] += 1
        ids = []
        for text in docs:
            n_instances += 1
            key = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).digest()
            idx = by_norm_hash.get(key)
            if idx is None:
                idx = len(uniq_texts)
                by_norm_hash[key] = idx
                uniq_texts.append(text)
            else:
                # Keep the longest surviving variant as the representative.
                if len(text) > len(uniq_texts[idx]):
                    uniq_texts[idx] = text
            if idx not in ids:
                ids.append(idx)

        rows.append({
            "row_id": i,
            "question": str(ex.get("question", "")),
            "answer": str(ex.get("answer", "")),
            "exact_ids": ids,
        })

        if (i + 1) % 100_000 == 0:
            log.info(f"  {i + 1:,} rows | {n_instances:,} instances | {len(uniq_texts):,} unique")

    return uniq_texts, rows, n_instances, docs_per_row


def fuzzy_merge(uniq_texts, args):
    """Pass 2. Bucket by prefix, verify with Jaccard, merge revision variants.

    Returns (canonical_of, n_buckets_multi, n_oversized, n_merged) where canonical_of
    maps every exact-dedup index to its canonical index.
    """
    uniq_norms = [normalize(t) for t in uniq_texts]

    buckets = {}
    for idx, norm in enumerate(uniq_norms):
        key = " ".join(norm.split()[:args.prefix_words])
        buckets.setdefault(key, []).append(idx)

    canonical_of = list(range(len(uniq_texts)))
    n_multi = n_oversized = n_merged = 0

    for members in buckets.values():
        if len(members) < 2:
            continue
        n_multi += 1

        if len(members) > args.max_bucket:
            # Pathological bucket: trust the prefix alone rather than go quadratic.
            n_oversized += 1
            rep = max(members, key=lambda i: len(uniq_texts[i]))
            for m in members:
                if m != rep:
                    canonical_of[m] = rep
                    n_merged += 1
            continue

        # Greedy grouping against group representatives (not all pairs).
        groups = []  # list of [rep_idx, rep_shingles]
        for m in members:
            sh = shingles(uniq_norms[m], args.shingle)
            placed = False
            for g in groups:
                if jaccard(sh, g[1]) >= args.jaccard:
                    # Canonical = longest variant in the group.
                    if len(uniq_texts[m]) > len(uniq_texts[g[0]]):
                        canonical_of[g[0]] = m
                        for other in g[2]:
                            canonical_of[other] = m
                        g[2].append(g[0])
                        g[0], g[1] = m, sh
                    else:
                        canonical_of[m] = g[0]
                        g[2].append(m)
                    n_merged += 1
                    placed = True
                    break
            if not placed:
                groups.append([m, sh, []])

    # Path-compress (a rep may itself have been reassigned to a longer variant).
    for i in range(len(canonical_of)):
        root = i
        seen = set()
        while canonical_of[root] != root and root not in seen:
            seen.add(root)
            root = canonical_of[root]
        canonical_of[i] = root

    return canonical_of, n_multi, n_oversized, n_merged


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uniq_texts, rows, n_instances, docs_per_row = stream_and_exact_dedup(args)
    n_exact = len(uniq_texts)
    if n_exact == 0:
        log.error("no documents extracted -- check the separator and field name")
        return 1
    log.info(f"exact pass: {n_instances:,} instances -> {n_exact:,} unique")

    if args.fuzzy:
        canonical_of, n_multi, n_oversized, n_merged = fuzzy_merge(uniq_texts, args)
        log.info(f"fuzzy pass: merged {n_merged:,} near-duplicates")
    else:
        canonical_of = list(range(n_exact))
        n_multi = n_oversized = n_merged = 0
        log.info("fuzzy pass: skipped (--fuzzy not set); per-hop bridge facts preserved")

    # Assign final contiguous doc_ids to surviving canonical docs.
    doc_id_of = {}
    corpus_ids, corpus_texts = [], []
    for idx in range(n_exact):
        if canonical_of[idx] == idx:
            doc_id_of[idx] = len(corpus_ids)
            corpus_ids.append(doc_id_of[idx])
            corpus_texts.append(uniq_texts[idx])
    n_final = len(corpus_ids)

    pq.write_table(
        pa.table({"doc_id": pa.array(corpus_ids, pa.int32()),
                  "text": pa.array(corpus_texts, pa.string())}),
        out_dir / "doc_corpus.parquet",
    )

    # Remap each row's positives onto final doc_ids.
    row_ids, questions, answers, pos_ids_col = [], [], [], []
    pos_counts = Counter()
    for r in rows:
        pos = []
        for e in r["exact_ids"]:
            did = doc_id_of[canonical_of[e]]
            if did not in pos:
                pos.append(did)
        row_ids.append(r["row_id"])
        questions.append(r["question"])
        answers.append(r["answer"])
        pos_ids_col.append(pos)
        pos_counts[len(pos)] += 1

    pq.write_table(
        pa.table({"row_id": pa.array(row_ids, pa.int32()),
                  "question": pa.array(questions, pa.string()),
                  "answer": pa.array(answers, pa.string()),
                  "pos_doc_ids": pa.array(pos_ids_col, pa.list_(pa.int32()))}),
        out_dir / "row_pos_doc_ids.parquet",
    )

    lengths = sorted(len(t) for t in corpus_texts)
    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "max_rows": args.max_rows,
        "source_rows": len(rows),
        "doc_instances": n_instances,
        "unique_after_exact": n_exact,
        "unique_after_fuzzy": n_final,
        "collapse_ratio": round(1 - n_final / n_instances, 4) if n_instances else None,
        "fuzzy_enabled": args.fuzzy,
        "fuzzy_merged_docs": n_merged,
        "multi_member_prefix_buckets": n_multi,
        "oversized_buckets": n_oversized,
        "docs_per_row": dict(sorted(docs_per_row.items())),
        "pos_docs_per_row": dict(sorted(pos_counts.items())),
        "doc_chars": {
            "mean": round(sum(lengths) / len(lengths)),
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)],
            "max": lengths[-1],
        },
        "params": {"prefix_words": args.prefix_words, "jaccard": args.jaccard,
                   "shingle": args.shingle},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 62)
    print("STAGE 0 — deduplicated document corpus")
    print("=" * 62)
    print(f"  source rows            : {len(rows):,}")
    print(f"  doc instances          : {n_instances:,}")
    print(f"  unique after exact     : {n_exact:,}")
    if args.fuzzy:
        print(f"  unique after fuzzy     : {n_final:,}   (merged {n_merged:,} near-duplicates)")
        print(f"  multi-member buckets   : {n_multi:,}" + (f"  (oversized: {n_oversized})" if n_oversized else ""))
    else:
        print(f"  fuzzy pass             : skipped (bridge facts preserved)")
    print(f"  corpus docs            : {n_final:,}")
    print(f"  overall collapse       : {stats['collapse_ratio']:.1%}")
    print(f"  docs/row               : {stats['docs_per_row']}")
    print(f"  doc chars              : mean {stats['doc_chars']['mean']:,}  "
          f"p50 {stats['doc_chars']['p50']:,}  p90 {stats['doc_chars']['p90']:,}")
    print(f"\n  wrote {out_dir}/doc_corpus.parquet, row_pos_doc_ids.parquet, stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
