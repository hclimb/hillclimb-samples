#!/usr/bin/env python3
"""
Ingest DBpedia triples and abstracts into a local SQLite database.

Downloads (if not already present):
  <data_dir>/mappingbased-objects_en.ttl.bz2   (~1.5 GB)
  <data_dir>/long-abstracts_en.ttl.bz2          (~1.5 GB)

Output:
  <data_dir>/dbpedia.db   (~2 GB — all object-property triples + abstracts)

A `relation_stats` table is computed after ingestion. At generation time,
only relations with count >= MIN_RELATION_COUNT are used (data-driven whitelist).

Usage:
  python scripts/build_dbpedia_db.py
  python scripts/build_dbpedia_db.py --data-dir /mnt/data
  python scripts/build_dbpedia_db.py --min-count 500
"""
import argparse
import bz2
import os
import re
import sqlite3
import sys
import urllib.request

TRIPLES_URL   = "https://downloads.dbpedia.org/repo/dbpedia/mappings/mappingbased-objects/2022.09.01/mappingbased-objects_lang%3Den.ttl.bz2"
ABSTRACTS_URL = "https://downloads.dbpedia.org/repo/dbpedia/text/long-abstracts/2022.09.01/long-abstracts_lang%3Den.ttl.bz2"

# Skip navigation/media artifacts that are never useful for reasoning
_BLACKLIST_KEYWORDS = ("wikiPage", "thumbnail", "depiction", "logo", "image", "signature", "map")

_TRIPLE_RE   = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>')
_ABSTRACT_RE = re.compile(r'^<([^>]+)>\s+\S+abstract\S*\s+"(.*)"@en\s*\.\s*$')


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  already exists, skipping: {dest}")
        return
    print(f"  downloading {url}")

    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r  {pct:5.1f}%  ({count * block / 1e9:.2f} GB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _make_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA page_size=8192")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triples (
            subject   TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS abstracts (
            entity   TEXT PRIMARY KEY,
            abstract TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relation_stats (
            predicate TEXT PRIMARY KEY,
            count     INTEGER NOT NULL
        )
    """)
    return conn


def ingest_triples(conn: sqlite3.Connection, ttl_bz2: str) -> None:
    total, skipped = 0, 0
    batch = []

    with bz2.open(ttl_bz2, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            m = _TRIPLE_RE.match(line)
            if not m:
                skipped += 1
                continue
            subj, pred, obj = m.group(1), m.group(2), m.group(3)

            # Keep only DBpedia ontology object properties; skip navigation artifacts
            if not pred.startswith("http://dbpedia.org/ontology/"):
                continue
            if any(kw in pred for kw in _BLACKLIST_KEYWORDS):
                continue

            batch.append((subj, pred, obj))
            if len(batch) >= 10_000:
                conn.executemany("INSERT INTO triples VALUES (?,?,?)", batch)
                total += len(batch)
                batch.clear()
                print(f"\r  {total:>10,} triples", end="", flush=True)

    if batch:
        conn.executemany("INSERT INTO triples VALUES (?,?,?)", batch)
        total += len(batch)

    conn.commit()
    print(f"\r  {total:,} triples inserted  ({skipped:,} non-matching lines skipped)")


def build_relation_stats(conn: sqlite3.Connection) -> None:
    print("  computing relation frequencies...")
    conn.execute("DELETE FROM relation_stats")
    conn.execute("""
        INSERT INTO relation_stats (predicate, count)
        SELECT predicate, COUNT(*) FROM triples GROUP BY predicate
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_triples ON triples(subject, predicate)")
    conn.commit()

    rows = conn.execute(
        "SELECT predicate, count FROM relation_stats ORDER BY count DESC LIMIT 20"
    ).fetchall()
    print("  top 20 relations by coverage:")
    for pred, cnt in rows:
        short = pred.split("/")[-1]
        print(f"    {cnt:>8,}  {short}")


def ingest_abstracts(conn: sqlite3.Connection, ttl_bz2: str) -> None:
    total, skipped = 0, 0
    batch = []

    with bz2.open(ttl_bz2, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            m = _ABSTRACT_RE.match(line)
            if not m:
                skipped += 1
                continue
            entity, abstract = m.group(1), m.group(2)
            if not entity.startswith("http://dbpedia.org/resource/") or not abstract.strip():
                continue
            abstract = abstract.replace('\\"', '"').replace("\\n", "\n")
            batch.append((entity, abstract))
            if len(batch) >= 10_000:
                conn.executemany("INSERT OR REPLACE INTO abstracts VALUES (?,?)", batch)
                total += len(batch)
                batch.clear()
                print(f"\r  {total:>10,} abstracts", end="", flush=True)

    if batch:
        conn.executemany("INSERT OR REPLACE INTO abstracts VALUES (?,?)", batch)
        total += len(batch)

    conn.commit()
    print(f"\r  {total:,} abstracts inserted  ({skipped:,} non-matching lines skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--min-count", type=int, default=1000,
                        help="relations with fewer triples than this are excluded at generation time (default: 1000)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    triples_path   = os.path.join(data_dir, "mappingbased-objects_en.ttl.bz2")
    abstracts_path = os.path.join(data_dir, "long-abstracts_en.ttl.bz2")
    db_path        = os.path.join(data_dir, "dbpedia.db")

    print("=== build_dbpedia_db ===")
    print(f"output: {db_path}\n")

    print("[1/5] downloading triples dump...")
    _download(TRIPLES_URL, triples_path)

    print("[2/5] downloading abstracts dump...")
    _download(ABSTRACTS_URL, abstracts_path)

    conn = _make_db(db_path)

    print("[3/5] ingesting triples (all dbo: object properties)...")
    ingest_triples(conn, triples_path)

    print("[4/5] computing relation stats + building index...")
    build_relation_stats(conn)

    print("[5/5] ingesting abstracts...")
    ingest_abstracts(conn, abstracts_path)

    # Persist the min_count threshold so dbpedia.py can read it
    conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT OR REPLACE INTO config VALUES ('min_relation_count', ?)", [str(args.min_count)])
    conn.commit()
    conn.close()

    size_gb = os.path.getsize(db_path) / 1e9
    print(f"\ndone — {db_path}  ({size_gb:.1f} GB)")
    print(f"  relations with count >= {args.min_count} will be used at generation time")
    print(f"\nexport DBPEDIA_DB={db_path}")


if __name__ == "__main__":
    main()
