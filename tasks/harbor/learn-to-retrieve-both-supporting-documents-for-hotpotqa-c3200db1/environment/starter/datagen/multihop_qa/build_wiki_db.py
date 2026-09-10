#!/usr/bin/env python3
"""
Ingest English Wikipedia into a local SQLite database.

Downloads (if not already present):
  <data_dir>/enwiki-latest-pages-articles.xml.bz2   (~22 GB compressed)

Output:
  <data_dir>/wiki.db

Usage:
  python scripts/build_wiki_db.py
  python scripts/build_wiki_db.py --data-dir /mnt/data
  python scripts/build_wiki_db.py --limit 100000   # for testing
  python scripts/build_wiki_db.py --workers 50
"""
import argparse
import bz2
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request
import zlib
from multiprocessing import Pool, cpu_count

WIKI_DUMP_URL  = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
WIKI_DUMP_SIZE = 22 * 1024 ** 3  # ~22 GB
WRITE_BATCH    = 10000


# ── download ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > WIKI_DUMP_SIZE * 0.99:
        print(f"  already exists ({os.path.getsize(dest)/1e9:.1f} GB), skipping.")
        return
    if os.path.exists(dest):
        print(f"  partial file found ({os.path.getsize(dest)/1e9:.1f} GB), re-downloading...")
        os.remove(dest)

    # aria2c: 4 connections (Wikimedia throttles >4)
    if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
        print(f"  downloading with aria2c (4 connections): {url}")
        r = subprocess.run([
            "aria2c",
            "--split=4", "--max-connection-per-server=4", "--min-split-size=200M",
            "--file-allocation=none", "--continue=true",
            "-o", os.path.basename(dest), "-d", os.path.dirname(dest),
            url,
        ])
        if r.returncode == 0:
            return
        print("  aria2c failed, falling back to wget...")

    # wget fallback
    if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
        print(f"  downloading with wget: {url}")
        subprocess.run(["wget", "-c", "--progress=bar:force", "-O", dest, url], check=True)
        return

    # urllib last resort
    print(f"  downloading with urllib (~22 GB): {url}")
    def _progress(count, block, total):
        if total > 0:
            pct = min(count * block / total * 100, 100)
            print(f"\r  {pct:5.1f}%  ({count * block / 1e9:.2f} GB)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


# ── article processing (runs in worker processes) ─────────────────────────────

def _process_article(args):
    title, wikitext = args
    intro      = _extract_intro(wikitext)
    compressed = zlib.compress(wikitext.encode("utf-8"), level=1)
    return (title, intro, compressed)


def _extract_intro(wikitext: str) -> str:
    intro_raw = re.split(r"\n==", wikitext)[0]
    try:
        import mwparserfromhell
        text = mwparserfromhell.parse(intro_raw).strip_code()
    except Exception:
        text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", intro_raw)
        text = re.sub(r"\{\{[^}]*\}\}", "", text)
        text = re.sub(r"'{2,}", "", text)
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── db ────────────────────────────────────────────────────────────────────────

def _make_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA page_size=8192")
    conn.execute("PRAGMA cache_size=-2000000")      # 2 GB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=10737418240")    # 10 GB mmap
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            title    TEXT PRIMARY KEY,
            intro    TEXT NOT NULL,
            wikitext BLOB NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS redirects (
            from_title TEXT PRIMARY KEY,
            to_title   TEXT NOT NULL
        )
    """)
    return conn


# ── ingestion ─────────────────────────────────────────────────────────────────

def ingest(conn: sqlite3.Connection, dump_path: str, limit: int | None, workers: int) -> None:
    try:
        import mwxml
    except ImportError:
        print("ERROR: mwxml is required.  pip install mwxml")
        sys.exit(1)

    page_batch     = []
    redirect_batch = []
    total = 0

    # Single-pass generator: yields article (title, wikitext) tuples for the pool,
    # collects redirects as a side-effect into redirect_batch.
    def _article_stream():
        dump = mwxml.Dump.from_file(bz2.open(dump_path, "rb"))
        seen = 0
        for page in dump.pages:
            if page.namespace != 0:
                continue
            revision = next(iter(page), None)
            if revision is None or not revision.text:
                continue
            wikitext = revision.text
            if wikitext.strip().lower().startswith("#redirect"):
                m = re.search(r'#redirect\s*\[\[([^\]|#]+)', wikitext, re.IGNORECASE)
                if m:
                    redirect_batch.append((page.title, m.group(1).strip()))
                continue
            yield (page.title, wikitext)
            seen += 1
            if limit and seen >= limit:
                break

    # imap_unordered: all workers stay busy — no blocking on batch boundaries
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_process_article, _article_stream(), chunksize=200):
            page_batch.append(result)
            if len(page_batch) >= WRITE_BATCH:
                conn.executemany("INSERT OR REPLACE INTO pages VALUES (?,?,?)", page_batch)
                conn.commit()
                total += len(page_batch)
                page_batch.clear()
                print(f"\r  {total:>10,} pages", end="", flush=True)

    if page_batch:
        conn.executemany("INSERT OR REPLACE INTO pages VALUES (?,?,?)", page_batch)
        total += len(page_batch)
    if redirect_batch:
        conn.executemany("INSERT OR REPLACE INTO redirects VALUES (?,?)", redirect_batch)
    conn.commit()

    print(f"\r  {total:,} pages inserted")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--limit",   type=int, default=None, help="stop after N articles (for testing)")
    parser.add_argument("--workers", type=int, default=150,  help="parallel worker processes for ingestion")
    args = parser.parse_args()

    data_dir  = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    dump_path  = os.path.join(data_dir, "enwiki-latest-pages-articles.xml.bz2")
    db_path    = os.path.join(data_dir, "wiki.db")
    db_shm_path = "/dev/shm/wiki.db"

    print("=== build_wiki_db ===")
    print(f"output:  {db_path}")
    print(f"workers: {args.workers}\n")

    print("[1/2] downloading Wikipedia dump...")
    _download(WIKI_DUMP_URL, dump_path)

    # Copy to /dev/shm (RAM) for zero-cost reads during ingestion
    shm_path = os.path.join("/dev/shm", os.path.basename(dump_path))
    if not os.path.exists(shm_path) or os.path.getsize(shm_path) != os.path.getsize(dump_path):
        print(f"  copying dump to RAM ({os.path.getsize(dump_path)/1e9:.1f} GB → /dev/shm)...")
        import shutil
        shutil.copy2(dump_path, shm_path)
        print("  done.")
    else:
        print(f"  already in RAM: {shm_path}")
    dump_path = shm_path

    print(f"[2/2] ingesting articles with {args.workers} workers (writing to RAM: {db_shm_path})...")
    conn = _make_db(db_shm_path)
    ingest(conn, dump_path, limit=args.limit, workers=args.workers)
    conn.close()

    size_gb = os.path.getsize(db_shm_path) / 1e9
    print(f"\ndone — {db_shm_path}  ({size_gb:.1f} GB)")
    print(f"export WIKI_DB={db_shm_path}")


if __name__ == "__main__":
    main()
