"""
DBpedia interface — uses a local SQLite database when DBPEDIA_DB is set,
falls back to the public SPARQL endpoint otherwise.
"""
import os
import random
import sqlite3
import threading
import time

_DB_PATH = os.getenv("DBPEDIA_DB")

# Per-thread connections — eliminates "database is locked" under concurrent workers
_local = threading.local()

# Pre-sampled entity pool — avoids full-table-scan ORDER BY RANDOM() on every call
_entity_pool: list[str] = []
_entity_pool_lock = threading.Lock()
_POOL_SIZE = 200_000


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        conn = _local.conn
    return conn


def _ensure_entity_pool():
    """Load entity pool once at first use (one full-scan amortized across all calls)."""
    global _entity_pool
    if _entity_pool:
        return
    with _entity_pool_lock:
        if _entity_pool:
            return
        if not _DB_PATH:
            return
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        # Fast probabilistic sample: keep ~1-in-30 rows via modulo on rowid,
        # then filter compound entities in Python and shuffle.
        rows = conn.execute(
            "SELECT subject FROM triples WHERE rowid % 30 = 0"
        ).fetchall()
        conn.close()
        pool = list({
            r["subject"] for r in rows
            if "__" not in r["subject"].split("/resource/")[-1]
        })
        random.shuffle(pool)
        _entity_pool = pool[:_POOL_SIZE]


# ── Public API ────────────────────────────────────────────────────────────────

def get_neighbors(entity: str, relations: set[str] | None = None, limit: int = 30) -> list[tuple[str, str, str]]:
    if _DB_PATH:
        try:
            conn = _get_conn()
            if relations is None:
                rows = conn.execute(
                    "SELECT subject, predicate, object FROM triples WHERE subject = ? LIMIT ?",
                    [entity, limit],
                ).fetchall()
            else:
                placeholders = ",".join("?" * len(relations))
                rows = conn.execute(
                    f"SELECT subject, predicate, object FROM triples "
                    f"WHERE subject = ? AND predicate IN ({placeholders}) LIMIT ?",
                    [entity, *relations, limit],
                ).fetchall()
            return [(r["subject"], r["predicate"], r["object"]) for r in rows]
        except Exception:
            return []
    else:
        from config import GOOD_RELATIONS
        return _get_neighbors_remote(entity, relations or GOOD_RELATIONS, limit)


def get_random_entity() -> str | None:
    """Return a random entity from the pre-sampled pool (O(1), no DB hit per call)."""
    if not _DB_PATH:
        return None
    _ensure_entity_pool()
    return random.choice(_entity_pool) if _entity_pool else None


def get_abstract(entity: str) -> str | None:
    if _DB_PATH:
        try:
            row = _get_conn().execute(
                "SELECT abstract FROM abstracts WHERE entity = ?", [entity]
            ).fetchone()
            return row["abstract"] if row else None
        except Exception:
            return None
    else:
        return _get_abstract_remote(entity)


# ── Remote fallbacks ──────────────────────────────────────────────────────────

def _sparql_query(q: str, retries: int = 3):
    from SPARQLWrapper import SPARQLWrapper, JSON
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    for attempt in range(retries):
        try:
            return sparql.query().convert()
        except Exception:
            time.sleep(2 ** attempt)
    return None


def _get_neighbors_remote(entity: str, relations: set[str], limit: int) -> list[tuple[str, str, str]]:
    q = f"""
    SELECT ?rel ?obj WHERE {{
      <{entity}> ?rel ?obj .
      FILTER(isIRI(?obj))
      FILTER(strstarts(str(?obj), 'http://dbpedia.org/resource/'))
      FILTER(!strstarts(str(?obj), 'http://dbpedia.org/resource/Category:'))
      FILTER(strstarts(str(?rel), 'http://dbpedia.org/ontology/'))
      FILTER(!contains(str(?rel), 'wikiPage'))
      FILTER(!contains(str(?rel), 'thumbnail'))
      FILTER(!contains(str(?rel), 'abstract'))
      FILTER(!contains(str(?rel), 'description'))
    }} LIMIT {limit}
    """
    res = _sparql_query(q)
    if not res:
        return []
    return [
        (entity, r["rel"]["value"], r["obj"]["value"])
        for r in res["results"]["bindings"]
    ]


def _get_abstract_remote(entity: str) -> str | None:
    q = f"""
    SELECT ?abstract WHERE {{
      <{entity}> <http://dbpedia.org/ontology/abstract> ?abstract .
      FILTER (lang(?abstract) = 'en')
    }} LIMIT 1
    """
    res = _sparql_query(q)
    if res and res["results"]["bindings"]:
        return res["results"]["bindings"][0]["abstract"]["value"]
    return None
