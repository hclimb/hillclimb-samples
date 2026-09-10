import os
import re
import sqlite3
import sys
import threading
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_WIKI_DB_PATH = os.getenv("WIKI_DB")

# Per-thread connections — eliminates SQLite "database is locked" under concurrent workers
_local = threading.local()


def _get_wiki_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _local.conn = sqlite3.connect(_WIKI_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        conn = _local.conn
    return conn


def _normalize(title: str) -> str:
    return title.replace("_", " ")


# ── Wikipedia helpers ──────────────────────────────────────────────────────────

def entity_to_title(uri):
    return uri.split("/")[-1]


def _resolve_title(title: str) -> str:
    """Follow redirect chain (up to 3 levels) in the local DB."""
    conn = _get_wiki_conn()
    current = _normalize(title)
    for _ in range(3):
        row = conn.execute(
            "SELECT to_title FROM redirects WHERE from_title = ?", [current]
        ).fetchone()
        if not row:
            break
        current = _normalize(row["to_title"])
    return current


def get_intro(title: str) -> str:
    if _WIKI_DB_PATH:
        try:
            conn = _get_wiki_conn()
            norm = _normalize(title)
            row = conn.execute("SELECT intro FROM pages WHERE title = ?", [norm]).fetchone()
            if not row:
                norm = _resolve_title(title)
                row = conn.execute("SELECT intro FROM pages WHERE title = ?", [norm]).fetchone()
            return row["intro"] if row else ""
        except Exception:
            return ""
    else:
        return _get_intro_remote(title)


def get_wikitext(title: str) -> str | None:
    if _WIKI_DB_PATH:
        try:
            conn = _get_wiki_conn()
            norm = _normalize(title)
            row = conn.execute("SELECT wikitext FROM pages WHERE title = ?", [norm]).fetchone()
            if not row:
                norm = _resolve_title(title)
                row = conn.execute("SELECT wikitext FROM pages WHERE title = ?", [norm]).fetchone()
            if not row:
                return None
            return zlib.decompress(row["wikitext"]).decode("utf-8")
        except Exception:
            return None
    else:
        return _get_wikitext_remote(title)


def _get_intro_remote(title: str) -> str:
    try:
        import requests
        headers = {"User-Agent": "multihop-qa-research/1.0 (research project)"}
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("extract", "")
    except Exception:
        return ""


def _get_wikitext_remote(title: str) -> str | None:
    try:
        import requests
        headers = {"User-Agent": "multihop-qa-research/1.0 (research project)"}
        params = {
            "action": "parse",
            "page": _normalize(title),
            "prop": "wikitext",
            "format": "json",
        }
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params,
                         headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["parse"]["wikitext"]["*"]
    except Exception:
        return None


def _clean(para):
    para = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", para)
    para = re.sub(r"\{\{[^}]*\}\}", "", para)
    para = re.sub(r"'{2,}", "", para)
    para = re.sub(r"<ref[^>]*>.*?</ref>", "", para, flags=re.DOTALL)
    para = re.sub(r"<[^>]+>", "", para)
    return para.strip()


def anchor_linked_paragraphs(source_title, target_title):
    wikitext = get_wikitext(source_title)
    if not wikitext:
        return []

    target_norm   = target_title.replace(" ", "_").lower()
    target_spaced = target_title.replace("_", " ").lower()

    results = []
    for para in wikitext.split("\n"):
        raw_links = re.findall(r"\[\[([^\]|#]+)", para)
        for link in raw_links:
            if link.strip().replace(" ", "_").lower() in (target_norm, target_spaced):
                clean = _clean(para)
                if len(clean) > 50:
                    results.append(clean)
                break
    return results


# ── LLM helpers ───────────────────────────────────────────────────────────────

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lm import chat as _chat_shared

def _chat(system, prompt, max_new_tokens=64):
    return _chat_shared(system, prompt, max_new_tokens=max_new_tokens)


def llm_judges_paragraph(paragraph, source, target, relation):
    try:
        rel_name = relation.split("/")[-1].replace("_", " ")
        prompt = (
            f'Does this paragraph establish that "{source.replace("_"," ")}"\'s {rel_name} is "{target.replace("_"," ")}"?\n\n'
            f'Paragraph: {paragraph}\n\n'
            f'Reply with only "yes" or "no".'
        )
        reply = _chat("You are a precise fact-checker. Reply only 'yes' or 'no'.", prompt, max_new_tokens=8)
        return reply.lower().startswith("yes")
    except Exception:
        return False


def llm_write_grounding_sentence(source, relation, target):
    try:
        rel_name = relation.split("/")[-1].replace("_", " ")
        prompt = (
            f'Write exactly one natural English sentence that states: '
            f'"{source.replace("_"," ")}" has {rel_name} "{target.replace("_"," ")}".\n'
            f'Be concise and factual.'
        )
        return _chat("You write short factual sentences.", prompt, max_new_tokens=64)
    except Exception:
        return ""


# ── Main entry point ───────────────────────────────────────────────────────────

def get_paragraph_with_link(source_title, target_title, relation=None):
    intro      = get_intro(source_title)
    candidates = anchor_linked_paragraphs(source_title, target_title)

    for candidate in candidates:
        if llm_judges_paragraph(candidate, source_title, target_title, relation or ""):
            if candidate not in intro:
                return intro + "\n\n" + candidate
            return intro

    sentence = llm_write_grounding_sentence(source_title, relation or "", target_title)
    if sentence:
        return intro + "\n\n" + sentence
    return intro
