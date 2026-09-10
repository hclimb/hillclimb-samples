"""
Prepare NovelHopQA books dataset for memory-layers evaluation.

Fetches the full text for each of the 44 novels in NovelHopQA from Project
Gutenberg and uploads a small books dataset (one row per book) to HuggingFace.

The QA rows themselves are used directly from the original dataset
(abhaygupta1266/novelhopqa) — no need to re-upload those.

Usage:
    python data/utils/prepare_novelhopqa.py
    python data/utils/prepare_novelhopqa.py --output_dir ./data/cache/novelhopqa --hf_upload mihir-1999/novelhopqa-books
"""

import argparse
import json
import os
import re
import time

import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


# ---------------------------------------------------------------------------
# Gutenberg ID mapping — keys are the EXACT strings from the `book` column
# in abhaygupta1266/novelhopqa (verified from dataset output).
# El Filibusterismo has no reliable fixed ID so it falls back to gutendex.
# ---------------------------------------------------------------------------
GUTENBERG_IDS = {
    "A Connecticut Yankee in King Arthur's Court": 86,
    "Adventures of Huckleberry Finn": 76,
    "Anna Karenina": 1399,
    "Anne of Green Gables": 45,
    "Billy Budd": 76513,
    "Crime and Punishment": 2554,
    "David Copperfield": 766,
    "Dead Souls": 1081,
    "Don Quixote": 996,
    "Dracula": 345,
    "Emma": 158,
    "Great Expectations": 1400,
    "Jane Eyre": 1260,
    "Les Misérables": 135,
    "Little Women": 514,
    "Mansfield Park": 141,
    "Middlemarch": 145,
    "Moby Dick": 2701,
    "Noli Me Tangere": 6737,
    "Oliver Twist": 730,
    "Pride and Prejudice": 1342,
    "Resurrection": 1938,
    "Robinson Crusoe": 521,
    "Sense and Sensibility": 161,
    "Tess of the d'Urbervilles": 110,
    "The Adventures of Sherlock Holmes": 1661,
    "The Arabian Nights": 128,
    "The Brothers Karamazov": 28054,
    "The Count of Monte Cristo": 1184,
    "The Decameron": 23700,
    "The Idiot": 2638,
    "The Last Man": 18247,
    "The Last of the Mohicans": 27,
    "The Moonstone": 155,
    "The Portrait of a Lady": 2833,
    "The Three Musketeers": 1257,
    "This Side of Paradise": 805,
    "Twenty Thousand Leagues Under the Sea": 164,
    "Ulysses": 4300,
    "Uncle Tom's Cabin": 203,
    "Vanity Fair": 599,
    "War and Peace": 2600,
    "Wuthering Heights": 768,
    # El Filibusterismo — no fixed ID, resolved via gutendex fallback
}


def search_gutendex(title: str) -> int | None:
    """Search gutendex.com for a book by title and return its Gutenberg ID."""
    try:
        resp = requests.get(
            "https://gutendex.com/books/",
            params={"search": title},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  [gutendex] search failed for '{title}': {e}")
    return None


def get_gutenberg_id(title: str) -> int | None:
    """Look up Gutenberg ID: exact map first, then gutendex API fallback."""
    gid = GUTENBERG_IDS.get(title)
    if gid is not None:
        return gid
    print(f"  [lookup] '{title}' not in map, querying gutendex...")
    time.sleep(0.5)
    return search_gutendex(title)


def download_book_text(gutenberg_id: int) -> str | None:
    """Download plain-text of a book from Project Gutenberg."""
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]
    for url in candidates:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return _strip_gutenberg_boilerplate(resp.text)
        except Exception as e:
            print(f"  [download] {url} failed: {e}")
    return None


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def fetch_books(titles: list[str], cache_dir: str) -> list[dict]:
    """Fetch book texts for all titles, using a local file cache."""
    os.makedirs(cache_dir, exist_ok=True)
    rows = []

    for title in titles:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
        cache_path = os.path.join(cache_dir, f"{safe_name}.txt")

        if os.path.exists(cache_path):
            print(f"  [cache] {title}")
            with open(cache_path, "r", encoding="utf-8") as f:
                text = f.read()
            rows.append({"title": title, "text": text})
            continue

        gid = get_gutenberg_id(title)
        if gid is None:
            print(f"  [SKIP] No Gutenberg ID found for '{title}'")
            rows.append({"title": title, "text": ""})
            continue

        print(f"  [download] {title} (Gutenberg #{gid})")
        text = download_book_text(gid)
        if text:
            print(f"    → {len(text):,} chars")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
            rows.append({"title": title, "text": text})
        else:
            print(f"  [FAIL] Could not download '{title}'")
            rows.append({"title": title, "text": ""})

        time.sleep(0.3)

    return rows


def prepare(output_dir: str, hf_upload: str | None, hf_token: str | None):
    book_cache_dir = os.path.join(output_dir, "books")

    # Get unique book titles from all hops
    print("Loading book titles from novelhopqa...")
    all_titles = set()
    for split in ["hop_1", "hop_2", "hop_3", "hop_4"]:
        ds = load_dataset("abhaygupta1266/novelhopqa", split=split, token=hf_token)
        all_titles.update(ds["book"])
    titles = sorted(all_titles)
    print(f"Found {len(titles)} unique books: {titles}\n")

    print("Fetching books from Project Gutenberg...")
    rows = fetch_books(titles, book_cache_dir)

    missing = [r["title"] for r in rows if not r["text"]]
    if missing:
        print(f"\n[WARNING] Missing texts for: {missing}")

    books_dataset = Dataset.from_list(rows)
    print(f"\nBooks dataset: {len(books_dataset)} rows")

    local_path = os.path.join(output_dir, "books_dataset")
    books_dataset.save_to_disk(local_path)
    print(f"Saved to {local_path}")

    if hf_upload:
        print(f"\nUploading to {hf_upload}...")
        books_dataset.push_to_hub(hf_upload, token=hf_token)
        print(f"Done → https://huggingface.co/datasets/{hf_upload}")

    # Save id map for reference
    id_map = {t: get_gutenberg_id(t) for t in titles}
    with open(os.path.join(output_dir, "gutenberg_ids.json"), "w") as f:
        json.dump(id_map, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/cache/novelhopqa")
    hf_username = os.environ.get("HF_USERNAME", "")
    default_repo = f"{hf_username}/novelhopqa-books" if hf_username else None
    parser.add_argument("--hf_upload", default=default_repo)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    prepare(args.output_dir, args.hf_upload, token)
