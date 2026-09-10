"""
Fetch a Wikipedia page by page ID and a specific revision using Wikipedia-API.

Wikipedia-API works with titles, so we first resolve the page ID to a title
via the MediaWiki API, then fetch the specific revision's wikitext directly.
"""

import requests
import wikipediaapi

PAGE_ID = 18468611
REV_ID = 1172527864
API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "memory-layers-test/1.0 (research)"}


def resolve_title(page_id: int) -> str:
    resp = requests.get(API_URL, headers=HEADERS, params={
        "action": "query",
        "pageids": page_id,
        "format": "json",
    })
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    return pages[str(page_id)]["title"]


def fetch_revision_text(rev_id: int) -> str:
    resp = requests.get(API_URL, headers=HEADERS, params={
        "action": "query",
        "revids": rev_id,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    })
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page["revisions"][0]["slots"]["main"]["*"]


if __name__ == "__main__":
    # Resolve title and fetch current page summary via Wikipedia-API
    title = resolve_title(PAGE_ID)
    print(f"Page title: {title}")

    wiki = wikipediaapi.Wikipedia(user_agent="memory-layers-test/1.0", language="en")
    page = wiki.page(title)
    print(f"Summary:\n{page.summary[:500]}\n")

    # Fetch the specific revision's full wikitext
    wikitext = fetch_revision_text(REV_ID)
    print(f"Revision {REV_ID} wikitext:\n{wikitext}")
