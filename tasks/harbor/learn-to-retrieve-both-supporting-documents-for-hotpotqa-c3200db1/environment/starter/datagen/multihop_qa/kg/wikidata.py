"""Wikidata-based path sampler. Returns paths as (wiki_title, prop_label, wiki_title) triples."""
import random
import time
import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "multihop-qa-research/1.0", "Accept": "application/sparql-results+json"}

# Curated properties: no artifact nodes, clear semantics, good for multi-hop
GOOD_PROPS = {
    "P19": "birth place",
    "P20": "death place",
    "P22": "father",
    "P25": "mother",
    "P26": "spouse",
    "P27": "country of citizenship",
    "P40": "child",
    "P57": "director",
    "P69": "educated at",
    "P108": "employer",
    "P112": "founded by",
    "P131": "located in",
    "P155": "predecessor",
    "P156": "successor",
    "P159": "headquarters",
    "P166": "award received",
    "P184": "doctoral advisor",
    "P185": "doctoral student",
    "P17":  "country",
    "P36":  "capital",
    "P6":   "head of government",
    "P35":  "head of state",
    "P194": "legislative body",
    "P276": "location",
    "P361": "part of",
    "P495": "country of origin",
    "P607": "conflict",
    "P1066":"student of",
    "P101": "field of work",
    "P463": "member of",
    "P50":  "author",
    "P170": "creator",
    "P175": "performer",
    "P264": "record label",
}

PROPS_VALUES = " ".join(f"wdt:{p}" for p in GOOD_PROPS)


def _sparql(query, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(SPARQL_URL, params={"query": query, "format": "json"},
                             headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def wikipedia_title_to_qid(title):
    """Albert_Einstein -> Q937"""
    query = f"""
    SELECT ?item WHERE {{
      <https://en.wikipedia.org/wiki/{title}> schema:about ?item .
    }} LIMIT 1
    """
    res = _sparql(query)
    if res and res["results"]["bindings"]:
        return res["results"]["bindings"][0]["item"]["value"].split("/")[-1]
    return None


def qid_to_wikipedia_title(qid):
    """Q937 -> Albert_Einstein"""
    query = f"""
    SELECT ?title WHERE {{
      ?article schema:about wd:{qid} ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?title .
    }} LIMIT 1
    """
    res = _sparql(query)
    if res and res["results"]["bindings"]:
        return res["results"]["bindings"][0]["title"]["value"].replace(" ", "_")
    return None


def get_neighbors(qid, limit=25):
    """Return list of (qid_src, prop_id, prop_label, qid_dst, wiki_title_dst)."""
    values_clause = " ".join(f"wdt:{p}" for p in GOOD_PROPS)
    query = f"""
    SELECT ?prop ?obj ?objWikiTitle WHERE {{
      VALUES ?prop {{ {values_clause} }}
      wd:{qid} ?prop ?obj .
      FILTER(isIRI(?obj))
      ?article schema:about ?obj ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?objWikiTitle .
    }} LIMIT {limit}
    """
    res = _sparql(query)
    neighbors = []
    if not res:
        return neighbors
    for r in res["results"]["bindings"]:
        prop_uri = r["prop"]["value"]
        prop_id  = prop_uri.split("/")[-1]
        prop_label = GOOD_PROPS.get(prop_id, prop_id)
        obj_qid  = r["obj"]["value"].split("/")[-1]
        wiki_title = r["objWikiTitle"]["value"].replace(" ", "_")
        neighbors.append((qid, prop_label, obj_qid, wiki_title))
    return neighbors


def sample_path(start_qid, start_title, hops=3):
    """
    Returns list of {"subject": wiki_title, "relation": label, "object": wiki_title}
    or None on failure.
    """
    path = []
    current_qid = start_qid
    current_title = start_title
    visited_qids = {current_qid}

    for _ in range(hops):
        neighbors = get_neighbors(current_qid)
        neighbors = [n for n in neighbors if n[2] not in visited_qids]
        if not neighbors:
            return None
        _, prop_label, next_qid, next_title = random.choice(neighbors)
        path.append({"subject": current_title, "relation": prop_label, "object": next_title})
        visited_qids.add(next_qid)
        current_qid = next_qid
        current_title = next_title

    return path
