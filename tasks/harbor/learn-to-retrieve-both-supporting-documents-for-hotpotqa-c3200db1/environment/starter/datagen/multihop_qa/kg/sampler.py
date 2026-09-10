import random
from kg.dbpedia import get_neighbors, get_random_entity

# Relations whose objects are compound/non-article entities that never resolve
# to a Wikipedia page (e.g. "Chicago Storm (soccer)  Todd Short  1").
BAD_RELATIONS = {
    "http://dbpedia.org/ontology/careerStation",
    "http://dbpedia.org/ontology/currentMember",
    "http://dbpedia.org/ontology/termPeriod",
    "http://dbpedia.org/ontology/timeZone",
}

def sample_path(start, hops=6):
    path = []
    current = start
    visited = set([current])
    for _ in range(hops):
        try:
            neighbors = get_neighbors(current)
        except Exception:
            return None
        neighbors = [t for t in neighbors if t[2] not in visited and t[1] not in BAD_RELATIONS and "__" not in t[2].split("/resource/")[-1]]
        if not neighbors:
            return None
        triple = random.choice(neighbors)
        path.append(triple)
        current = triple[2]
        visited.add(current)
    return path

def sample_paths(start=None, num_paths=None, hops=None):
    from config import NUM_PATHS, HOPS
    if num_paths is None:
        num_paths = NUM_PATHS
    if hops is None:
        hops = HOPS
    paths = []
    attempts = 0
    while len(paths) < num_paths and attempts < num_paths * 10:
        node = start if start is not None else get_random_entity()
        if node is None:
            break
        p = sample_path(node, hops)
        if p:
            paths.append(p)
        attempts += 1
    return paths
