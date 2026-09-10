import os

START_ENTITIES = [
    # Scientists
    "http://dbpedia.org/resource/Albert_Einstein",
    "http://dbpedia.org/resource/Marie_Curie",
    "http://dbpedia.org/resource/Isaac_Newton",
    "http://dbpedia.org/resource/Charles_Darwin",
    "http://dbpedia.org/resource/Nikola_Tesla",
    # Political figures
    "http://dbpedia.org/resource/Napoleon",
    "http://dbpedia.org/resource/Abraham_Lincoln",
    "http://dbpedia.org/resource/Winston_Churchill",
    "http://dbpedia.org/resource/Cleopatra",
    "http://dbpedia.org/resource/Genghis_Khan",
    # Artists / writers
    "http://dbpedia.org/resource/Leonardo_da_Vinci",
    "http://dbpedia.org/resource/William_Shakespeare",
    "http://dbpedia.org/resource/Ludwig_van_Beethoven",
    "http://dbpedia.org/resource/Pablo_Picasso",
    "http://dbpedia.org/resource/Wolfgang_Amadeus_Mozart",
    # Places
    "http://dbpedia.org/resource/Paris",
    "http://dbpedia.org/resource/Tokyo",
    "http://dbpedia.org/resource/Rome",
    "http://dbpedia.org/resource/Cairo",
    "http://dbpedia.org/resource/New_York_City",
    # Organizations / institutions
    "http://dbpedia.org/resource/United_Nations",
    "http://dbpedia.org/resource/Harvard_University",
    "http://dbpedia.org/resource/NASA",
    "http://dbpedia.org/resource/Olympic_Games",
    "http://dbpedia.org/resource/World_Health_Organization",
    # Events
    "http://dbpedia.org/resource/World_War_II",
    "http://dbpedia.org/resource/French_Revolution",
    "http://dbpedia.org/resource/Apollo_11",
    "http://dbpedia.org/resource/Black_Death",
    "http://dbpedia.org/resource/Renaissance",
]

HOPS = 6
NUM_PATHS = 5

GOOD_RELATIONS = {
    "http://dbpedia.org/ontology/birthPlace",
    "http://dbpedia.org/ontology/deathPlace",
    "http://dbpedia.org/ontology/nationality",
    "http://dbpedia.org/ontology/country",
    "http://dbpedia.org/ontology/spouse",
    "http://dbpedia.org/ontology/child",
    "http://dbpedia.org/ontology/parent",
    "http://dbpedia.org/ontology/almaMater",
    "http://dbpedia.org/ontology/award",
    "http://dbpedia.org/ontology/employer",
    "http://dbpedia.org/ontology/field",
    "http://dbpedia.org/ontology/knownFor",
    "http://dbpedia.org/ontology/institution",
    "http://dbpedia.org/ontology/doctoralAdvisor",
    "http://dbpedia.org/ontology/doctoralStudent",
    "http://dbpedia.org/ontology/successor",
    "http://dbpedia.org/ontology/predecessor",
    "http://dbpedia.org/ontology/foundedBy",
    "http://dbpedia.org/ontology/founder",
    "http://dbpedia.org/ontology/leader",
    "http://dbpedia.org/ontology/capital",
    "http://dbpedia.org/ontology/largestCity",
    "http://dbpedia.org/ontology/anthem",
    "http://dbpedia.org/ontology/language",
    "http://dbpedia.org/ontology/architect",
    "http://dbpedia.org/ontology/composer",
    "http://dbpedia.org/ontology/author",
    "http://dbpedia.org/ontology/director",
    "http://dbpedia.org/ontology/genre",
    "http://dbpedia.org/ontology/publisher",
    "http://dbpedia.org/ontology/region",
    "http://dbpedia.org/ontology/district",
    "http://dbpedia.org/ontology/subdivision",
    "http://dbpedia.org/ontology/location",
    "http://dbpedia.org/ontology/city",
    "http://dbpedia.org/ontology/party",
    "http://dbpedia.org/ontology/influenced",
    "http://dbpedia.org/ontology/influencedBy",
    "http://dbpedia.org/ontology/movement",
    "http://dbpedia.org/ontology/militaryBranch",
    "http://dbpedia.org/ontology/battle",
    "http://dbpedia.org/ontology/restingPlace",
    "http://dbpedia.org/ontology/college",
    "http://dbpedia.org/ontology/residence",
    "http://dbpedia.org/ontology/musicalBand",
    "http://dbpedia.org/ontology/bandMember",
    "http://dbpedia.org/ontology/associatedMusicalArtist",
}

MAX_TOKENS = 8192

# vLLM server settings (TPU)
VLLM_MODEL    = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
