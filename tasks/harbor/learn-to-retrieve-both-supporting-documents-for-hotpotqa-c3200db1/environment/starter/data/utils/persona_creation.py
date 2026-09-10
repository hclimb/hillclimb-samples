import random
import json
from faker import Faker

# --- CONSTANTS & EXACT QUANTITIES ---
NUM_PEOPLE = 100000
CHUNK_SIZE = 50000
POOL_SIZES = {
    'first_name': 400,
    'middle_name': 400,
    'last_name': 1000,
    'cities': 200,
    'universities': 300,
    'majors': 100,
    'companies': 263
}

fake = Faker()

def get_unique_list(generator_func, count):
    items = set()
    while len(items) < count:
        items.add(generator_func())
    return list(items)

FIRST_NAMES = get_unique_list(fake.first_name, POOL_SIZES['first_name'])
MIDDLE_NAMES = get_unique_list(fake.first_name, POOL_SIZES['middle_name'])
LAST_NAMES = get_unique_list(fake.last_name, POOL_SIZES['last_name'])
CITIES = get_unique_list(lambda: f"{fake.city()}, {fake.state_abbr()}", POOL_SIZES['cities'])
UNIVERSITIES = get_unique_list(lambda: f"{fake.company()} University", POOL_SIZES['universities'])
MAJORS = get_unique_list(fake.job, POOL_SIZES['majors'])
COMPANY_LIST = get_unique_list(fake.company, POOL_SIZES['companies'])
COMPANY_MAP = {comp: random.choice(CITIES) for comp in COMPANY_LIST}

# --- 50 DIVERSE TEMPLATES PER ATTRIBUTE ---
TEMPLATES = {
    'b_date': [
        "{sbj} was born on {val}.", "{sbj}'s birthday is {val}.", "{sbj} entered the world on {val}.",
        "{sbj} entered life on {val}.", "{sbj} celebrated {val} as their birth.",
        "{sbj}'s birth took place on {val}.", "{sbj} was born officially on {val}.",
        "{sbj}'s start was {val}.", "{sbj} has {val} as their date of birth.",
        "{sbj} was born on the date of {val}.", "{sbj} celebrates life since {val}.",
        "{sbj} has {val} marked as their birth.", "{sbj} was born on {val} in the past.",
        "{sbj} began on {val}.", "{sbj} came into being on {val}.", 
        "{sbj} has a birth date of {val}.", "{sbj} began their journey on {val}.", 
        "{sbj} was delivered on {val}.", "{sbj} celebrates their birth on {val}.",
        "{sbj} first saw the light of day on {val}.", "The birth of {sbj} occurred on {val}.",
        "{sbj} was brought into the world on {val}.", "{sbj} made their debut on {val}.",
        "{sbj}'s life started on {val}.", "{sbj} was welcomed on {val}.", 
        "{sbj} dates their birth to {val}.", "{sbj} was born into this world on {val}.",
        "{sbj}'s origin date is {val}.", "{sbj}'s entry into life was on {val}.",
        "{sbj}'s birth is recorded as {val}.", "{sbj} appeared on {val}.",
        "{sbj} marks their arrival on {val}.", "{sbj} has been around since {val}.",
        "{sbj}'s presence began on {val}.", "On {val}, {sbj} was born.", 
        "{sbj}'s existence dates back to {val}.", "{sbj} arrived in this world on {val}.",
        "{sbj} joined us on {val}.", "{sbj} was birthed on {val}.",
        "{sbj}'s first birthday was {val}.", "{sbj} was born {val}.",
        "{sbj} originated on {val}.", "{sbj} started life on {val}.",
        "{sbj} was born at {val}.", "{sbj}'s time of birth was {val}.",
        "{sbj} celebrates annually on {val}.", "{sbj} was born specifically on {val}.",
        "{sbj} first arrived on {val}.", "{sbj} was born during {val}.",
        "{sbj} came into being on {val}."
    ],
    'b_city': [
        "{sbj} spent early years in {val}.", "{sbj} originated from {val}.", "{sbj} hails from {val}.",
        "{sbj} was raised in {val}.", "{sbj} grew up in {val}.", "{sbj} is a native of {val}.",
        "{sbj} calls {val} their hometown.", "{sbj} started life in {val}.", 
        "{sbj} spent their childhood in {val}.", "{sbj} was born and bred in {val}.",
        "{sbj} claims {val} as their place of origin.", "{sbj} is originally from {val}.",
        "{sbj} was brought up in {val}.", "{sbj} lived their youth in {val}.",
        "{sbj} comes from {val}.", "{sbj} emerged from {val}.", "{sbj} began their story in {val}.",
        "{sbj} was birthed in {val}.", "{sbj} has roots in {val}.", "{sbj} belongs to {val}.",
        "{sbj} spent formative years in {val}.", "{sbj} resided in {val} during childhood.",
        "{sbj} was established in {val}.", "{sbj} is a product of {val}.",
        "{sbj} spent their infancy in {val}.", "{sbj} was situated in {val} early on.",
        "{sbj} lived in {val} as a child.", "{sbj} hails originally from {val}.",
        "{sbj} developed early on in {val}.", "{sbj} spent their beginning in {val}.",
        "{sbj} was based in {val} initially.", "{sbj} originated locally in {val}.",
        "{sbj} spent the start of life in {val}.", "{sbj} was a resident of {val} early.",
        "{sbj} stayed in {val} as a youngster.", "{sbj} occupied {val} in their youth.",
        "{sbj} was found in {val} at the start.", "{sbj} was a local in {val}.",
        "{sbj} grew through their youth in {val}.", "{sbj} was located in {val} early.",
        "{sbj} spent their first years in {val}.", "{sbj} was a {val} native.",
        "{sbj} spent their younger years in {val}.", "{sbj} was a citizen of {val} at birth.",
        "{sbj} spent the early chapters of life in {val}.", "{sbj} began in {val}.",
        "{sbj} resided early in {val}.", "{sbj} grew in {val}.", 
        "{sbj} was nurtured in {val}.", "{sbj} lived in {val} during their early days."
    ],
    'univ': [
        "{sbj} attended {val}.", "{sbj} graduated from {val}.", "{sbj} studied at {val}.",
        "{sbj} completed education at {val}.", "{sbj} earned their degree from {val}.",
        "{sbj} went to {val}.", "{sbj} was a student at {val}.", "{sbj} finished at {val}.",
        "{sbj} received schooling from {val}.", "{sbj} was educated at {val}.",
        "{sbj} matriculated at {val}.", "{sbj} went through {val}.", 
        "{sbj} spent their college years at {val}.", "{sbj} is an alum of {val}.",
        "{sbj} received their training at {val}.", "{sbj} pursued higher ed at {val}.",
        "{sbj} was part of the student body at {val}.", "{sbj} took classes at {val}.",
        "{sbj} developed academically at {val}.", "{sbj} spent time studying at {val}.",
        "{sbj} was a pupil at {val}.", "{sbj} earned an academic credential from {val}.",
        "{sbj} found their academic path at {val}.", "{sbj} pursued a degree at {val}.",
        "{sbj} was a learner at {val}.", "{sbj} spent university years at {val}.",
        "{sbj} attended classes at {val}.", "{sbj} graduated with honors from {val}.",
        "{sbj} completed their course at {val}.", "{sbj} was taught at {val}.",
        "{sbj} attended school at {val}.", "{sbj} was enrolled in {val}.",
        "{sbj} finished school at {val}.", "{sbj} received a diploma from {val}.",
        "{sbj} was at {val} for college.", "{sbj} joined {val} for their studies.",
        "{sbj} entered {val}.", "{sbj} studied hard at {val}.", 
        "{sbj} was at {val} during their education.", "{sbj} went to {val} for school.",
        "{sbj} graduated after attending {val}.", "{sbj} was schooled in {val}.",
        "{sbj} went for a degree at {val}.", "{sbj} completed a program at {val}.",
        "{sbj} studied their field at {val}.", "{sbj} was situated at {val} during college.",
        "{sbj} obtained a degree at {val}.", "{sbj} finished their studies at {val}.",
        "{sbj} spent time at {val}.", "{sbj} learned everything at {val}."
    ],
    'major': [
        "{sbj} focused on {val}.", "{sbj} specialized in {val}.", "{sbj} earned a degree in {val}.",
        "{sbj} majored in {val}.", "{sbj} concentrated on {val}.", "{sbj} studied {val}.",
        "{sbj} followed a path in {val}.", "{sbj} chose {val} as their major.",
        "{sbj} received training in {val}.", "{sbj} pursued a curriculum in {val}.",
        "{sbj} dedicated their studies to {val}.", "{sbj} focused their academic efforts on {val}.",
        "{sbj} was an expert in {val}.", "{sbj} gained knowledge in {val}.",
        "{sbj} followed the {val} program.", "{sbj} completed a major in {val}.",
        "{sbj} specialized their education in {val}.", "{sbj} spent their degree on {val}.",
        "{sbj} was focused primarily on {val}.", "{sbj} learned the ins and outs of {val}.",
        "{sbj} mastered {val}.", "{sbj} took a major in {val}.", 
        "{sbj} studied {val} deeply.", "{sbj} was a student of {val}.",
        "{sbj} earned their major in {val}.", "{sbj} chose to study {val}.",
        "{sbj} was immersed in {val}.", "{sbj} followed a {val} track.",
        "{sbj} was trained in {val}.", "{sbj} focused on the field of {val}.",
        "{sbj} had a concentration in {val}.", "{sbj} chose to major in {val}.",
        "{sbj} worked toward a degree in {val}.", "{sbj} studied the subject of {val}.",
        "{sbj} focused on learning {val}.", "{sbj} was a {val} specialist.",
        "{sbj} pursued {val} as a major.", "{sbj} obtained a degree focusing on {val}.",
        "{sbj} was a major of {val}.", "{sbj} chose {val}.",
        "{sbj} specialized in the study of {val}.", "{sbj} focused academic life on {val}.",
        "{sbj} had {val} as a major.", "{sbj} was a {val} major student.",
        "{sbj} chose the field of {val}.", "{sbj} specialized in {val} during school.",
        "{sbj} was focused on the discipline of {val}.", "{sbj} graduated in {val}.",
        "{sbj} had a degree for {val}.", "{sbj} focused on {val} at university."
    ],
    'c_name': [
        "{sbj} worked for {val}.", "{sbj} joined {val}.", "{sbj} was employed by {val}.",
        "{sbj} had a career at {val}.", "{sbj} worked at {val}.", "{sbj} was part of {val}.",
        "{sbj} took a job at {val}.", "{sbj} contributed to {val}.", "{sbj} was at {val}.",
        "{sbj} spent time at {val} professionally.", "{sbj} built their career at {val}.",
        "{sbj} was associated with {val}.", "{sbj} performed work for {val}.",
        "{sbj} was on the team at {val}.", "{sbj} served as an employee for {val}.",
        "{sbj} was hired by {val}.", "{sbj} spent their work life at {val}.",
        "{sbj} was found working at {val}.", "{sbj} joined the workforce at {val}.",
        "{sbj} was a member of the staff at {val}.", "{sbj} pursued their career at {val}.",
        "{sbj} was at {val} for work.", "{sbj} took a professional role at {val}.",
        "{sbj} was employed in {val}.", "{sbj} worked for the company {val}.",
        "{sbj} was a professional at {val}.", "{sbj} worked for the firm {val}.",
        "{sbj} was an employee of {val}.", "{sbj} spent years working at {val}.",
        "{sbj} was a part of {val}'s team.", "{sbj} held a position at {val}.",
        "{sbj} worked professionally at {val}.", "{sbj} worked with {val}.",
        "{sbj} was at {val} as a professional.", "{sbj} served {val}.",
        "{sbj} was engaged by {val}.", "{sbj} worked for {val}'s group.",
        "{sbj} worked as part of {val}.", "{sbj} was hired into {val}.",
        "{sbj} took employment with {val}.", "{sbj} was with {val}.",
        "{sbj} worked for {val} in industry.", "{sbj} found work at {val}.",
        "{sbj} was found at {val} for their career.", "{sbj} worked for {val} during life.",
        "{sbj} was a worker at {val}.", "{sbj} was at {val} for their job.",
        "{sbj} worked for the organization {val}.", "{sbj} was employed by {val} during adulthood.",
        "{sbj} had a role at {val}."
    ],
    'c_city': [
        "{sbj} was based in {val}.", "{sbj} worked in {val}.", "{sbj} gained experience in {val}.",
        "{sbj} performed their duties in {val}.", "{sbj} was located in {val}.",
        "{sbj} stayed in {val} for work.", "{sbj} was situated in {val} professionally.",
        "{sbj} carried out work in {val}.", "{sbj} was found in {val} for their job.",
        "{sbj} lived in {val} while working.", "{sbj} held their role in {val}.",
        "{sbj} was a professional resident of {val}.", "{sbj} worked in the city of {val}.",
        "{sbj} spent their work hours in {val}.", "{sbj} was employed in the {val} area.",
        "{sbj} worked at the office in {val}.", "{sbj} worked from {val}.",
        "{sbj} was located in {val} for their career.", "{sbj} was at the {val} location.",
        "{sbj} worked for years in {val}.", "{sbj} was situated in {val}.",
        "{sbj} worked on-site in {val}.", "{sbj} was at {val} during work.",
        "{sbj} resided in {val} for employment.", "{sbj} was in {val} for their job.",
        "{sbj} worked within {val}.", "{sbj} was found at work in {val}.",
        "{sbj} stayed in {val} during their employment.", "{sbj} was a worker in {val}.",
        "{sbj} performed roles in {val}.", "{sbj} was located in the {val} office.",
        "{sbj} worked in {val} for their employer.", "{sbj} was at a job in {val}.",
        "{sbj} worked primarily in {val}.", "{sbj} was in the city {val} for work.",
        "{sbj} worked for the company in {val}.", "{sbj} spent time in {val} at work.",
        "{sbj} was based out of {val}.", "{sbj} worked at their career in {val}.",
        "{sbj} stayed in {val} for professional reasons.", "{sbj} was at work within {val}.",
        "{sbj} was a {val} resident worker.", "{sbj} was found in {val} for work life.",
        "{sbj} spent their professional life in {val}.", "{sbj} was found at the {val} branch.",
        "{sbj} worked during the day in {val}.", "{sbj} was employed in {val}.",
        "{sbj} stayed at {val} during the work day.", "{sbj} worked at their role in {val}.",
        "{sbj} was based in the city {val}."
    ]
}

# --- GENERATION LOGIC ---

def generate_population(count):
    """Generates a unique population with attributes [cite: 2353-2357, 3116-3130]."""
    people = []
    seen_names = set()
    while len(people) < count:
        fn, mn, ln = random.choice(FIRST_NAMES), random.choice(MIDDLE_NAMES), random.choice(LAST_NAMES)
        full_name = f"{fn} {mn} {ln}"
        if full_name not in seen_names:
            seen_names.add(full_name)
            company = random.choice(COMPANY_LIST)
            people.append({
                "id": len(people),
                "name": full_name,
                "pronoun": random.choice(["He", "She"]),
                "b_date": f"{fake.month_name()} {random.randint(1, 28)}, {random.randint(1970, 2000)}",
                "b_city": random.choice(CITIES),
                "univ": random.choice(UNIVERSITIES),
                "major": random.choice(MAJORS),
                "c_name": company,
                "c_city": COMPANY_MAP[company]
            })
    return people

def create_bios_data(person, permute_n=1):
    """Generates bioS entries with permute rule: name in first sentence[cite: 3153, 3168]."""
    keys = ['b_date', 'b_city', 'univ', 'major', 'c_name', 'c_city']
    raw_facts = [(key, person[key]) for key in keys]
    random.shuffle(raw_facts) # Initial shuffle for bioS permute
    
    entry_sentences = []
    for i, (key, val) in enumerate(raw_facts):
        # Name in first sentence, pronoun in others [cite: 3154, 3168]
        subject = person['name'] if i == 0 else person['pronoun'].capitalize()
        template = random.choice(TEMPLATES[key])
        entry_sentences.append(template.format(sbj=subject, val=val))
    
    return " ".join(entry_sentences)

def create_qa_pairs(person):
    """Generates the 6 specific QA pairs for an individual ."""
    return [
        {"question": f"What is the birth date of {person['name']}?", "answer": person['b_date']},
        {"question": f"What is the birth city of {person['name']}?", "answer": person['b_city']},
        {"question": f"Which university did {person['name']} study?", "answer": person['univ']},
        {"question": f"What major did {person['name']} study?", "answer": person['major']},
        {"question": f"Which company did {person['name']} work for?", "answer": person['c_name']},
        {"question": f"Where did {person['name']} work?", "answer": person['c_city']}
    ]

# --- MAIN EXECUTION ---

print("Generating 100,000 unique identities...")
full_population = generate_population(NUM_PEOPLE)

for i in range(2):
    start = i * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk = full_population[start:end]
    
    bios_output = []
    qa_output = []
    
    print(f"Processing Part {i+1} (50,000 records)...")
    for person in chunk:
        bios_output.append({
            "id": person['id'],
            "name": person['name'],
            "biography": create_bios_data(person)
        })
        qa_output.append({
            "id": person['id'],
            "name": person['name'],
            "qa": create_qa_pairs(person)
        })
        
    # Save files
    with open(f'./bios-data/bios_part{i+1}.json', 'w') as f:
        json.dump(bios_output, f, indent=4)
    with open(f'./bios-qa/qa_part{i+1}.json', 'w') as f:
        json.dump(qa_output, f, indent=4)

print("Data generation complete. 4 files created: bios_part1/2.json and qa_part1/2.json.")