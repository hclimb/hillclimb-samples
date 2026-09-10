# Multihop hard-neg training data vs. HotpotQA/MuSiQue evals: a construction-method mismatch

**Date:** 2026-08-16 · **Author:** claude (session with rohunagrawal) · **Status:** CONFIRMED
discrepancy against **both** benchmarks — likely explanation for the warm-start regression in
[2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md](2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md);
**not yet isolated as the causal factor** (see Follow-ups).

> **CORRECTION (2026-08-17):** the `0.4609→0.3828` regression cited below was measured with a
> judge that had a parsing bug inflating `llm_judge_accuracy` (fixed in
> [2026-08-17-llm-judge-last-tag-parsing-fix.md](../implementations/2026-08-17-llm-judge-last-tag-parsing-fix.md)).
> Corrected figures: **0.4297→0.3438** — the regression direction and rough magnitude are
> unchanged; every argument below still holds against the corrected numbers.

## Conclusion

**`ragrawal36/multihop_qa` (the source of every `multihop_qa_sft`/`multihop_hard_neg_full`
training recipe) is built by almost diametrically opposite methods from both HotpotQA and
MuSiQue.** Ours is a DBpedia-knowledge-graph random walk, LLM-phrased into a question, with
LLM-fabricated document text when no real paragraph states the fact, and **no verification that
the chain is actually necessary to answer the question**. HotpotQA is human-authored questions
over two real Wikipedia paragraphs, with bridge entities manually curated specifically to exclude
arbitrary schema-driven relation chains. MuSiQue goes further: it formally defines a **"connected
reasoning" condition** and verifies it *empirically with trained QA models* for every question it
ships, specifically to prevent the shortcut/disconnected-reasoning failure mode that HotpotQA
itself was later shown to be vulnerable to — a rigor our synthetic pipeline makes no attempt at.
Both papers frame KB-schema-driven question generation (exactly what our pipeline is) as the
anti-pattern their construction methods were designed to avoid or improve on. This is a
substantially better-evidenced explanation for the CE-only arm's hotpotqa-hybrid regression
(§[2026-08-16 write-up](2026-08-16-hotpotqa-hybrid-ce-only-checkpoint.md): `llm_judge_accuracy`
0.4609→0.3828, corrected to 0.4297→0.3438 — see the correction note above, after 42,800 steps
of further multihop warm-start training) than the CoT-target theory floated earlier — the
training data was never trying to look like either benchmark.

## Hypothesis & motivation

rohunagrawal, on hearing that further multihop warm-start finetuning regressed hotpotqa-hybrid
judge accuracy relative to the pre-finetuning source checkpoint, proposed an alternative
explanation to the CoT-intervention theory: a distribution mismatch between the hard-neg
training data and how the hotpotqa eval is set up. Asked to investigate by (1) reading
`datagen/multihop_qa/` to understand how the training data is actually generated, and (2)
reading the original HotpotQA paper to understand its construction method, then compare. Later
extended to a second, equally-important benchmark this framework evaluates on: rohunagrawal asked
for the same comparison against the **MuSiQue** paper, since MuSiQue eval results are something
they specifically care about.

## Setup — what was investigated

- **Training-data pipeline**, read in full: `datagen/multihop_qa/config.py`,
  `kg/dbpedia.py`, `kg/sampler.py`, `generation/generator.py`, `retrieval/wiki.py`,
  `dbpedia_gen.py`, `build_wiki_db.py`, `build_dbpedia_db.py`, `lm.py`, plus the downstream
  `datagen/generate_multihop_sft.py` (adds CoT) already read in the prior session (see
  [2026-08-13-doc-access-per-query-loss-investigation.md](2026-08-13-doc-access-per-query-loss-investigation.md)).
- **Live sample rows**, pulled via the HF `datasets-server` rows API (no TPU/local-accelerator
  needed — pure HTTP): 3 rows each from `ragrawal36/multihop_qa_sft` (training, 1,341,045 rows
  total), `ragrawal36/msa-hotpotqa-qa-with-ids` (hotpotqa eval), and
  `ragrawal36/msa-musique-qa-with-ids` (musique eval).
- **HotpotQA paper**: Yang et al., *"HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering"*, EMNLP 2018, [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) —
  read in full (§1–§4, the data collection and dataset-analysis sections).
- **MuSiQue paper**: Trivedi, Balasubramanian, Khot, Sabharwal, *"MuSiQue: Multihop Questions via
  Single-hop Question Composition"*, TACL 2022, [arXiv:2108.00573](https://arxiv.org/abs/2108.00573)
  — read §1–§7.1 (motivation, the MuSiQue condition, and the full 8-stage construction pipeline).

## Results

### How `ragrawal36/multihop_qa` is generated (`datagen/multihop_qa/`)

1. `build_dbpedia_db.py` ingests a **DBpedia 2022.09 snapshot** (`mappingbased-objects_en.ttl` +
   `long-abstracts_en.ttl`) into local SQLite — all object-property triples + English abstracts.
2. `kg/dbpedia.py::get_random_entity()` returns a **uniformly random entity from all of DBpedia**
   (a probabilistic 1-in-30 sample of every subject in the graph, filtered only to drop
   compound/non-article entities — no curation for "makes a good question subject").
3. `kg/sampler.py::sample_path()` random-walks from that entity for `hops` steps (CLI default 3;
   `config.py`'s unused `HOPS=6` looks like a stale default from an earlier iteration), at each
   step choosing a **uniformly random edge** restricted to `config.py`'s `GOOD_RELATIONS` — a
   fixed whitelist of ~55 DBpedia ontology predicates, mostly biographical/institutional
   (`birthPlace`, `spouse`, `doctoralAdvisor`, `employer`, `party`, `founder`, `genre`,
   `composer`, `capital`, …).
4. `retrieval/wiki.py::get_paragraph_with_link()` looks for a real Wikipedia paragraph in the
   hop's source article containing a wikilink to the target entity, and has an LLM
   (`lm.py`, default `meta-llama/Llama-3.3-70B-Instruct`) **judge** whether that paragraph
   actually states the claimed relation. **If none is found, it fabricates one**:
   `llm_write_grounding_sentence()` has the LLM write a synthetic one-line assertion of the fact
   from scratch — this is not flagged or filtered downstream, so some fraction of "positive
   documents" in the training corpus are LLM-generated prose, not organic Wikipedia text.
5. `generation/generator.py::generate_question()` has an LLM turn the relation chain into a
   question. The prompt explicitly instructs: don't leak intermediate entities, don't use
   relative clauses ("which/who/that"), answer with a specific fact from the *terminal* entity's
   passage rather than the entity name itself. **`_validate()` only checks entity-leak
   conditions — it never checks the no-relative-clause instruction.** The LLM routinely ignores
   it with no downstream filter catching it (see sample rows below).

**Live sample rows** (`ragrawal36/multihop_qa_sft`, via `datasets-server`):

> "Sergei Gorlukovich was born in a village in a country that is a unitary state, where the
> central government has the power to create or abolish administrative divisions, and such units
> exercise only the powers that the central government chooses to delegate; what is the number of
> UN member countries that have a unitary system of government?" → **"166 out of 193"**

> "Alison Mosshart is associated with a band that is signed to a label owned by a company that
> was acquired for how much in a transaction announced in March 2026?" → **"$7 billion"**

Both violate the generator's own "no relative clauses" instruction, and both terminate on a
context-free trivia fact (UN governance statistics; a corporate acquisition price) that has no
narrative connection to the walk's starting entity — a random-walk artifact, not a designed
"interesting" bridge. The second row also shows the underlying Wikipedia ingestion is **live/
current** (a "March 2026" fact), not a fixed historical snapshot.

### How HotpotQA is generated (Yang et al. 2018, §2)

1. Full Wikipedia dump → a hyperlink graph built **only from links in each article's first
   paragraph**.
2. The **bridge entity is restricted to a manually curated set of pages** (Appendix A) —
   explicit, named reasons: countries and other high-in-degree hubs "don't necessarily have much
   in common with all incoming links," and technical entities like "the IPv4 protocol" don't
   support a meaningful multi-hop question. Comparison questions similarly sample from **42
   manually curated lists of similar entities** (e.g. "Highest Mountains on Earth"), not
   arbitrary same-category entities.
3. **Human crowdworkers** (Mechanical Turk) are shown the real two-paragraph pair and write the
   question themselves, plus mark sentence-level supporting facts — no LLM in the loop for either
   question authorship or document text.
4. The paper states its motivation for this design explicitly, naming the class of dataset it is
   reacting against: *"existing datasets that target multi-hop reasoning, such as QAngaroo...
   and ComplexWebQuestions... are constructed using existing knowledge bases (KBs). As a result,
   these datasets are constrained by the schema of the KBs they use, and therefore the diversity
   of questions and answers is inherently limited. Instead, in this work, we focus on text-based
   question answering."*
5. Reasoning-type breakdown, sampled from dev/test (Table 3): Type I bridge 42%, Comparison 27%,
   Type II (multi-property) 15%, Type III 6%, Other (>2 supporting facts) 2%, single-hop 6%,
   unanswerable 2% — **the dominant structure is exactly two supporting paragraphs**, not chains.
6. Answer types (Table 2, sampled): Person 30%, Group/Org 13%, Location 10%, Date 9%, Number 8%,
   Artwork 8%, Yes/No 6%, Adjective 4%, Event 1% — dominated by entity-type answers, salient to a
   human reader by construction (a crowdworker chose to ask about them).
7. `test-fullwiki` — the setting our hybrid full-corpus eval protocol mirrors — requires the
   model to locate the two gold paragraphs among **the first paragraphs of every Wikipedia
   article**, with no gold paragraphs given, "to truly test the performance of the systems'
   ability at multi-hop reasoning in the wild."

### How MuSiQue is generated (Trivedi et al. 2022, §1–§5)

MuSiQue takes a fundamentally different approach from both HotpotQA and our pipeline: instead of
generating questions from scratch (KG walk + LLM, or human authorship over a curated pair), it
**programmatically composes existing, real single-hop QA pairs** and only afterwards has humans
phrase the final composed question:

1. **Starts from ~2M single-hop questions drawn from 5 real QA datasets** — SQuAD, Natural
   Questions, MLQA, T-REx, Zero Shot RE — not generated at all.
2. **Composability criteria**: two single-hop pairs `(q1,a1)`/`(q2,a2)` may be composed into a
   2-hop question only if `a1` is a named entity that is *also mentioned in* `q2` (verified by 3
   independent entity-resolution checks agreeing: spaCy NER type match, identical Wikipedia
   search top-result, and a SOTA wikification model) — composition is driven by real
   entity-string overlap between real questions, not a KG-schema relation walk.
3. **The MuSiQue condition — formally verified, not assumed.** A question is defined to require
   *connected reasoning* only if masking out a predecessor's answer from a later sub-question
   makes that sub-question **unanswerable to a trained QA model** (Eqn. 2: `M(qᵢ^mⱼ, C) ≠ aᵢ` for
   every edge, and `M(qᵢ, ∅) ≠ aᵢ` for every node). This is checked empirically with **two
   Longformer-Large models per fold** on held-out splits (S2/S3, "Disconnection Filtering") —
   any single-hop pair or full chain a trained model can shortcut through is filtered *out*. Our
   pipeline's `_validate()` only checks that intermediate entity *names* aren't leaked in the
   question text — it never checks whether the chain is actually necessary, i.e. it makes zero
   attempt at the property MuSiQue treats as its central contribution.
4. **Distractors are adversarially retrieved, not embedding-mined off a synthetic corpus**: each
   context has ~20 paragraphs — the supporting docs plus distractors retrieved via BM25 using the
   *concatenation of the sub-questions with intermediate answer mentions removed* as the query —
   specifically engineered to be topically confusable with the real supporting docs.
5. **Length is explicitly capped** (single-hop ≤10 tokens; 2/3-hop total ≤15/≤20 tokens) to
   prevent exactly the long, over-specified, chain-spelled-out-in-the-question style our sample
   rows show.
6. **Final question phrasing is still human**: crowdworkers see the DAG of single-hop questions
   and their bridge entities and write ONE coherent natural question using all of them — an LLM
   is never the question author.
7. **Explicit train/test leakage minimization** (S5: no single-hop question, answer, or paragraph
   overlaps across splits) and an **unanswerable-question variant** (MuSiQue-Full: remove one
   supporting paragraph per question to test insufficient-context detection) — both entirely
   absent from our pipeline.
8. Motivation stated directly: MuSiQue was built because **HotpotQA itself was later shown to be
   "largely solvable without multi-hop reasoning"** via shortcuts (over-specified sub-questions,
   train-test leakage, weak distractors) — MuSiQue reports a measured 3× larger human-machine gap
   and a substantially lower "disconnected reasoning" (DiRe) score than HotpotQA as evidence its
   construction method actually closes that gap.

**Live sample rows** (`ragrawal36/msa-musique-qa-with-ids`, via `datasets-server`):

> "What county is Erik Hort's birthplace a part of?" → **"Rockland County, New York"**

> "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
> → **"June 1982"**

Notably shorter and closer to natural phrasing than our training-data samples, consistent with
the paper's explicit token caps and human final-authorship step — though still visibly more
compressed/terse than HotpotQA's samples, consistent with MuSiQue's harder, 2-4-hop, less-cheatable
design intent.

### Side-by-side (three-way)

| | HotpotQA | MuSiQue | Our multihop training data |
|---|---|---|---|
| Question origin | Human-authored from scratch, given a real paragraph pair | **Programmatic composition of real single-hop QA pairs** (SQuAD/NQ/MLQA/T-REx/ZS-RE), then human-phrased | Fully synthetic — LLM generates the question from a KG relation chain |
| Bridge/entity selection | Manually curated "good bridge" pages; 42 curated comparison-entity lists | Entity-string overlap between real single-hop Q&A pairs, verified by 3-way entity resolution | Uniform random sample over *all* DBpedia entities |
| Relation type | Whatever a human notices reading real prose — unconstrained | Whatever relation the underlying single-hop datasets happen to test — unconstrained | Fixed whitelist of ~55 DBpedia ontology predicates, mostly biographical |
| **Chain necessity verified?** | Not formally verified (later shown exploitable) | **Yes — formally defined (MuSiQue condition) and empirically checked with trained QA models**, unnecessary chains filtered out | **Never checked** — only an entity-leak text check |
| Hop count | ~2 paragraphs for 92%+ of hard examples | 2–4 hops, 6 defined reasoning-graph shapes, explicit length caps | 3+ hops by construction, no length cap |
| Question authorship | Human (Mechanical Turk) | Human (Mechanical Turk), composing from a fixed decomposition | LLM — its own anti-relative-clause instruction is unenforced and routinely violated |
| Document grounding | Always real, organic Wikipedia paragraphs | Always real (drawn from the source single-hop datasets); distractors adversarially BM25-retrieved | Real paragraph text when a linked mention exists; **LLM-fabricated sentence** when it doesn't |
| Train/test leakage control | Not a stated focus | Explicit dedup procedure (no shared single-hop Q/A/paragraph across splits) | N/A (training corpus, not a controlled benchmark) |
| Explicit design stance | Written to *avoid* KB-schema-driven generation (names QAngaroo/ComplexWebQuestions as the anti-pattern) | Written to fix HotpotQA's own demonstrated shortcut-ability (measured 3× human-machine gap, lower DiRe score) | Is exactly a KB(DBpedia)-schema-driven pipeline, with no shortcut-checking at all |

## Interpretation

The mismatch is not merely "different topics" — it's a difference in generative process at
every stage, and it holds against *both* benchmarks this framework evaluates on: entity selection
(curated/composed-from-real-data vs. uniform-random), relation vocabulary (open vs. a
~55-predicate whitelist), hop structure (2–4 paragraphs, length-capped vs. 3+-hop uncapped
chains), authorship (human vs. LLM), and document grounding (always-real vs.
sometimes-fabricated). Both papers explicitly frame KB-schema-constrained question generation —
exactly what our pipeline is — as the specific flaw they were built to avoid or correct; MuSiQue
goes further and formally verifies, per question, that the multi-hop chain is actually necessary
(the property our data generation never checks at all). Extensive further training on the
DBpedia-walk data plausibly pulls the model's retrieval/discrimination and generation behavior
toward that narrower, chain-structured, biographical-relation-heavy distribution — and, absent
any connected-reasoning verification, may also be teaching shortcut-friendly retrieval behavior
rather than genuine multi-hop discrimination. Either mechanism would show up as *reduced*
transfer to both HotpotQA's and MuSiQue's real eval distributions — consistent with, and a
better-evidenced explanation for, the regression observed after the CE-only arm's 42,800
additional training steps.

This does not rule out the earlier CoT-target hypothesis or "more training in general" as
contributing factors — it adds a third, more structural candidate explanation that hadn't been
considered. All three are consistent with the same observed regression; only new experiments can
separate them.

## Follow-ups & risks — not yet done

- **Not yet a controlled test.** This investigation establishes the discrepancy is real and
  large; it does not yet measure how much of the observed regression it explains vs. the
  CoT-target intervention or plain continued training. A clean test: eval a checkpoint trained
  on **HotpotQA's own train split** (via `multihop_qa_sft_midtraining`-style recipes, or
  in-domain hard-neg mining over HotpotQA's corpus) at a matched step count, and see whether
  hotpotqa-hybrid judge accuracy holds or improves instead of regressing.
- **Scale of the "LLM-fabricated grounding sentence" fallback is unknown** — `get_paragraph_with_link`
  falls back to it whenever no real linked paragraph passes the LLM judge, but no rate/count was
  measured here. Worth instrumenting `datagen/multihop_qa/retrieval/wiki.py` to log how often the
  fallback fires, since a synthetic-document fraction is a bigger validity concern the higher it is.
- **The `_validate()` gap (no relative-clause check) means the shipped dataset's actual question
  style hasn't been quantified** — only illustrated via 3 sampled rows here. A systematic pass
  (regex or LLM classifier over a larger sample) would give a real rate, not an anecdote.
- **Not checked**: whether `mihir-1999/multihop_qa_sft-hard-neg-train`'s hard-negative mining
  (embedding-similarity search) draws candidates from the same DBpedia-walk document pool only,
  or pulls in any HotpotQA/MuSiQue-adjacent text — if entirely the former, the "hard negatives"
  the model learns to discriminate against are also out-of-domain relative to both evals' actual
  distractor pools.
- **No connected-reasoning check has ever been run on our data.** MuSiQue's own methodology (mask
  a predecessor's answer, see if a trained model still answers the sub-question) is directly
  reusable: run it against a sample of `ragrawal36/multihop_qa_sft` rows to get an actual
  shortcut-rate estimate, rather than inferring it from the absence of a check in `_validate()`.
  This would directly test the "teaching shortcuts" mechanism raised in Interpretation.

## Reproducibility

- **Code read**: `datagen/multihop_qa/{config.py,dbpedia_gen.py,build_wiki_db.py,build_dbpedia_db.py,lm.py}`,
  `datagen/multihop_qa/{kg/dbpedia.py,kg/sampler.py,generation/generator.py,retrieval/wiki.py}` —
  commit `b4d9541` on `multihop-finetuning`.
- **Sample rows** (no TPU/box needed — plain HTTP, reproducible from any machine):
  ```bash
  curl -s "https://datasets-server.huggingface.co/rows?dataset=ragrawal36%2Fmultihop_qa_sft&config=default&split=train&offset=0&length=3"
  curl -s "https://datasets-server.huggingface.co/rows?dataset=ragrawal36%2Fmsa-hotpotqa-qa-with-ids&config=default&split=train&offset=0&length=3"
  curl -s "https://datasets-server.huggingface.co/rows?dataset=ragrawal36%2Fmsa-musique-qa-with-ids&config=default&split=train&offset=0&length=3"
  ```
- **Papers**:
  - Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning. *HotpotQA: A Dataset for
    Diverse, Explainable Multi-hop Question Answering.* EMNLP 2018.
    [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) (§1–§4 read in full).
  - Trivedi, Balasubramanian, Khot, Sabharwal. *MuSiQue: Multihop Questions via Single-hop
    Question Composition.* TACL 2022. [arXiv:2108.00573](https://arxiv.org/abs/2108.00573)
    (§1–§7.1 read in full).
