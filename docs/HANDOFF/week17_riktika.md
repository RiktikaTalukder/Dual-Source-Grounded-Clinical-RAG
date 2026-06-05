# Week 17 Handoff — Riktika (M1) — June 2026

## What I did this week

### Task 1: Verified patient_retriever.py model
- Confirmed src/patient_retriever.py already uses
  pritamdeka/S-PubMedBert-MS-MARCO (Farhana's Week 14 fix
  was already on GitHub — no code change needed)
- No ClinicalBERT references remain in any src/ file

### Task 2: Rebuilt MIMIC patient index
- Rebuilt data/indexes/mimic_patients.index locally using
  S-PubMedBert + BHC field embeddings
- 938 patients indexed (803 BHC field + 135 note files)
- Ran 5 test queries — all returned clinically relevant
  patients with matching ICD code ranges:
    - Diabetes query → E11 codes ✅
    - Cardiac query  → I21/I50 codes ✅
    - COPD query     → J44/J96 codes ✅
    - Sepsis query   → A41/J18 codes ✅
    - CKD query      → N18/N19 codes ✅
- Index is local only (gitignored — MIMIC data is private)
- Rebuild command: python src/patient_retriever.py

### Task 3: Verified output dict schema
- Ran Pipeline().run() on one query
- Confirmed all new Week 16 fields are present:
    - answer_raw        ✅
    - answer_extracted  ✅
    - extraction_method ✅
- Full key list: query, answer, answer_raw, answer_extracted,
  extraction_method, confidence, s_al, s_ap, a_lp, penalty,
  weights, literature_passages, patient_summaries

### Task 4: Ran baselines.py
- All 4 baselines execute without crashing ✅
- All return structured output with confidence scores ✅
- [baselines.py] All 4 baselines working correctly! ✅

---

## ⚠️ CRITICAL ISSUE FOR FARHANA — Week 18 priority

### PMC corpus contains non-clinical articles

The pmc_articles.index contains non-clinical papers that are
ranked as top results for clinical queries. Confirmed examples:

Query: "Does aspirin reduce cardiovascular risk?"
Top result: "Choosing to learn: The importance of student
             autonomy in higher education" (score: 0.9289)

Query: "Is metformin effective for diabetes?"
Top result: "An immunocompromised BALB/c mouse model for
             respiratory syncytial virus infection"

Query: "Does beta blocker reduce heart failure mortality?"
Top result: "Medical Students' and Residents' preferred site
             characteristics and preceptor behaviours"

### Impact
- baseline_literature_only returns wrong answers
- baseline_fixed_chunk_literature returns wrong answers
- patient_only and no_retrieval baselines are unaffected

### What Farhana needs to do in Week 18
1. Audit all 499 JSON files in data/pmc_literature/pmc_sample/
   to identify non-clinical articles
2. Remove non-clinical articles from that folder
3. Rebuild pmc_articles.index:
   python src/pmc_embedder.py
4. Re-run baselines.py and verify literature_only now returns
   clinically relevant answers
5. Verify with these 3 test queries:
   - "Does aspirin reduce cardiovascular risk?"
   - "Is metformin effective for diabetes?"
   - "Does beta blocker reduce heart failure mortality?"

### Reproduce the problem
python -c "
from src.pmc_retriever import retrieve_literature
results = retrieve_literature('Does aspirin reduce cardiovascular risk?', k=3)
for i, r in enumerate(results):
    print(f'Rank {i+1}:', r['passage'][:150])
"

---

## What is incomplete / known issues

- pmc_articles.index contains non-clinical papers — see above
- ICD codes show nan for some patients — pre-existing, not blocking
- MODEL_REVISIONS hashes still set to 'main' not pinned commit
  hashes — final pinning deferred to Week 18 sweep (GAP 7 fix)
- docs/HANDOFF/week15_farhana.md on GitHub contains incorrect
  content (Claude instructions) — pre-existing documentation issue

---

## What works

- Full pipeline end-to-end: Pipeline().run(query) → dict ✅
- All 3 new output schema fields present ✅
- Patient retriever: S-PubMedBert, 938 patients indexed ✅
- baselines.py: all 4 baselines run without crash ✅
- Smoke test from Week 16: 5/5 yes/no/maybe via direct path ✅

---

## Exact command for Farhana to start Week 18

1. Rebuild indexes (both are local only):
   python src/patient_retriever.py
   python src/pmc_embedder.py

2. Fix PMC corpus FIRST (see critical issue above)
   before running any baseline experiments

3. Read the Week 18 row in the updated relay workplan
