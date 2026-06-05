# Week 15 Handoff — Farhana (M2) — June 2026

Note: Week 15 was assigned to Riktika (M1) in the workplan but completed
by Farhana (M2) as part of an accelerated session to speed up progress.

---

## What I did this week

- Added Section 4 "Index Files" to docs/SETUP.md (GAP 5 fix)
  Documents all three FAISS indexes (mimic_patients.index,
  mimic_chunks.index, pmc_articles.index) with their purpose,
  which code uses each, and exact rebuild commands
- Fixed docs/HANDOFF.md index: corrected Weeks 12/13/14 entries
  (previously listed as three separate files, now correctly points
  to the single combined file week12_13_14_farhana.md)
- Added Week 15 entry to docs/HANDOFF.md index

Note: The actual patient_retriever.py rebuild and 5 test query
verification was already completed in Week 14. Week 15 only required
the SETUP.md and HANDOFF.md documentation updates per the workplan.

---

## What works

Both indexes rebuild correctly using these commands (run from project root
with conda environment activated):

Rebuild MIMIC patient index (938 patients, S-PubMedBert, bhc embeddings):
python src/patient_retriever.py

Rebuild PMC literature index (499 clinical articles, S-PubMedBert):
python src/pmc_embedder.py

Test patient retrieval (5 queries, ICD topic verification):
Running `python src/patient_retriever.py` runs 5 test queries automatically.
Results from Week 14 confirmed correct ICD topic matching:
- Diabetes query → E11-range ICD codes ✅
- Cardiac query → I21, I50-range codes ✅
- COPD query → J441, J96-range codes, ICD Jaccard 0.40 ✅
- Sepsis/pneumonia query → A419, J189 codes ✅
- Renal query → N186 codes ✅

---

## What is incomplete / known issues

- MODEL_REVISIONS hashes in config.py are still set to "main" (not pinned
  commit hashes) — final pinning is deferred to Week 18 sweep (GAP 7 fix)
- mimic_patients.index is local only (gitignored) — rebuild using command above
- pmc_articles.index is local only (gitignored) — rebuild using command above
- patient_metadata_stratified.csv is local only (gitignored) — rebuilt
  automatically when running patient_retriever.py

---

## Exact command for next member to start Week 16

Week 16 is assigned to Farhana (M2) — redesign confidence_scorer.py.

Before starting Week 16 work, rebuild both indexes locally:
python src/patient_retriever.py
python src/pmc_embedder.py

Then proceed with Week 16 tasks:
- Redesign confidence_scorer.py (remove ClinicalBERT, use S-PubMedBert only)
- Redesign score_alignment() as A(L,P): NLI between sources, no answer argument
- Add NEUTRAL_SCORE = 0.5 to config.py with availability flags
- Test all 4 scenarios (dual, lit-only, pat-only, no-retrieval)