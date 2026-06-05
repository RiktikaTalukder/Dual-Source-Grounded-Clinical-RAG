Pushed cleanly. ✅

Now you need to update `docs/HANDOFF/week15_farhana.md` — the file you wrote earlier only documented the SETUP.md work. It needs to also document the `confidence_scorer.py` redesign, the test results, and the correct "exact command to continue" for Week 16.

Open `docs/HANDOFF/week15_farhana.md` in VS Code. **Replace the entire contents** with this:

markdown
# Week 15 Handoff — Farhana (M2) — June 2026

Note: Week 15 was assigned to Riktika (M1) in the workplan but completed
by Farhana (M2) as part of an accelerated session to speed up progress.

---

## What I did this week

### Documentation (GAP 5 fix)
- Added Section 4 "Index Files" to docs/SETUP.md
  Documents all three FAISS indexes (mimic_patients.index,
  mimic_chunks.index, pmc_articles.index) with their purpose,
  which code uses each, and exact rebuild commands
- Fixed docs/HANDOFF.md index: corrected Weeks 12/13/14 entries
  (previously listed as three separate files, now correctly points
  to the single combined file week12_13_14_farhana.md)
- Added Week 15 entry to docs/HANDOFF.md index

Note: patient_retriever.py rebuild and 5 test query verification was
completed in Week 14. Week 15 documentation tasks carried those results
forward.

### confidence_scorer.py redesign (Week 15 core task)
- Removed ALL references to medicalai/ClinicalBERT
- Now uses pritamdeka/S-PubMedBert-MS-MARCO with pinned revision
  for all embedding operations
- Redesigned score_alignment() as A(L,P):
  - Takes only literature_passages and patient_chunks — NO answer argument
  - Premise: top-1 literature passage truncated to 500 chars
  - Hypothesis: top-1 patient chunk truncated to 350 chars
  - Returns NLI entailment probability from facebook/bart-large-mnli
  - Structurally independent of S(AL) and S(AP) by construction
- Added NEUTRAL_SCORE = 0.5 to config.py (GAP 3 fix)
- Redesigned compute_confidence() with lit_available and pat_available flags:
  - Flags checked FIRST before any encoder call
  - Both False: returns all 0.5 immediately, no model called
  - lit_available=False only: s_al=0.5, a_lp=0.5, only s_ap computed
  - pat_available=False only: s_ap=0.5, a_lp=0.5, only s_al computed

---

## What works

Run the 4-scenario test:

python src/confidence_scorer.py


Test results confirmed June 2026:

SCENARIO 1 — Both sources available:
  S(AL)=0.9495  S(AP)=0.9195  A(L,P)=0.0017
  Confidence=0.393  Penalty=True  ✅ PASS

SCENARIO 2 — Literature only (pat_available=False):
  S(AL)=0.9495  S(AP)=0.5  A(L,P)=0.5
  Confidence=0.6348  Penalty=False  ✅ PASS

SCENARIO 3 — Patient only (lit_available=False):
  S(AL)=0.5  S(AP)=0.9195  A(L,P)=0.5
  Confidence=0.6259  Penalty=False  ✅ PASS

SCENARIO 4 — No retrieval (both False):
  S(AL)=0.5  S(AP)=0.5  A(L,P)=0.5
  Confidence=0.5  Penalty=False  ✅ PASS

Note on A(L,P)=0.0017 in Scenario 1: this is expected behaviour on
dummy test text. The NLI model is working correctly — it scored low
entailment on these particular dummy passages. Not a bug.

Note on BertModel UNEXPECTED embeddings.position_ids warning:
harmless — known minor mismatch when loading S-PubMedBert with
SentenceTransformers. Can be ignored.

---

## What is incomplete / known issues

- MODEL_REVISIONS hashes in config.py are still set to "main" (not pinned
  commit hashes) — final pinning deferred to Week 18 sweep (GAP 7 fix)
- mimic_patients.index is local only (gitignored) — rebuild:
  python src/patient_retriever.py
- pmc_articles.index is local only (gitignored) — rebuild:
  python src/pmc_embedder.py
- patient_metadata_stratified.csv is local only (gitignored) — rebuilt
  automatically when running patient_retriever.py

---

## Exact command for next member to start Week 16

Week 16 is assigned to Riktika (M1) — update evidence_aligner.py,
generator.py, and pipeline.py with constrained yes/no/maybe prompt,
real note chunks in patient blocks, and updated output schema.

Before starting, rebuild both indexes locally:

python src/patient_retriever.py
python src/pmc_embedder.py


Then read docs/HANDOFF/week15_farhana.md and the Week 16 row in the
updated relay workplan before writing any code.

