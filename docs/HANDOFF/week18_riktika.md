# Week 18 Handoff — Riktika (M1) — June 2026

## Note on relay swap
Week 18 was originally assigned to Farhana (M2). Riktika covered
it this week due to a schedule agreement between both members.
Farhana's next active week is Week 20 (grid search).

## What I did this week

### Task 1 — Fixed PMC corpus (critical issue from Week 17)
- Audited all 499 JSON files in data/pmc_literature/pmc_sample/
- Identified and removed 12 non-clinical articles:
  PMC466998, PMC515368, PMC521074, PMC523851, PMC524177,
  PMC524254, PMC528726, PMC539059, PMC543452, PMC546210,
  PMC546234, PMC549522
- Kept 6 borderline-but-clinical articles (blood pressure,
  depression in nursing homes, echocardiography, etc.)
- Rebuilt pmc_articles.index from 487 clean articles
- Verified 3 test queries now return clinical passages:
    - "Does aspirin reduce cardiovascular risk?" -> antiplatelet/HDL ✅
    - "Is metformin effective for diabetes?" -> diabetes/islet tx ✅
    - "Does beta blocker reduce heart failure mortality?" -> cardiac ✅

### Task 2 — Updated baselines.py
- Constrained yes/no/maybe prompt (matching evidence_aligner.py)
- max_new_tokens=10 (matching generator.py)
- _extract_answer() added (matching generator.py)
- answer_raw, answer_extracted, extraction_method in all return dicts
- Patient-only baseline uses real BHC chunks (bhc_chunks field),
  falls back to metadata summary only if bhc_chunks is empty
- lit_available=False / pat_available=False flags passed to
  compute_confidence() correctly for all 4 baselines
- No-retrieval baseline: both flags False -> 0.5 returned immediately,
  no encoder called (pre-embedding guard confirmed working)

### Task 3 — GAP 6 fix: Archived invalid grid search results
- Renamed results/grid_search/grid_search_results.json to
  grid_search_results_v1_INVALID.json (local only — gitignored)
- Added _INVALID_NOTICE key explaining why results are invalid
- NOTE FOR FARHANA: rename on your machine before Week 20:
  mv results/grid_search/grid_search_results.json \
     results/grid_search/grid_search_results_v1_INVALID.json

### Task 4 — GAP 7 fix: Final model revision pin sweep
- Fixed all from_pretrained() and SentenceTransformer() calls
  across ALL src/ files — every call now uses revision=
- Also replaced ClinicalBERT with S-PubMedBert in chunking_baselines.py
- Pinned REAL commit hashes in config.py (was "main"):
    pritamdeka/S-PubMedBert-MS-MARCO: 96786c7024f95c5aac7f2b9a18086c7b97b23036
    facebook/bart-large-mnli:         d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
    google/flan-t5-base:              7bcac572ce56db69c1ea7c8af255c5d7c9672fc2

### Task 5 — Smoke test: all 5 methods x 5 queries
All 25 outputs are yes/no/maybe. Zero abstains. No crashes.

Query                                  | lit  | pat  | none | chunk | dual
---------------------------------------|------|------|------|-------|------
Does aspirin reduce cardiovascular...  | no   | yes  | yes  | no    | no
Is metformin effective for diabetes?   | yes  | yes  | yes  | yes   | yes
Does beta blocker reduce heart fail... | yes  | yes  | yes  | yes   | yes
Is hypertension a risk factor for...   | yes  | yes  | yes  | yes   | yes
Does statin therapy reduce LDL...      | yes  | yes  | yes  | yes   | yes

Confidence scores:
- literature_only / patient_only / fixed_chunk: ~0.60-0.61 ✅
- no_retrieval: exactly 0.5 (pre-embedding guard confirmed) ✅
- dual_source: ~0.35 (A(L,P) agreement pulls score down — expected,
  reflects calibrated cross-source confidence, not a bug) ✅

---

## What is incomplete / known issues

- mimic_patients.index is local only (gitignored) — rebuild:
  python src/patient_retriever.py
- pmc_articles.index is local only (gitignored) — rebuild:
  python src/pmc_embedder.py
- grid_search_results_v1_INVALID.json is local only (gitignored).
  Farhana must rename the file on her machine before Week 20.
- Pipeline() instantiated once per query in smoke test script —
  causes repeated model loading. Test script issue only, not a bug.
- dual_source confidence consistently lower than baselines (~0.35
  vs ~0.60) — expected behaviour but worth discussing in thesis
  limitations if pattern holds on full 800-question experiment.

---

## What works

- PMC corpus: 487 clinical articles, clean retrieval ✅
- baselines.py: all 4 baselines produce yes/no/maybe ✅
- All src/ files: real commit hash revision pins in place ✅
- Smoke test: 5/5 queries x 5/5 methods = 25/25 valid outputs ✅
- GAP 6: old grid search results archived as v1_INVALID ✅
- GAP 7: final pin sweep complete with real hashes ✅

---

## Exact command for next member to start Week 19

Week 19 is assigned to Riktika (M1) — build src/evaluate.py.

Before starting, rebuild both indexes locally:
  python src/patient_retriever.py
  python src/pmc_embedder.py

Then read the Week 19 row in the updated relay workplan.
evaluate.py must implement: three-class accuracy, macro-F1,
BERTScore (deberta judge), Recall@5/10 (all-mpnet judge),
ECE (three-class, no 0.5 for maybe), McNemar+Cohen's h,
abstain rate metric, reliability diagram.

---

## Additional observation: dual_source confidence consistently lower than baselines

In the smoke test, dual_source confidence was ~0.35 across all 5 queries
while all baselines scored ~0.60. This pattern is consistent — not a
one-off result.

This is expected behaviour by design: dual_source confidence is harder
to earn because A(L,P) (cross-source agreement) has the largest weight
(gamma=0.4) and tends to be low when the retrieved literature and patient
cases don't discuss the same clinical angle.

However — if this pattern holds consistently across the full 800-question
experiment, it raises a calibration question: does lower confidence in
dual_source actually predict lower accuracy? Or is dual_source equally
accurate but just systematically under-confident?

This should be examined explicitly in Week 21 when full results are
available. Specifically:
- Compare dual_source accuracy vs baseline accuracy
- Compare ECE across all 5 methods (evaluate.py, Week 19)
- If dual_source has lower ECE than baselines despite lower raw
  confidence, the calibration argument holds and is a thesis strength
- If dual_source has higher ECE, this needs to be addressed as a
  limitation in thesis Section 5

Riktika (Week 19): keep this in mind when verifying evaluate.py
output on the 20-question pilot.
