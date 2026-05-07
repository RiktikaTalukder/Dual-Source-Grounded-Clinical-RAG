# Week 8 Handoff — Farhana (M2) — 07 May 2026

## What I did this week
- Ran `03_patient_retrieval_test.ipynb` and added detailed observations
- Fixed `src/patient_retriever.py`: hardcoded METADATA_PATH and INDEX_PATH now resolve
  correctly on both Windows (my machine) and Linux (Riktika's machine) using `os.path.abspath`
- Wrote `src/confidence_scorer.py` implementing the full confidence scoring formula
- Tested on 10 dummy (answer, literature, patient) triples — all scores sensible
- Key finding: Case 10 (appendicitis) correctly triggered disagreement penalty
  (A(L,P)=0.281 < 0.3), confidence dropped to 0.4236
- Case 4 (sepsis) achieved highest confidence 0.8389 — literature and patient evidence
  strongly aligned

## What works
- `03_patient_retrieval_test.ipynb` runs top-to-bottom without errors on Windows
- `patient_retriever.py` path fix: works on both Windows and Linux automatically
- `compute_confidence(answer, literature_passages, patient_passages)` → returns
  s_al, s_ap, a_lp, confidence, penalty
- Disagreement penalty working correctly (multiplies by 0.7 when A(L,P) < 0.3)
- Models used: medicalai/ClinicalBERT (embeddings), facebook/bart-large-mnli (NLI)

## What is incomplete
- confidence_scorer.py is tested on dummy data only — not yet connected to real
  retriever outputs
- Riktika (Week 9) will integrate this into evidence_aligner.py and generator.py

## Key observations from 03_patient_retrieval_test.ipynb
- Hybrid scoring works as intended: ICD Jaccard successfully re-ranked results in
  Queries 1 and 2 despite lower cosine scores
- ICD Jaccard is 0.0 for most results due to two reasons:
  1. Only 275 patients in sample — too small for reliable overlap
  2. ICD version mismatch: query codes (ICD-9) vs some patient records (ICD-10)
- flan-t5-base summarizer works but repeats output on short inputs — not a bug,
  known behaviour of the model
- All cosine scores fall in 0.62–0.70 range — narrow band expected with small sample

## How to continue (Riktika, Week 9)
- Import `compute_confidence` from `src/confidence_scorer.py`
- Call it: `result = compute_confidence(answer, literature_passages, patient_passages)`
- Returns a dict with keys: s_al, s_ap, a_lp, confidence, penalty
- First run takes ~30 min (downloads bart-large-mnli ~1.6GB) — subsequent runs are fast
- Note: patient_retriever.py path fix is committed — no further changes needed there

## Files changed this week
- `src/confidence_scorer.py` — new file, fully tested
- `src/patient_retriever.py` — METADATA_PATH and INDEX_PATH fixed for cross-platform use
- `notebooks/03_patient_retrieval_test.ipynb` — observations added