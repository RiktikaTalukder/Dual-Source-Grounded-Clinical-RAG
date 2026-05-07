# Week 8 Handoff — Farhana (M2) — May 2026

## What I did this week
- Wrote `src/confidence_scorer.py` implementing the full confidence scoring formula
- Tested on 10 dummy (answer, literature, patient) triples — all scores sensible
- Key finding: Case 10 (appendicitis) correctly triggered disagreement penalty (A(L,P)=0.281 < 0.3), confidence dropped to 0.4236
- Case 4 (sepsis) achieved highest confidence 0.8389 — literature and patient evidence strongly aligned

## What works
- `compute_confidence(answer, literature_passages, patient_passages)` → returns s_al, s_ap, a_lp, confidence, penalty
- Disagreement penalty working correctly (multiplies by 0.7 when A(L,P) < 0.3)
- Models used: medicalai/ClinicalBERT (embeddings), facebook/bart-large-mnli (NLI)

## What is incomplete
- confidence_scorer.py is tested on dummy data only — not yet connected to real retriever outputs
- Riktika (Week 9) will integrate this into evidence_aligner.py and generator.py

## How to continue (Riktika, Week 9)
- Import `compute_confidence` from `src/confidence_scorer.py`
- Call it: `result = compute_confidence(answer, literature_passages, patient_passages)`
- Returns a dict with keys: s_al, s_ap, a_lp, confidence, penalty
- First run takes ~30 min (downloads bart-large-mnli ~1.6GB) — subsequent runs are fast