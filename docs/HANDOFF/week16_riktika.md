# Week 16 Handoff — Riktika (M1) — June 2026

Note: Riktika (M1) is completing both Week 16 and Week 17 as part of a
combined session (Farhana completed Weeks 12-15).

---

## What I did this week (Week 16)

### evidence_aligner.py
- Added constrained yes/no/maybe instruction to system prompt
- Changed `[ANSWER:]` tag to `[ANSWER (yes, no, or maybe):]`
- Redesigned patient block construction to accept dicts with metadata and chunks
- Format: `Patient N (age X, gender Y, admission Z, ICD: codes, outcome: location): chunk1 ... chunk2`
- Fallback to plain string if dict not provided

### generator.py
- Changed `max_new_tokens` from 200 to 10
- Added `_extract_answer()`: three-step extraction (direct → fallback_regex → abstain)
- Updated output dict: `answer_raw`, `answer_extracted`, `extraction_method`
- `compute_confidence()` now receives `answer_raw`

### pipeline.py
- Added `pandas`, `glob`, `chunk_dynamic` imports
- Added `pat_meta_full` loading from `patient_metadata_stratified.csv`
- Added `_get_patient_chunks()`: reads `.txt` note, calls `chunk_dynamic(top_n=2)`, falls back to `bhc`
- Step 2 now builds patient summary dicts with real chunks and discharge_location
- Step 3 uses `align_evidence()` from evidence_aligner
- Step 4 uses `_extract_answer()` from generator
- Added pat_texts extraction before confidence scoring
- Updated return dict with all new fields

---

## Smoke test results — Week 16

5/5 queries returned yes/no/maybe via direct path:

Q1: yes | method=direct | conf=0.5168
Q2: no  | method=direct | conf=0.4636
Q3: no  | method=direct | conf=0.4318
Q4: yes | method=direct | conf=0.4897
Q5: yes | method=direct | conf=0.4331

---

## What is incomplete / known issues

- `patient_retriever.py` still loads `medicalai/ClinicalBERT` — Farhana's Week 14
  fix did not reach this file on GitHub. Needs fixing in Week 17.
- ICD codes show `nan` for many patients — not blocking for now
- `docs/HANDOFF/week15_farhana.md` on GitHub contains incorrect content
- `data_OLD` and `data_backup_old` exist locally — safe to delete after Week 17

---

## Continuing into Week 17 (same session)

Week 17 tasks per workplan:
- If smoke test <4/5: implement Option 2 post-hoc label extraction — NOT NEEDED (5/5 passed)
- Update output dict schema in generator.py and pipeline.py (GAP 8 fix) — DONE in Week 16
- Verify 5 smoke test outputs contain all new fields — DONE
- Fix `patient_retriever.py` ClinicalBERT issue
- Run baselines.py verification

---

## Exact command for next member (Week 18 — Farhana)

Before starting Week 18 work, rebuild both indexes locally:
python src/patient_retriever.py
python src/pmc_embedder.py

Then read `docs/HANDOFF/week16_riktika.md` and the Week 18 row in the
relay workplan before writing any code.
