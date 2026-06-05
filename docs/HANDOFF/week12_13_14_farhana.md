# Weeks 12–14 Handoff — Farhana (M2) — June 2026

Note: Weeks 12 and 14 were assigned to Riktika (M1) in the workplan
but were completed by Farhana (M2) as part of a combined session.
Week 13 is Farhana's assigned week.

---

## WEEK 12 — Phase A: MIMIC Data Rebuild

### What I did

- Downloaded MIMIC-IV v2.2 hosp module from PhysioNet (credentialed access):
  diagnoses_icd.csv (4,756,326 rows), admissions.csv (431,231 rows),
  patients.csv — stored locally at data/mimic/processed/ (gitignored)
- Wrote src/build_stratified_corpus.py: filters diagnoses to ICD-10 only,
  samples up to 100 admissions per clinical topic across 7 topic groups,
  extracts corresponding discharge notes from discharge.csv and saves as
  .txt files to data/mimic/mimic_sample/
- Wrote src/extract_bhc.py: rebuilds patient_metadata.csv from scratch
  using full hosp module tables, extracts Brief Hospital Course section
  from each .txt note (up to 800 chars), adds discharge_location column,
  stores ICD-10 codes only as zero-padded strings

### Results

- 637 discharge notes saved to data/mimic/mimic_sample/
- BHC extraction: 803 found, 135 not found (85.6% success rate — above
  80% threshold required by workplan)
- patient_metadata.csv rebuilt: 431,231 rows
  Columns: subject_id, hadm_id, age, gender, admission_type,
  discharge_location, icd_codes_top5, bhc

### PubMedQA Topic Clusters — 7 Clinical Groups

Identified from ori_pqal.json MESHES and QUESTION fields.
Used to build the stratified MIMIC corpus in build_stratified_corpus.py.

| Topic | ICD-10 Range | Admissions Found | Selected |
|---|---|---|---|
| Cardiovascular | I00–I99 | 103,249 | 100 |
| Diabetes | E10–E14 | 39,388 | 100 |
| Respiratory | J00–J99 | 47,503 | 100 |
| Neurological | G00–G99 | 56,746 | 100 |
| Infections | A00–B99 | 31,188 | 100 |
| Oncology | C00–D49 | 29,011 | 100 |
| Renal | N00–N39 | 48,391 | 100 |

Total unique admissions selected: 697
Matched discharge notes: 637 (60 admissions had no note in note module)

### What works

Rebuild stratified corpus (run from project root):

python src/build_stratified_corpus.py


Rebuild patient_metadata.csv with BHC extraction:

python src/extract_bhc.py


### What is incomplete / known issues

- data/mimic/mimic_sample/ .txt files are local only (gitignored)
- patient_metadata.csv is local only (gitignored)
- Old *_meta.json files in mimic_sample/ were deleted — no longer used
- diagnoses_icd_combined.csv (old 275-patient sample) replaced by full
  diagnoses_icd.csv from MIMIC-IV v2.2 hosp module
- MIMIC-IV hosp files must be downloaded from PhysioNet by each user
  individually (credentialed access required, version 2.2)

---

## WEEK 13 — Phase A: PMC Corpus Fix + Model Pinning

### What I did

- Ran build_stratified_corpus.py and extract_bhc.py on Windows to verify
  they work correctly (both passed)
- Wrote src/check_recall_overlap.py: discovered all 50 original recall
  evaluation IDs (seed=42) overlapped entirely with the 800-question test
  set. Fixed by sampling 50 IDs from val set only (seed=99, zero overlap
  confirmed). Saved clean sample to data/pubmedqa/processed/recall_ids.json
- Fixed PMC corpus: removed PMC387275.json (non-clinical gene expression
  paper). Verified all remaining 499 articles pass 2+ clinical keyword
  filter. Skipped running download_pmc.py because 499 articles already
  exceeds the 300+ target
- Added MODEL_REVISIONS dict to src/config.py (GAP 10 fix):
  pritamdeka/S-PubMedBert-MS-MARCO, facebook/bart-large-mnli,
  google/flan-t5-base — all set to "main" as placeholder; final hash
  pinning deferred to Week 18 verification sweep per workplan
- Updated src/pmc_embedder.py to reference MODEL_REVISIONS
- Rebuilt pmc_articles.index with 499 clean clinical articles (499 vectors,
  dimension 768)
- Verified retrieval with 3 clinical test queries — all results clinically
  relevant, no gene expression or non-clinical papers returned

### Results

- recall_ids.json: 50 clean IDs from val set, zero test overlap confirmed
- pmc_articles.index: 499 vectors rebuilt from clean corpus
- recall_test_overlap_check.txt: documents the contamination and fix

### What works

Check recall/test overlap:

python src/check_recall_overlap.py


Rebuild pmc_articles.index from scratch:

python src/pmc_embedder.py


Verify PMC retrieval:

python src/pmc_retriever.py


### What is incomplete / known issues

- MODEL_REVISIONS hashes are set to "main" not pinned commit hashes —
  final pinning is a Week 18 task (GAP 7 fix)
- pmc_articles.index is local only (gitignored) — must be rebuilt using
  command above
- Original recall evaluation results from Week 6 (Recall@5=0.87,
  Recall@10=0.89) are contaminated and cannot be reported in the thesis.
  evaluate_recall.py must be re-run using recall_ids.json going forward

---

## WEEK 14 — Phase B: Patient Retriever Rebuild

### What I did

- Rewrote src/patient_retriever.py completely:
  - Changed embedding model from medicalai/ClinicalBERT to
    pritamdeka/S-PubMedBert-MS-MARCO (unified model across all components)
  - Fixed jaccard(): normalizes all ICD codes to first 3 characters before
    computing set intersection. Example: E1165 → E11, I5032 → I50
  - Added assert checks: both ICD sets must contain only strings, never
    integers
  - Fixed build_patient_index(): embeds bhc field instead of raw note first
    1000 chars. Fallback to first 500 chars of .txt file if bhc is empty.
    No demographic string fallback needed (all 938 patients had bhc or note)
  - Fixed extract_icd_hints() in config.py: removed all ICD-9 codes, now
    ICD-10 only with 3-char category codes
  - load_resources() now loads patient_metadata_stratified.csv (938 rows)
    instead of full patient_metadata.csv (431,231 rows) to keep index and
    metadata aligned
- Created data/mimic/processed/patient_metadata_stratified.csv (938 rows,
  local only, gitignored) — filtered to patients with BHC content or
  matching note files
- Rebuilt mimic_patients.index with 938 vectors (dimension 768)
- Ran 5 test queries — all confirmed correct ICD topic matching:
  - Diabetes query → returned E11-range ICD codes ✅
  - Cardiac query → returned I21, I50-range codes ✅
  - COPD query → returned J441, J96-range codes with ICD Jaccard 0.40 ✅
  - Sepsis/pneumonia query → returned A419, J189 codes ✅
  - Renal query → returned N186 codes ✅

### Embedding sources breakdown

- BHC field used: 803 patients
- Note file fallback: 135 patients
- Demographic string fallback: 0 patients
- Total indexed: 938 vectors

### What works

Rebuild mimic_patients.index from scratch:

python src/patient_retriever.py


This will:
1. Load patient_metadata.csv and filter to 938 patients with notes/BHC
2. Embed using S-PubMedBert-MS-MARCO
3. Save mimic_patients.index and patient_metadata_stratified.csv
4. Run 5 test queries automatically

Query the index in other scripts:
python
from src.patient_retriever import load_resources, retrieve
model, meta, index = load_resources()
results = retrieve("your clinical query", {"E11", "I50"}, model, meta, index)


### What is incomplete / known issues

- mimic_patients.index is local only (gitignored) — must be rebuilt using
  command above
- patient_metadata_stratified.csv is local only (gitignored) — also
  rebuilt by running patient_retriever.py
- Some patients have discharge_location: nan — these are patients from
  the full hosp metadata where location was not recorded. Not a problem
  for retrieval
- Some patients have ICD codes: nan — minor data quality issue from hosp
  module, does not affect retrieval significantly
- SETUP.md "Index Files" section not yet updated — this is a Week 15
  task assigned to Riktika (GAP 5 fix)
- MODEL_REVISIONS revision pins not yet added to patient_retriever.py —
  deferred to Week 18 final sweep (GAP 7 fix)

---

## Exact command for Riktika to start Week 15

First, rebuild both local indexes (they are gitignored):

python src/patient_retriever.py


python src/pmc_embedder.py


Then proceed with Week 15 tasks:
- Update SETUP.md with Index Files section (GAP 5 fix)
- patient_retriever.py is already updated — Week 15 just needs SETUP.md
  and HANDOFF.md index entries for Weeks 7–11 verified

## Files committed this session (Weeks 12–14)

- src/build_stratified_corpus.py — new
- src/extract_bhc.py — new
- src/check_recall_overlap.py — new
- src/config.py — updated (MODEL_REVISIONS, ICD-10 only keyword map)
- src/pmc_embedder.py — updated (MODEL_REVISIONS reference)
- src/patient_retriever.py — fully rewritten
- data/pubmedqa/processed/recall_ids.json — new
- data/indexes/pmc_texts.json — updated (499 articles)
- docs/recall_test_overlap_check.txt — new
- docs/HANDOFF.md — updated (weeks 12–14 added to index)
- data/pmc_literature/pmc_sample/PMC387275.json — deleted
