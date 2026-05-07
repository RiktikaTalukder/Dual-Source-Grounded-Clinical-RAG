# Week 7 Handoff — Riktika (M1) — 06 May 2026

## What I did this week
- Rebuilt PMC FAISS index using `python src/pmc_embedder.py` (500 articles, model: pritamdeka/S-PubMedBert-MS-MARCO)
- Verified PMC retriever works correctly (`python src/pmc_retriever.py`)
- Ran `mimic_preprocess.py`: saved 200 cleaned MIMIC discharge notes to `data/mimic_sample/`, built `patient_metadata.csv` (275 rows)
- Fixed wrong path in `build_mimic_index.py` (was `data/mimic/processed/discharge.csv`, corrected to `data/discharge.csv.gz`)
- Built MIMIC FAISS index: saved to `data/indexes/mimic_chunks.index` and `data/indexes/mimic_chunks_text.csv`

## What works
- PMC retrieval: fully functional, Recall@5=0.87 baseline intact
- MIMIC preprocessing: 200 sample notes cleaned and saved
- MIMIC FAISS index: built from 500 sampled discharge notes

## What is incomplete
- No MIMIC retriever test script yet — Farhana should write/run a mimic_retriever.py to verify MIMIC index works
- The two indexes (PMC + MIMIC) are not yet merged into a dual-source retriever

## How to continue (Farhana, Week 8)
- IMPORTANT: `mimic_chunks.index` is a binary file — NOT committed to Git (too large). Run `python src/build_mimic_index.py` to rebuild.
- Write `src/mimic_retriever.py` modeled after `pmc_retriever.py` but loading `data/indexes/mimic_chunks.index`
- Begin building `src/dual_retriever.py` that queries BOTH indexes and merges results
- Embedding model for MIMIC index: `medicalai/ClinicalBERT`

## Files changed this week
- `src/build_mimic_index.py` — fixed discharge_path
- `data/mimic_sample/` — 200 cleaned notes added
- `data/patient_metadata.csv` — rebuilt
- `data/indexes/mimic_chunks.index` — new (not committed, binary)
- `data/indexes/mimic_chunks_text.csv` — new (NOT committed, in .gitignore — rebuild with build_mimic_index.py)

## Update — Week 7 completed fully (08 May 2026)
- patient_retriever.py: DONE — hybrid similarity (0.6×cosine + 0.4×ICD Jaccard)
- mimic_patients.index: DONE — 275 patient vectors, saved to data/indexes/
- 03_patient_retrieval_test.ipynb: DONE — 5 sample queries with ranked results
- dynamic_chunker.py: DONE — flan-t5-base summarizer, compresses chunks to ~150 tokens

## How to call patient_retriever.py
```python
from src.patient_retriever import load_resources, retrieve
model, meta, index = load_resources()
results = retrieve("your query text here", {"icd1","icd2"}, model, meta, index)
```

## Note on ICD Jaccard scores
Currently low (often 0.0) because sample has only 275 patients.
Will improve significantly when full MIMIC data is used in later phases.
