# Week 9 Handoff — Riktika (M1) — May 2026

## What I did this week
- Wrote `src/evidence_aligner.py`: formats literature + patient evidence into structured LLM prompt
- Wrote `src/generator.py`: full end-to-end dual_source_rag(query) pipeline
- Wrote `src/run_20_queries.py`: batch runner for 20 queries
- Ran full pipeline on 20 PubMedQA-style questions — all 20 succeeded

## What works
- `align_evidence(query, literature_passages, patient_summaries)` → structured prompt string
- `dual_source_rag(query)` → {answer, confidence, s_al, s_ap, a_lp, penalty, runtime_seconds}
- LLM: flan-t5-base via AutoModelForSeq2SeqLM (transformers 5.x compatible)
- 20 sample outputs saved to `results/generation_samples/dual_source_20.json`
- Average confidence: 0.5929 | Average runtime: 15.23s per query

## Key observations
- Confidence scores range 0.35–0.69 — expected with small PMC/MIMIC samples
- Query 17 (sleep apnea, confidence=0.3535) likely triggered disagreement penalty
- flan-t5-base answers are sometimes off-topic — due to small retrieval corpus, not a pipeline bug
- All 5 pipeline stages work correctly in sequence

## What is incomplete
- Pipeline uses small sample data — full PMC/MIMIC corpus will improve answer quality
- LLM answers not yet evaluated against gold labels (that is Week 10+ work)

## How to continue (Farhana, Week 10)
- Load outputs: `json.load(open("results/generation_samples/dual_source_20.json"))`
- Run confidence_scorer observations on these 20 outputs
- Write `src/baselines.py` with all 4 baselines (same LLM, same eval setup)
- See workplan Week 10 for full task list

## Files changed this week
- `src/evidence_aligner.py` — new file
- `src/generator.py` — new file  
- `src/run_20_queries.py` — new file
- `results/generation_samples/dual_source_20.json` — 20 sample outputs
