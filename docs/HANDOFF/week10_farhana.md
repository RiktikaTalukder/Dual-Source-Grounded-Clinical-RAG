# Week 10 Handoff — Farhana (M2) — May 2026

## What I did this week

- Ran `src/run_20_queries.py` to generate 20 dual-source RAG outputs →
  `results/generation_samples/dual_source_20.json`
- Checked confidence scores on all 20 outputs and logged observations (see below)
- Wrote `src/baselines.py` implementing all 4 baselines using same LLM (flan-t5-base)
  and same confidence scorer as `generator.py`
- Wrote `src/run_baselines_batch.py` to run all 4 baselines on the same 20 questions
- Ran all 4 baselines — all 80 runs (4 × 20) succeeded with no errors
- Wrote `src/split_pubmedqa.py` and ran it to produce the validation/test split

---

## What works

- `dual_source_rag()` — 20/20 queries succeeded, avg confidence 0.5829
- `baselines.py` — all 4 baseline functions working correctly:
  - `baseline_literature_only(query)`
  - `baseline_patient_only(query)`
  - `baseline_no_retrieval(query)`
  - `baseline_fixed_chunk_literature(query)`
- `run_all_baselines(query)` — runs all 4 at once, returns dict keyed by baseline name
- PubMedQA split: 200 validation / 800 test, seed=42, zero overlap confirmed

---

## Confidence score observations (dual_source_rag, 20 queries)

- **Only 1 penalty** (Query 17: sleep apnea) — A(L,P) = 0.2191, below 0.3 threshold,
  final confidence multiplied by 0.7 → 0.3462
- **Highest confidence**: Query 3 (aspirin + cardiovascular, 0.6875) — literature and
  patient evidence aligned well
- **Lowest S_AP scores**: Queries 8 (laparoscopic surgery) and 13 (vitamin D) — these
  clinical topics do not match MIMIC patient profiles well, pulling patient similarity down
- All 19 non-penalised queries scored between 0.48–0.69 — reasonable range for a small corpus
- Scores look calibrated: higher S_AL and S_AP generally correspond to queries where
  the topic is well-represented in both PMC and MIMIC sample data

---

## Baseline results summary (all 4 × 20 queries)

| Method                | Avg Confidence | Penalties (out of 20) |
|-----------------------|---------------|----------------------|
| dual_source_rag (W9)  | 0.5829        | 1                    |
| literature_only       | 0.6027        | 1                    |
| patient_only          | 0.2721        | 20                   |
| no_retrieval          | 0.2578        | 20                   |
| fixed_chunk_lit       | 0.6714        | 0                    |

**Key observations on baselines:**

- `patient_only` and `no_retrieval` received the disagreement penalty on **every single
  query** (20/20). Without literature evidence, the A(L,P) NLI alignment score
  consistently fell below 0.3, triggering the 0.7 penalty multiplier. This confirms
  that literature evidence is essential for cross-source agreement in our framework.
- `literature_only` performed similarly to `dual_source_rag` in average confidence
  (0.6027 vs 0.5829), but it uses a placeholder for patient evidence — its S_AP
  score reflects similarity to that placeholder, not real patient cases. This means
  its confidence is inflated relative to what it actually retrieves.
- `fixed_chunk_literature` showed the highest average confidence (0.6714, 0 penalties).
  This is because truncating passages to 512 characters tends to keep only the most
  topic-dense opening sentences, which score high on cosine similarity. This does not
  mean fixed chunking produces better answers — it means the confidence metric
  favours shorter, denser passages. This is worth noting as a limitation in Week 16.
- `dual_source_rag` is the only method using real evidence from both sources,
  making its confidence score the most meaningful of the five.

---

## Files changed this week

- `src/baselines.py` — new file (4 baseline functions + `run_all_baselines`)
- `src/run_baselines_batch.py` — new file (batch runner for 20 queries × 4 baselines)
- `src/split_pubmedqa.py` — new file (200 val / 800 test split, seed=42)
- `results/generation_samples/dual_source_20.json` — 20 dual-source outputs (local only)
- `results/generation_samples/baseline_literature_only_20.json` — local only
- `results/generation_samples/baseline_patient_only_20.json` — local only
- `results/generation_samples/baseline_no_retrieval_20.json` — local only
- `results/generation_samples/baseline_fixed_chunk_20.json` — local only
- `data/pubmedqa/val_ids.json` — 200 validation question IDs
- `data/pubmedqa/test_ids.json` — 800 test question IDs

---

## What is incomplete / known issues

- Baseline confidence scores for `patient_only` and `no_retrieval` use placeholder
  evidence strings for the source that is absent. This is intentional — these
  baselines are not designed to use both sources. Riktika should be aware when
  interpreting ECE in Week 11.
- All output JSON files contain full literature passages and patient summaries —
  these are local only (gitignored) because they are derived from MIMIC data.
- Fixed-chunk baseline simulates fixed chunking by character truncation (512 chars),
  not by the full token-based chunking in `chunking_baselines.py`. This is a
  reasonable approximation for a 20-question pilot but should be noted in §5.

---

## How to continue (Riktika, Week 11)

- All 5 method outputs for 20 questions are in `results/generation_samples/`
- Run grid search on the 200-question validation set using `data/pubmedqa/val_ids.json`
- Load PubMedQA questions by ID: `data = json.load(open('data/pubmedqa/raw/ori_pqal.json'))`
  then filter by `val_ids`
- Wrap pipeline into `Pipeline` class in `src/pipeline.py` with `Pipeline().run(query)`
- Try confidence weight combos: [1/3,1/3,1/3], [0.5,0.3,0.2], [0.4,0.4,0.2],
  [0.3,0.3,0.4], [0.5,0.25,0.25]
- Compute ECE and plot reliability diagrams for each combo
- Save best weights to `src/config.py` as `CONFIDENCE_WEIGHTS = (alpha, beta, gamma)`
- See workplan Week 11 for full task list
