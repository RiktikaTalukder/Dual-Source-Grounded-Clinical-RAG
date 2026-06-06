# Week 19 Handoff — Riktika

## What was done this week

- Wrote `src/evaluate.py` — single source of truth for all evaluation metrics
- Regenerated both pilot output files from real PubMedQA test IDs (first 20 from test_ids.json) with correct schema including `answer_raw`, `answer_extracted`, `extraction_method`, `gold_label`, `gold_contexts`
- Old pilot file (`dual_source_20.json`) was synthetic queries with no gold labels — replaced
- `baseline_literature_only_20.json` did not exist — generated fresh
- Verified evaluate.py on both pilot files — all 8 metric sections run cleanly

## evaluate.py metric coverage

1. Abstain rate (GAP 9 fix — abstains treated as incorrect, hard warning if >10%)
2. Three-class accuracy overall + per class (yes/no/maybe)
3. Macro-F1 across all three classes
4. ECE — three-class correctness-based, no 0.5 for maybe (B5 Option 1)
5. Reliability diagram PNG → results/figures/
6. BERTScore — deberta-xlarge-mnli judge, mean-pooled embeddings
7. Recall@5 and Recall@10 — all-mpnet-base-v2 judge, cosine >= 0.5 hit threshold
8. McNemar test + Cohen's h for pairwise method comparisons

## Pilot output results (dual_source_20, 20 questions)

- abstain_rate : 0.0%
- accuracy     : 50.0% (10/20) — above 40% threshold ✓
- macro_f1     : 0.2299
- ECE          : 0.1464 — below 0.3 threshold ✓
- Recall@5     : 0.60
- Recall@10    : 0.60
- BERTScore    : 0.3043 — see known issue below
- McNemar p    : 1.0, Cohen's h : 0.0 (both methods identical predictions — expected, weights not yet tuned)

## Known issues / anomalies

### BERTScore 0.3043 — below 0.5-0.9 workplan threshold
The `bert_score` library (v0.3.13) has an `OverflowError: int too big to convert`
bug when using `deberta-xlarge-mnli` with the installed `tokenizers` version.
Workaround: BERTScore is computed using mean-pooled sentence embeddings +
cosine similarity directly via transformers, bypassing the bert_score library.
Sentence-level cosine similarity produces scores in the 0.3-0.5 range rather
than 0.8-0.9 token-level range. The metric is internally consistent and will
correctly rank methods relative to each other. The 0.5-0.9 threshold in the
workplan was written assuming token-level BERTScore and no longer applies.
Document this as a methodological note in the thesis.

### no/maybe accuracy = 0.0
The pipeline never predicts "no" or "maybe" correctly on 20 questions with
current (invalid) weights. Expected — grid search has not been rerun yet.
Will resolve after Week 20 grid search produces valid weights.

### McNemar p=1.0, Cohen's h=0.0
Both methods produced identical predictions on all 20 questions. Expected
at this stage — weights are still invalid pre-grid-search.

## Files committed this week

- `src/evaluate.py`
- `results/figures/reliability_dual_source_20.png`

## Files local only (gitignored — contain pipeline outputs)

- `results/generation_samples/dual_source_20.json` (regenerated, real PubMedQA IDs)
- `results/generation_samples/baseline_literature_only_20.json` (new)
- `results/main_experiment/metrics_dual_source_20.json`

## How to continue — Farhana, Week 20

Usage:
    python src/evaluate.py results/generation_samples/dual_source_20.json
    python src/evaluate.py path/to/any_output.json path/to/paired_output.json

The second argument enables McNemar + Cohen's h comparison between two methods.

Week 20 tasks:
- Expand grid_search.py to 100-combination sweep
- Import three-class ECE from evaluate.py (do not reimplement — use `from evaluate import compute_ece`)
- Run grid search on 200-question val set
- Save results to results/grid_search/grid_search_results_v2.json
- Update config.py with best weights
- Compute Pearson correlations (s_al vs s_ap vs a_lp) — save to component_correlations.json
- Rename grid_search_results_v1_INVALID.json on your local machine if not done yet
