# Week 11 Handoff — Riktika (M1) — May 2026

## What I did this week

- Downloaded PubMedQA raw data (`ori_pqal.json`) — 1000 questions total
- Wrote `src/pipeline.py` — wraps full dual-source RAG into a `Pipeline` class with support for custom confidence weights
- Wrote `src/grid_search.py` — runs all 5 weight combos on 200 validation questions and computes ECE for each
- Wrote `src/ece_plot.py` — plots reliability diagrams for all 5 combos and saves PNG files
- Ran full grid search: 5 combos × 200 questions = 1000 pipeline runs, zero errors
- Generated reliability diagrams (saved as PNG files in `results/grid_search/`)
- Updated `src/config.py` with best weights found

## What works

### Pipeline class (`src/pipeline.py`)
- `Pipeline(weights=None).run(query)` — runs full dual-source RAG on one query
- Accepts custom `weights=(alpha, beta, gamma)` tuple for grid search
- Uses same LLM (flan-t5-base), same retrievers, same confidence scorer as Farhana's generator.py
- Returns dict with answer, confidence, s_al, s_ap, a_lp, penalty, weights, passages

### Grid search (`src/grid_search.py`)
- Loads 200 validation questions from `data/pubmedqa/processed/val_ids.json`
- Matches questions to `data/pubmedqa/raw/ori_pqal.json`
- Runs Pipeline with each of 5 weight combos
- Computes ECE per combo using 10-bin reliability diagram method
- Saves full results to `results/grid_search/grid_search_results.json`
- Saves best weights to `results/grid_search/best_weights.json`

### ECE plot (`src/ece_plot.py`)
- Reads `grid_search_results.json`
- Plots all 5 reliability diagrams side by side → `reliability_diagrams.png`
- Plots detailed single diagram for best combo → `reliability_alignment_heavy.png`

## Grid search results (200 validation questions)

| Combo | Weights (α,β,γ) | ECE | Avg Conf | Penalties |
|---|---|---|---|---|
| equal_thirds | (0.33,0.33,0.33) | 0.0827 | 0.5705 | 15/200 |
| literature_heavy | (0.50,0.30,0.20) | 0.0929 | 0.5924 | 15/200 |
| balanced_lit_patient | (0.40,0.40,0.20) | 0.0787 | 0.5789 | 15/200 |
| **alignment_heavy** | **(0.30,0.30,0.40)** | **0.0593** | 0.5663 | 15/200 |
| strong_literature | (0.50,0.25,0.25) | 0.0941 | 0.5926 | 15/200 |

**Best combo: `alignment_heavy` — ECE = 0.0593**
**Best weights saved to `src/config.py`: `CONFIDENCE_WEIGHTS = (0.3, 0.3, 0.4)`**

## Key observations

- All 5 combos had exactly 15/200 penalties — penalty rate is stable across weight combos
- `alignment_heavy` (γ=0.4) gave the lowest ECE, meaning giving more weight to literature-patient agreement produces the most honest confidence scores
- This makes sense: when both sources agree, the answer is genuinely more trustworthy
- `strong_literature` and `literature_heavy` had the highest ECE — relying too heavily on S_AL alone hurts calibration
- ECE values are all below 0.10, which is a reasonable range for a small corpus system

## Files changed this week

- `src/pipeline.py` — new file (Pipeline class with custom weights)
- `src/grid_search.py` — new file (full grid search, ECE computation)
- `src/ece_plot.py` — new file (reliability diagram plots)
- `src/config.py` — updated with best weights (0.3, 0.3, 0.4)
- `data/pubmedqa/raw/ori_pqal.json` — downloaded (local only, gitignored)
- `results/grid_search/grid_search_results.json` — local only (gitignored)
- `results/grid_search/best_weights.json` — local only (gitignored)
- `results/grid_search/reliability_diagrams.png` — local only (gitignored)
- `results/grid_search/reliability_alignment_heavy.png` — local only (gitignored)

## What is incomplete / known issues

- `ori_pqal.json` and all `results/grid_search/` files are local only (not pushed to GitHub) because they are large or derived files — Farhana will need to re-download `ori_pqal.json` using the wget command in the README or run `wget -O data/pubmedqa/raw/ori_pqal.json https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json`
- ECE uses gold labels mapped as yes=1.0, no=0.0, maybe=0.5 — the maybe→0.5 mapping is an approximation and worth noting as a limitation in §5
- Reliability diagrams show some empty bins (no questions fell in that confidence range) — this is expected for a 200-question validation set

## How to continue (Farhana, Week 12)

- All pipeline outputs and best weights are ready
- Use `from config import CONFIDENCE_WEIGHTS` in any new evaluation code
- Use `Pipeline(weights=CONFIDENCE_WEIGHTS).run(query)` for all Week 12 evaluations
- Run evaluation on the 800-question **test set** using `data/pubmedqa/processed/test_ids.json`
- Compute final metrics: accuracy, F1, confidence calibration on test set
- See workplan Week 12 for full task list
