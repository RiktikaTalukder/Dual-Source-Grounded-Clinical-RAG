"""
config.py
Week 11 — Riktika (M1)

Stores the best confidence weights found by grid search.
Best combo: alignment_heavy — found via 200-question validation grid search.

Weights = (alpha, beta, gamma)
  alpha = weight for S_AL (answer vs literature)
  beta  = weight for S_AP (answer vs patient)
  gamma = weight for A_LP (literature vs patient alignment)
"""

# ── Best weights from Week 11 grid search ─────────────────────────────────
# Winner: alignment_heavy  ECE = 0.0593  (lowest among 5 combos)
# Validated on 200 PubMedQA questions, seed=42 split by Farhana (Week 10)
CONFIDENCE_WEIGHTS = (0.3, 0.3, 0.4)   # alpha, beta, gamma

# Disagreement penalty threshold (do not change)
PENALTY_THRESHOLD  = 0.3
PENALTY_MULTIPLIER = 0.7

# Retrieval settings
TOP_K_LITERATURE = 3
TOP_K_PATIENTS   = 3

# Grid search record (for reference)
GRID_SEARCH_RESULTS = {
    "equal_thirds":         {"weights": (0.33, 0.33, 0.33), "ece": 0.0827},
    "literature_heavy":     {"weights": (0.50, 0.30, 0.20), "ece": 0.0929},
    "balanced_lit_patient": {"weights": (0.40, 0.40, 0.20), "ece": 0.0787},
    "alignment_heavy":      {"weights": (0.30, 0.30, 0.40), "ece": 0.0593},
    "strong_literature":    {"weights": (0.50, 0.25, 0.25), "ece": 0.0941},
}
