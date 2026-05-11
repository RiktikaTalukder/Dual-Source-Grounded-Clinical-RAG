"""
config.py
Week 11 — Riktika (M1)

Stores the best confidence weights found by grid search.
Update CONFIDENCE_WEIGHTS after running grid_search.py.

Weights = (alpha, beta, gamma)
  alpha = weight for S_AL (answer vs literature)
  beta  = weight for S_AP (answer vs patient)
  gamma = weight for A_LP (literature vs patient alignment)
"""

# ── Best weights from Week 11 grid search ─────────────────────────────────
# UPDATE THIS after running src/grid_search.py
# Replace with the best combo found (lowest ECE)
CONFIDENCE_WEIGHTS = (1/3, 1/3, 1/3)   # placeholder — update after grid search

# Disagreement penalty threshold (do not change)
PENALTY_THRESHOLD = 0.3
PENALTY_MULTIPLIER = 0.7

# Retrieval settings
TOP_K_LITERATURE = 3
TOP_K_PATIENTS   = 3
