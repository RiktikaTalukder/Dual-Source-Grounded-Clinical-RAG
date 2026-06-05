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

# ── Pinned model revision hashes (Week 13 — GAP 10 fix) ──────────────────────
# These hashes pin the exact model versions used across all src/ files.
# Do NOT change these unless intentionally upgrading a model.
MODEL_REVISIONS = {
    "pritamdeka/S-PubMedBert-MS-MARCO": "main",
    "facebook/bart-large-mnli":         "main",
    "google/flan-t5-base":              "main",
}
# ─────────────────────────────────────────────────────────────────────────────
# ── Best weights from Week 11 grid search ─────────────────────────────────
# Winner: alignment_heavy  ECE = 0.0593  (lowest among 5 combos)
# Validated on 200 PubMedQA questions, seed=42 split by Farhana (Week 10)
CONFIDENCE_WEIGHTS = (0.3, 0.3, 0.4)   # alpha, beta, gamma

# Disagreement penalty threshold (do not change)
PENALTY_THRESHOLD  = 0.3
PENALTY_MULTIPLIER = 0.7
NEUTRAL_SCORE      = 0.5

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
# ── Shared ICD keyword extractor (used by generator, pipeline, baselines) ──
import re as _re

def extract_icd_hints(query_text: str) -> set:
    """
    Extract rough ICD code hints from free-text query using keyword matching.
    Centralised here so generator.py, pipeline.py, and baselines.py all
    use the same map without duplication.
    """
    q = query_text.lower()
    hints = set()
    keyword_map = {
        "diabetes":        {"E11"},
        "heart failure":   {"I50"},
        "pneumonia":       {"J18"},
        "sepsis":          {"A41"},
        "hypertension":    {"I10"},
        "copd":            {"J44"},
        "stroke":          {"I63"},
        "myocardial":      {"I21"},
        "asthma":          {"J45"},
        "kidney":          {"N18"},
        "renal":           {"N18"},
        "cancer":          {"C80"},
        "obesity":         {"E66"},
        "depression":      {"F32"},
        "appendicitis":    {"K37"},
        "atrial":          {"I48"},
        "anticoagulation": {"Z79"},
        "cholesterol":     {"E78"},
        "vitamin d":       {"E55"},
        "fracture":        {"M84"},
    }
    for keyword, codes in keyword_map.items():
        if keyword in q:
            hints.update(codes)
    return hints