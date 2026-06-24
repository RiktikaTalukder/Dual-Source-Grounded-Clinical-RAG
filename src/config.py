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

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0 if torch.cuda.is_available() else -1

# ── Pinned model revision hashes (Week 13 — GAP 10 fix) ──────────────────────
# These hashes pin the exact model versions used across all src/ files.
# Do NOT change these unless intentionally upgrading a model.
MODEL_REVISIONS = {
    "pritamdeka/S-PubMedBert-MS-MARCO": "96786c7024f95c5aac7f2b9a18086c7b97b23036",
    "facebook/bart-large-mnli":         "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
    "google/flan-t5-base":              "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "google/flan-t5-large":             "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    "roberta-large":                    "722cf37b1afa9454edce342e7895e588b6ff1d59",
    "sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
}

# Active generator model — change this to switch between flan-t5-base and flan-t5-large
GENERATOR_MODEL = "google/flan-t5-large"
# ─────────────────────────────────────────────────────────────────────────────
# ── Best weights from Week 20 grid search (flan-t5-large, 100 combos) ─────
# Winner: strong_literature  ECE = 0.019, accuracy = 0.53
# Validated on 200 PubMedQA questions. Source: results/grid_search/grid_search_results_v2.json
CONFIDENCE_WEIGHTS = (0.5, 0.25, 0.25)   # alpha, beta, gamma

PENALTY_THRESHOLD  = 0.35
PENALTY_MULTIPLIER = 0.8
NEUTRAL_SCORE      = 0.5

# Retrieval settings
TOP_K_LITERATURE = 5
TOP_K_PATIENTS   = 5

# Grid search record (for reference)
GRID_SEARCH_RESULTS = {
    "equal_thirds":         {"weights": (0.33, 0.33, 0.33), "ece": 0.0827},
    "literature_heavy":     {"weights": (0.50, 0.30, 0.20), "ece": 0.0929},
    "balanced_lit_patient": {"weights": (0.40, 0.40, 0.20), "ece": 0.0787},
    "alignment_heavy":      {"weights": (0.30, 0.30, 0.40), "ece": 0.0593},
    "strong_literature":    {"weights": (0.50, 0.25, 0.25), "ece": 0.019},  # Week 20 100-combo winner, corrected from stale Week 11 value
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
