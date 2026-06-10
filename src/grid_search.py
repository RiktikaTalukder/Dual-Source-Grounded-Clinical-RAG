"""
grid_search.py
Week 20 — Farhana (M2)

Expands the grid search from 5 to 100 parameter combinations:
  5 weight combos × 5 penalty thresholds × 4 penalty multipliers = 100 total

Strategy: run the pipeline ONCE per question (200 total runs), cache the raw
component scores (s_al, s_ap, a_lp, answer, gold_label), then sweep all 100
combinations by doing arithmetic on the cached scores — no pipeline re-runs.

ECE is imported from evaluate.py — NOT reimplemented here.

Usage:
    python src/grid_search.py
"""

import json
import os
import sys
import numpy as np
from itertools import product
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import compute_ece

# ── Search space ─────────────────────────────────────────────────────────
WEIGHT_COMBOS = [
    (1/3,  1/3,  1/3),
    (0.5,  0.3,  0.2),
    (0.4,  0.4,  0.2),
    (0.3,  0.3,  0.4),
    (0.5,  0.25, 0.25),
]

WEIGHT_NAMES = [
    "equal_thirds",
    "literature_heavy",
    "balanced_lit_patient",
    "alignment_heavy",
    "strong_literature",
]

PENALTY_THRESHOLDS  = [0.20, 0.25, 0.30, 0.35, 0.40]
PENALTY_MULTIPLIERS = [0.5,  0.6,  0.7,  0.8]


# ── Load 200 validation questions ────────────────────────────────────────
def load_val_questions():
    base         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_ids_path = os.path.join(base, "data", "pubmedqa", "processed", "val_ids.json")
    raw_path     = os.path.join(base, "data", "pubmedqa", "raw", "ori_pqal.json")

    print(f"[grid_search] Loading val IDs from:      {val_ids_path}")
    print(f"[grid_search] Loading PubMedQA raw from: {raw_path}")

    with open(val_ids_path) as f:
        val_ids = set(json.load(f))
    with open(raw_path) as f:
        raw = json.load(f)

    questions = []
    for pmid, entry in raw.items():
        if pmid not in val_ids:
            continue
        gold = entry.get("final_decision", "maybe").lower().strip()
        questions.append({
            "pmid":       pmid,
            "question":   entry.get("QUESTION", ""),
            "gold_label": gold,
        })

    print(f"[grid_search] Loaded {len(questions)} validation questions.\n")
    return questions


# ── Phase 1: Run pipeline ONCE per question, cache raw scores ─────────────
def collect_raw_scores(questions):
    """
    Runs the pipeline once on each of the 200 validation questions.
    Returns a list of dicts with s_al, s_ap, a_lp, answer, gold_label.
    This is the only time the LLM and retrievers are called.
    """
    from pipeline import Pipeline

    print("[grid_search] Loading pipeline (models load once)...")
    pipe = Pipeline(weights=(1/3, 1/3, 1/3))   # weights don't matter here
    print("[grid_search] Pipeline ready.\n")

    cached = []
    n = len(questions)

    for i, q in enumerate(questions):
        print(f"  [{i+1:>3}/{n}] {q['question'][:75]}...")
        try:
            result  = pipe.run(q["question"])
            s_al    = float(result.get("s_al",  0.5))
            s_ap    = float(result.get("s_ap",  0.5))
            a_lp    = float(result.get("a_lp",  0.5))
            answer  = result.get("answer", "abstain").strip().lower()
            cached.append({
                "pmid":       q["pmid"],
                "question":   q["question"],
                "gold_label": q["gold_label"],
                "answer":     answer,
                "s_al":       s_al,
                "s_ap":       s_ap,
                "a_lp":       a_lp,
            })
        except Exception as e:
            print(f"  [WARNING] Error on question {i+1}: {e}")
            cached.append({
                "pmid":       q["pmid"],
                "question":   q["question"],
                "gold_label": q["gold_label"],
                "answer":     "abstain",
                "s_al":       0.5,
                "s_ap":       0.5,
                "a_lp":       0.5,
            })

    n_abstain = sum(1 for c in cached if c["answer"] == "abstain")
    abstain_rate = n_abstain / len(cached)
    print(f"\n[grid_search] Pipeline runs complete.")
    print(f"  Total questions : {len(cached)}")
    print(f"  Abstains        : {n_abstain} ({abstain_rate:.1%})")

    if abstain_rate > 0.10:
        print("\n  ⚠️  WARNING: abstain rate exceeds 10%.")
        print("  Do NOT proceed to main experiment until this is resolved.")
    else:
        print("  ✅ Abstain rate within acceptable range.")

    return cached


# ── Phase 2: Sweep 100 combos — pure arithmetic, no pipeline calls ────────
def sweep_combinations(cached):
    """
    For each of the 100 weight/threshold/multiplier combinations,
    recompute confidence scores from cached s_al, s_ap, a_lp values.
    No pipeline calls — just math.
    """
    total_combos  = len(WEIGHT_COMBOS) * len(PENALTY_THRESHOLDS) * len(PENALTY_MULTIPLIERS)
    print(f"\n[grid_search] Sweeping {total_combos} combinations (no pipeline calls)...")

    # Pre-extract gold and predicted labels — same for all combos
    golds = [c["gold_label"] for c in cached]
    preds = [c["answer"]     for c in cached]

    all_results   = {}
    best_name     = None
    best_ece      = float("inf")
    combo_counter = 0

    for (w_name, weights), threshold, multiplier in product(
            zip(WEIGHT_NAMES, WEIGHT_COMBOS),
            PENALTY_THRESHOLDS,
            PENALTY_MULTIPLIERS):

        combo_counter += 1
        combo_key = f"{w_name}__thr{threshold}__mul{multiplier}"
        alpha, beta, gamma = weights

        confidences = []

        for c in cached:
            s_al = c["s_al"]
            s_ap = c["s_ap"]
            a_lp = c["a_lp"]

            raw_conf = alpha * s_al + beta * s_ap + gamma * a_lp
            if a_lp < threshold:
                raw_conf = raw_conf * multiplier
            raw_conf = float(np.clip(raw_conf, 0.0, 1.0))
            confidences.append(raw_conf)

        # Call compute_ece with the correct signature: (golds, preds, confidences)
        ece_result = compute_ece(golds, preds, confidences)
        ece        = ece_result["ece"]

        corrects = [1 if g == p else 0 for g, p in zip(golds, preds)]
        accuracy = sum(corrects) / len(corrects) if corrects else 0.0

        all_results[combo_key] = {
            "weight_name":        w_name,
            "weights":            list(weights),
            "penalty_threshold":  threshold,
            "penalty_multiplier": multiplier,
            "ece":                round(ece,      6),
            "accuracy":           round(accuracy, 4),
            "avg_confidence":     round(float(np.mean(confidences)), 4),
            "n_questions":        len(cached),
        }

        if ece < best_ece:
            best_ece  = ece
            best_name = combo_key

    print(f"[grid_search] Sweep complete — {combo_counter} combinations evaluated.")
    return all_results, best_name, best_ece


# ── Phase 3: Pearson correlations ─────────────────────────────────────────
def compute_pearson(cached):
    s_al_list = [c["s_al"] for c in cached]
    s_ap_list = [c["s_ap"] for c in cached]
    a_lp_list = [c["a_lp"] for c in cached]

    r_al_ap, _ = stats.pearsonr(s_al_list, s_ap_list)
    r_al_lp, _ = stats.pearsonr(s_al_list, a_lp_list)
    r_ap_lp, _ = stats.pearsonr(s_ap_list, a_lp_list)

    print(f"\n[grid_search] Pearson correlations between formula components:")
    print(f"  s_al vs s_ap : r = {r_al_ap:.4f}")
    print(f"  s_al vs a_lp : r = {r_al_lp:.4f}")
    print(f"  s_ap vs a_lp : r = {r_ap_lp:.4f}")

    max_r = max(abs(r_al_ap), abs(r_al_lp), abs(r_ap_lp))
    if max_r < 0.5:
        interp = "All correlations below 0.5 — formula components confirmed empirically independent. Supports A(L,P) redesign."
        print(f"  ✅ {interp}")
    elif max_r > 0.7:
        interp = "High correlation detected (>0.7) — must acknowledge in thesis Discussion that formula components may not be fully independent."
        print(f"  ⚠️  {interp}")
    else:
        interp = "Moderate correlation (0.5–0.7) — note in thesis but not a critical issue."
        print(f"  ℹ️  {interp}")

    return {
        "s_al_vs_s_ap":   round(r_al_ap, 4),
        "s_al_vs_a_lp":   round(r_al_lp, 4),
        "s_ap_vs_a_lp":   round(r_ap_lp, 4),
        "interpretation": interp,
    }


# ── Phase 4: Update config.py ─────────────────────────────────────────────
def update_config(best, best_name, best_ece, base):
    import datetime
    import re

    config_path = os.path.join(base, "src", "config.py")
    with open(config_path, "r") as f:
        content = f.read()

    today = datetime.date.today().isoformat()
    w     = best["weights"]
    thr   = best["penalty_threshold"]
    mul   = best["penalty_multiplier"]

    new_block = (
        f"# Tuned by 100-combination grid search on 200-question val set, {today}.\n"
        f"# Best combo: weights={w}, threshold={thr}, multiplier={mul}, val ECE={best_ece:.6f}\n"
        f"# Combo key: {best_name}\n"
        f"CONFIDENCE_WEIGHTS = ({w[0]}, {w[1]}, {w[2]})   # alpha, beta, gamma\n"
        f"PENALTY_THRESHOLD  = {thr}\n"
        f"PENALTY_MULTIPLIER = {mul}\n"
    )

    pattern = (
        r"(?:# .*\n)*"
        r"CONFIDENCE_WEIGHTS\s*=.*\n"
        r"PENALTY_THRESHOLD\s*=.*\n"
        r"PENALTY_MULTIPLIER\s*=.*\n"
    )
    updated = re.sub(pattern, new_block, content)

    with open(config_path, "w") as f:
        f.write(updated)

    print(f"\n[grid_search] config.py updated:")
    print(f"  CONFIDENCE_WEIGHTS = {tuple(w)}")
    print(f"  PENALTY_THRESHOLD  = {thr}")
    print(f"  PENALTY_MULTIPLIER = {mul}")


# ── Main ──────────────────────────────────────────────────────────────────
def run_grid_search():
    questions = load_val_questions()

    # Phase 1 — run pipeline once per question
    cached = collect_raw_scores(questions)

    # Phase 2 — sweep 100 combos (pure math, fast)
    all_results, best_name, best_ece = sweep_combinations(cached)

    # Phase 3 — Pearson correlations
    correlations = compute_pearson(cached)

    # Print top 10
    sorted_combos = sorted(all_results.items(), key=lambda x: x[1]["ece"])
    print(f"\n{'='*72}")
    print("TOP 10 COMBINATIONS BY ECE")
    print(f"{'='*72}")
    print(f"{'Rank':<5} {'Combo Key':<47} {'ECE':>7} {'Accuracy':>9}")
    print("-" * 72)
    for rank, (key, res) in enumerate(sorted_combos[:10], 1):
        print(f"{rank:<5} {key:<47} {res['ece']:>7.4f} {res['accuracy']:>9.4f}")

    best = all_results[best_name]
    print(f"\n🏆 Best combo : {best_name}")
    print(f"   Weights    : {best['weights']}")
    print(f"   Threshold  : {best['penalty_threshold']}")
    print(f"   Multiplier : {best['penalty_multiplier']}")
    print(f"   ECE        : {best_ece:.6f}")
    print(f"   Accuracy   : {best['accuracy']}")

    # Save files
    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, "results", "grid_search")
    os.makedirs(out_dir, exist_ok=True)

    # grid_search_results_v2.json
    out_path  = os.path.join(out_dir, "grid_search_results_v2.json")
    save_data = {
        "metadata": {
            "week":        "Week 20",
            "member":      "Farhana (M2)",
            "n_combos":    len(all_results),
            "n_questions": len(cached),
            "best_combo":  best_name,
            "best_ece":    best_ece,
        },
        "all_combos": all_results,
        "top5":       {k: v for k, v in sorted_combos[:5]},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[grid_search] Results saved  → {out_path}")

    # component_correlations.json
    corr_path = os.path.join(out_dir, "component_correlations.json")
    with open(corr_path, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"[grid_search] Correlations   → {corr_path}")

    # Update config.py
    update_config(best, best_name, best_ece, base)

    print("\n✅ Week 20 grid search complete.")
    return all_results, best_name, cached


if __name__ == "__main__":
    run_grid_search()