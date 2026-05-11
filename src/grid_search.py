"""
grid_search.py
Week 11 — Riktika (M1)

Runs the dual-source RAG Pipeline on all 200 PubMedQA validation questions
with 5 different confidence weight combinations, then computes ECE for each.

Weight combos to try (alpha=S_AL, beta=S_AP, gamma=A_LP):
    W1: [1/3, 1/3, 1/3]      — equal weights (current default)
    W2: [0.5, 0.3, 0.2]      — literature-heavy
    W3: [0.4, 0.4, 0.2]      — balanced lit+patient, low alignment
    W4: [0.3, 0.3, 0.4]      — alignment-heavy
    W5: [0.5, 0.25, 0.25]    — strong literature, equal patient+alignment

Usage:
    python src/grid_search.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Weight combos to try ───────────────────────────────────────────────────
WEIGHT_COMBOS = [
    (1/3,  1/3,  1/3),
    (0.5,  0.3,  0.2),
    (0.4,  0.4,  0.2),
    (0.3,  0.3,  0.4),
    (0.5,  0.25, 0.25),
]

COMBO_NAMES = [
    "equal_thirds",
    "literature_heavy",
    "balanced_lit_patient",
    "alignment_heavy",
    "strong_literature",
]

# ── ECE computation ────────────────────────────────────────────────────────
def compute_ece(confidences, corrects, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    confidences : list of float — predicted confidence for each question
    corrects    : list of int   — 1 if answer was "yes", 0 if "no"
                                  (we use PubMedQA gold label for this)
    n_bins      : int           — number of bins to divide confidence into

    Returns
    -------
    ece : float — lower is better (0.0 = perfectly calibrated)
    bin_data : list of dicts — data for plotting reliability diagram
    """
    confidences = np.array(confidences)
    corrects    = np.array(corrects, dtype=float)
    n           = len(confidences)

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    ece        = 0.0
    bin_data   = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Find all samples whose confidence falls in this bin
        in_bin = np.where((confidences >= lo) & (confidences < hi))[0]

        if len(in_bin) == 0:
            bin_data.append({
                "bin_lower":   round(lo, 2),
                "bin_upper":   round(hi, 2),
                "count":       0,
                "avg_conf":    None,
                "avg_correct": None,
                "gap":         None
            })
            continue

        avg_conf    = float(np.mean(confidences[in_bin]))
        avg_correct = float(np.mean(corrects[in_bin]))
        gap         = abs(avg_conf - avg_correct)

        # ECE contribution: weighted by how many samples are in this bin
        ece += (len(in_bin) / n) * gap

        bin_data.append({
            "bin_lower":   round(lo, 2),
            "bin_upper":   round(hi, 2),
            "count":       len(in_bin),
            "avg_conf":    round(avg_conf, 4),
            "avg_correct": round(avg_correct, 4),
            "gap":         round(gap, 4)
        })

    return round(float(ece), 4), bin_data


# ── Load PubMedQA validation questions ────────────────────────────────────
def load_val_questions():
    """
    Load the 200 validation questions from PubMedQA.
    Returns list of dicts: {pmid, question, gold_label}
    gold_label: 1 = yes, 0 = no, 0.5 = maybe
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    val_ids_path = os.path.join(base, "data", "pubmedqa", "processed", "val_ids.json")
    raw_path     = os.path.join(base, "data", "pubmedqa", "raw", "ori_pqal.json")

    print(f"[grid_search] Loading val IDs from: {val_ids_path}")
    with open(val_ids_path) as f:
        val_ids = set(json.load(f))

    print(f"[grid_search] Loading PubMedQA raw data from: {raw_path}")
    with open(raw_path) as f:
        raw = json.load(f)

    questions = []
    label_map = {"yes": 1.0, "no": 0.0, "maybe": 0.5}

    for pmid, entry in raw.items():
        if pmid not in val_ids:
            continue
        question    = entry.get("QUESTION", "")
        gold_label  = entry.get("final_decision", "maybe").lower().strip()
        gold_numeric = label_map.get(gold_label, 0.5)
        questions.append({
            "pmid":        pmid,
            "question":    question,
            "gold_label":  gold_label,
            "gold_numeric": gold_numeric
        })

    print(f"[grid_search] Loaded {len(questions)} validation questions.")
    return questions


# ── Run grid search ────────────────────────────────────────────────────────
def run_grid_search():
    from pipeline import Pipeline

    questions = load_val_questions()
    n = len(questions)
    print(f"\n[grid_search] Will run {len(WEIGHT_COMBOS)} weight combos × {n} questions")
    print("[grid_search] This will take a while — please be patient!\n")

    # We will store results per combo
    all_results = {}

    for combo, name in zip(WEIGHT_COMBOS, COMBO_NAMES):
        print(f"\n{'='*60}")
        print(f"Running combo: {name}  weights={combo}")
        print(f"{'='*60}")

        # Create a fresh Pipeline with this weight combo
        pipe = Pipeline(weights=combo)

        confidences = []
        gold_labels = []
        outputs     = []

        for i, q in enumerate(questions):
            print(f"  [{i+1}/{n}] {q['question'][:70]}...")
            try:
                result = pipe.run(q["question"])
                confidences.append(result["confidence"])
                gold_labels.append(q["gold_numeric"])
                outputs.append({
                    "pmid":       q["pmid"],
                    "question":   q["question"],
                    "gold_label": q["gold_label"],
                    "answer":     result["answer"],
                    "confidence": result["confidence"],
                    "s_al":       result["s_al"],
                    "s_ap":       result["s_ap"],
                    "a_lp":       result["a_lp"],
                    "penalty":    result["penalty"]
                })
            except Exception as e:
                print(f"  [ERROR] Question {i+1} failed: {e}")
                confidences.append(0.3)
                gold_labels.append(q["gold_numeric"])
                outputs.append({
                    "pmid":       q["pmid"],
                    "question":   q["question"],
                    "gold_label": q["gold_label"],
                    "answer":     "ERROR",
                    "confidence": 0.3,
                    "error":      str(e)
                })

        # Compute ECE for this combo
        ece, bin_data = compute_ece(confidences, gold_labels)
        avg_conf = round(float(np.mean(confidences)), 4)
        penalties = sum(1 for o in outputs if o.get("penalty", False))

        print(f"\n  ✅ Combo: {name}")
        print(f"     ECE         = {ece}  (lower is better)")
        print(f"     Avg confidence = {avg_conf}")
        print(f"     Penalties    = {penalties}/{n}")

        all_results[name] = {
            "weights":      combo,
            "ece":          ece,
            "avg_conf":     avg_conf,
            "penalties":    penalties,
            "bin_data":     bin_data,
            "outputs":      outputs
        }

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'Combo':<25} {'Weights':<30} {'ECE':>6} {'AvgConf':>9} {'Penalties':>10}")
    print("-"*80)

    best_name = None
    best_ece  = float("inf")

    for name, res in all_results.items():
        w = res["weights"]
        w_str = f"({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})"
        print(f"{name:<25} {w_str:<30} {res['ece']:>6.4f} {res['avg_conf']:>9.4f} {res['penalties']:>10}")
        if res["ece"] < best_ece:
            best_ece  = res["ece"]
            best_name = name

    print(f"\n🏆 Best combo: {best_name}  (ECE = {best_ece})")
    print(f"   Best weights: {all_results[best_name]['weights']}")

    # ── Save results ─────────────────────────────────────────────────────
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, "results", "grid_search")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "grid_search_results.json")
    # Convert tuple weights to lists for JSON serialisation
    save_data = {}
    for name, res in all_results.items():
        save_data[name] = dict(res)
        save_data[name]["weights"] = list(res["weights"])

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[grid_search] Results saved to: {out_path}")

    # ── Save best weights separately for config.py ────────────────────────
    best_weights = list(all_results[best_name]["weights"])
    best_path = os.path.join(out_dir, "best_weights.json")
    with open(best_path, "w") as f:
        json.dump({
            "best_combo_name": best_name,
            "best_weights":    best_weights,
            "best_ece":        best_ece
        }, f, indent=2)

    print(f"[grid_search] Best weights saved to: {best_path}")
    return all_results, best_name


if __name__ == "__main__":
    run_grid_search()
