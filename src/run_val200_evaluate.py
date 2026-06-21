"""
run_val200_evaluate.py
Reads the 5 saved generation JSONs from run_val200_generate.py and evaluates
each one. Run this in a FRESH terminal/process, separately from generation —
do not import this in the same process that ran the baselines or pipeline.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate as run_evaluation

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE, "results", "generation_samples")
EVAL_DIR = os.path.join(BASE, "results", "val200_metrics")
os.makedirs(EVAL_DIR, exist_ok=True)

METHODS = [
    ("dual_source",            "dual_source_val200.json"),
    ("baseline_literature_only","baseline_literature_only_val200.json"),
    ("baseline_patient_only",   "baseline_patient_only_val200.json"),
    ("baseline_no_retrieval",   "baseline_no_retrieval_val200.json"),
    ("baseline_fixed_chunk",    "baseline_fixed_chunk_val200.json"),
]

def main():
    summary = {}
    for name, filename in METHODS:
        out_path = os.path.join(OUT_DIR, filename)

        if not os.path.exists(out_path):
            print(f"\n--- {name}: SKIPPED — file not found at {out_path} ---")
            summary[name] = {"error": "generation file not found — did generation finish?"}
            continue

        with open(out_path) as f:
            n_entries = len(json.load(f))
        if n_entries != 200:
            print(f"\n--- {name}: WARNING — file has {n_entries}/200 entries, "
                  f"not evaluating an incomplete run ---")
            summary[name] = {"error": f"incomplete: only {n_entries}/200 questions present"}
            continue

        print(f"\n--- Evaluating: {name} ---")
        try:
            metrics = run_evaluation(out_path)
            acc     = metrics.get("accuracy", {})
            f1      = metrics.get("f1", {})
            ece     = metrics.get("ece", {})
            recall  = metrics.get("recall", {})
            bs      = metrics.get("bertscore", {})
            abstain = metrics.get("abstain", {})
            summary[name] = {
                "abstain_rate": abstain.get("abstain_rate"),
                "accuracy":     acc.get("accuracy_overall"),
                "macro_f1":     f1.get("macro_f1"),
                "ece":          ece.get("ece"),
                "recall@5":     recall.get("recall@5"),
                "recall@10":    recall.get("recall@10"),
                "bertscore":    bs.get("bertscore_mean"),
            }
        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")
            summary[name] = {"error": str(e)}

    print(f"\n\n{'='*80}")
    print("WEEK 20 VALIDATION RESULTS — 200 questions, post-fix best weights")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Acc':>5} {'F1':>6} {'ECE':>7} {'R@5':>5} {'R@10':>6} {'Abst%':>6}")
    print("-" * 80)
    for name, m in summary.items():
        if "error" in m:
            print(f"  {name:<28} ERROR: {m['error']}")
        else:
            acc  = f"{m['accuracy']*100:.1f}%" if m.get('accuracy') is not None else "?"
            f1   = f"{m['macro_f1']:.4f}"      if m.get('macro_f1')  is not None else "?"
            ece  = f"{m['ece']:.4f}"           if m.get('ece')       is not None else "?"
            r5   = f"{m['recall@5']:.2f}"      if m.get('recall@5')  is not None else "?"
            r10  = f"{m['recall@10']:.2f}"     if m.get('recall@10') is not None else "?"
            abst = f"{m['abstain_rate']*100:.1f}%" if m.get('abstain_rate') is not None else "?"
            print(f"  {name:<28} {acc:>5} {f1:>6} {ece:>7} {r5:>5} {r10:>6} {abst:>6}")

    summary_path = os.path.join(EVAL_DIR, "val200_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

if __name__ == "__main__":
    main()