"""
assemble_val200_summary.py
Builds results/val200_metrics/val200_metrics_summary.json by reading the
5 already-computed results/main_experiment/metrics_*_val200.json files
directly. Does NOT call evaluate.py and does NOT load any models — pure
JSON read/write, so it cannot reproduce the model-stacking crash that
run_val200_evaluate.py hit.

Run this AFTER all 5 metrics_*_val200.json files exist and look correct.

Usage:
    python src/assemble_val200_summary.py
"""

import json
import os

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.join(BASE, "results", "main_experiment")
EVAL_DIR = os.path.join(BASE, "results", "val200_metrics")
os.makedirs(EVAL_DIR, exist_ok=True)

# (summary_key, metrics_filename)
METHODS = [
    ("dual_source",             "metrics_dual_source_val200.json"),
    ("baseline_literature_only","metrics_baseline_literature_only_val200.json"),
    ("baseline_patient_only",   "metrics_baseline_patient_only_val200.json"),
    ("baseline_no_retrieval",   "metrics_baseline_no_retrieval_val200.json"),
    ("baseline_fixed_chunk",    "metrics_baseline_fixed_chunk_val200.json"),
    ("baseline_dual_source_random_patient", "metrics_baseline_dual_source_random_patient_val200.json"),
]

def main():
    summary = {}
    for name, filename in METHODS:
        path = os.path.join(MAIN_DIR, filename)
        if not os.path.exists(path):
            print(f"--- {name}: SKIPPED — {filename} not found ---")
            summary[name] = {"error": f"{filename} not found"}
            continue

        with open(path) as f:
            m = json.load(f)

        abstain = m.get("abstain", {})
        acc     = m.get("accuracy", {})
        f1      = m.get("f1", {})
        ece     = m.get("ece", {})
        recall  = m.get("recall", {})
        bs      = m.get("bertscore", {})

        summary[name] = {
            "abstain_rate": abstain.get("abstain_rate"),
            "accuracy":     acc.get("accuracy_overall"),
            "macro_f1":     f1.get("macro_f1"),
            "ece":          ece.get("ece"),
            "recall@5":     recall.get("recall@5"),
            "recall@10":    recall.get("recall@10"),
            "bertscore":    bs.get("bertscore_mean"),
        }

    print(f"\n{'='*80}")
    print("ASSEMBLED VAL200 SUMMARY (from individually-run metrics files)")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Acc':>7} {'F1':>7} {'ECE':>8} {'R@5':>6} {'R@10':>6} {'Abst%':>7} {'BERTScore':>10}")
    print("-" * 90)
    for name, m in summary.items():
        if "error" in m:
            print(f"  {name:<28} ERROR: {m['error']}")
            continue
        acc  = f"{m['accuracy']*100:.1f}%"     if m.get('accuracy')     is not None else "?"
        f1v  = f"{m['macro_f1']:.4f}"          if m.get('macro_f1')     is not None else "?"
        ecev = f"{m['ece']:.4f}"               if m.get('ece')          is not None else "?"
        r5   = f"{m['recall@5']:.3f}"          if m.get('recall@5')     is not None else "?"
        r10  = f"{m['recall@10']:.3f}"         if m.get('recall@10')    is not None else "?"
        abst = f"{m['abstain_rate']*100:.1f}%" if m.get('abstain_rate') is not None else "?"
        bsv  = f"{m['bertscore']:.4f}"         if m.get('bertscore')    is not None else "N/A"
        print(f"  {name:<28} {acc:>7} {f1v:>7} {ecev:>8} {r5:>6} {r10:>6} {abst:>7} {bsv:>10}")

    out_path = os.path.join(EVAL_DIR, "val200_metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {out_path}")

if __name__ == "__main__":
    main()