"""
run_val200.py
Week 20 — Farhana (M2)

Runs all 5 methods on the 200-question validation set using the best weights
from config.py (set by grid search). Saves output JSONs and runs evaluate.py
on each. Results go directly into the paper's results table.

Methods:
  1. dual_source           — full pipeline (pipeline.py)
  2. literature_only       — baseline_literature_only
  3. patient_only          — baseline_patient_only
  4. no_retrieval          — baseline_no_retrieval
  5. fixed_chunk           — baseline_fixed_chunk_literature

Usage:
    python src/run_val200.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (
    baseline_literature_only,
    baseline_patient_only,
    baseline_no_retrieval,
    baseline_fixed_chunk_literature,
)
from pipeline import Pipeline
from evaluate import evaluate as run_evaluation

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE, "results", "generation_samples")
EVAL_DIR = os.path.join(BASE, "results", "val200_metrics")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ── Load 200 validation questions with gold labels and gold contexts ───────
def load_val_questions():
    val_ids_path = os.path.join(BASE, "data", "pubmedqa", "processed", "val_ids.json")
    raw_path     = os.path.join(BASE, "data", "pubmedqa", "raw", "ori_pqal.json")

    with open(val_ids_path) as f:
        val_ids = set(json.load(f))
    with open(raw_path) as f:
        raw = json.load(f)

    questions = []
    for pmid, entry in raw.items():
        if pmid not in val_ids:
            continue
        questions.append({
            "pmid":          pmid,
            "question":      entry.get("QUESTION", ""),
            "gold_label":    entry.get("final_decision", "maybe").lower().strip(),
            "gold_contexts": entry.get("CONTEXTS", []),
        })
    print(f"[run_val200] Loaded {len(questions)} validation questions.")
    return questions


# ── Run one method on all 200 questions, resume if partially done ──────────
def run_method(name, fn, questions, out_path):
    print(f"\n{'='*60}")
    print(f"METHOD: {name}  ({len(questions)} questions)")
    print(f"Output: {out_path}")
    print("=" * 60)

    # Resume if already partially or fully done
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(questions):
            print(f"  Already complete ({len(existing)} questions) — skipping.")
            return existing
        else:
            print(f"  Resuming from question {len(existing)+1}...")

    results = list(existing)
    start_from = len(existing)

    for i, q in enumerate(questions[start_from:], start=start_from):
        print(f"  [{i+1:>3}/{len(questions)}] {q['question'][:70]}...")
        t0 = time.time()
        try:
            result = fn(q["question"])
            result["pmid"]          = q["pmid"]
            result["gold_label"]    = q["gold_label"]
            result["gold_contexts"] = q["gold_contexts"]
            result["status"]        = "success"
            result["runtime_s"]     = round(time.time() - t0, 2)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {
                "pmid":              q["pmid"],
                "question":          q["question"],
                "gold_label":        q["gold_label"],
                "gold_contexts":     q["gold_contexts"],
                "answer":            "abstain",
                "answer_raw":        "",
                "answer_extracted":  "abstain",
                "extraction_method": "error",
                "confidence":        0.5,
                "s_al": 0.5, "s_ap": 0.5, "a_lp": 0.5,
                "penalty":           False,
                "status":            "error",
                "error":             str(e),
            }
        results.append(result)

        # Save after every question so nothing is lost
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\n  Done: {success}/{len(questions)} successful")
    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    questions = load_val_questions()

    # Load pipeline once for dual_source
    print("\n[run_val200] Loading dual-source pipeline (models load once)...")
    pipe = Pipeline()
    print("[run_val200] Pipeline ready.\n")

    # All 5 methods
    methods = [
        ("dual_source",
         lambda q: pipe.run(q),
         "dual_source_val200.json"),
        ("baseline_literature_only",
         baseline_literature_only,
         "baseline_literature_only_val200.json"),
        ("baseline_patient_only",
         baseline_patient_only,
         "baseline_patient_only_val200.json"),
        ("baseline_no_retrieval",
         baseline_no_retrieval,
         "baseline_no_retrieval_val200.json"),
        ("baseline_fixed_chunk",
         baseline_fixed_chunk_literature,
         "baseline_fixed_chunk_val200.json"),
    ]

    all_output_paths = []
    for name, fn, filename in methods:
        out_path = os.path.join(OUT_DIR, filename)
        run_method(name, fn, questions, out_path)
        all_output_paths.append((name, out_path))

    # ── Run evaluate.py on each output file ───────────────────────────────
    print(f"\n\n{'='*60}")
    print("EVALUATING ALL 5 METHODS")
    print("=" * 60)

    summary = {}
    for name, out_path in all_output_paths:
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

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("WEEK 20 VALIDATION RESULTS — 200 questions, post-fix best weights")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Acc':>5} {'F1':>6} {'ECE':>7} {'R@5':>5} {'R@10':>6} {'Abst%':>6}")
    print("-" * 80)
    for name, m in summary.items():
        if "error" in m:
            print(f"  {name:<28} ERROR: {m['error']}")
        else:
            acc   = f"{m['accuracy']*100:.1f}%" if m.get('accuracy') is not None else "?"
            f1    = f"{m['macro_f1']:.4f}"      if m.get('macro_f1')  is not None else "?"
            ece   = f"{m['ece']:.4f}"           if m.get('ece')       is not None else "?"
            r5    = f"{m['recall@5']:.2f}"      if m.get('recall@5')  is not None else "?"
            r10   = f"{m['recall@10']:.2f}"     if m.get('recall@10') is not None else "?"
            abst  = f"{m['abstain_rate']*100:.1f}%" if m.get('abstain_rate') is not None else "?"
            print(f"  {name:<28} {acc:>5} {f1:>6} {ece:>7} {r5:>5} {r10:>6} {abst:>6}")

    # Save summary JSON
    summary_path = os.path.join(EVAL_DIR, "val200_metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run_val200] Summary saved → {summary_path}")
    print("\n✅ All 5 methods complete. These numbers go into your paper.")


if __name__ == "__main__":
    main()