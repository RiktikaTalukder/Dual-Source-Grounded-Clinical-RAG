"""
run_baselines_batch.py
Week 10 — Farhana (M2)

Runs all 4 baselines on the same 20 PubMedQA-style questions
used by Riktika in run_20_queries.py (Week 9).

Saves one output file per baseline to:
    results/generation_samples/baseline_literature_only_20.json
    results/generation_samples/baseline_patient_only_20.json
    results/generation_samples/baseline_no_retrieval_20.json
    results/generation_samples/baseline_fixed_chunk_20.json
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (
    baseline_literature_only,
    baseline_patient_only,
    baseline_no_retrieval,
    baseline_fixed_chunk_literature,
)

# ── Same 20 queries as run_20_queries.py ──────────────────────────────────
QUERIES = [
    "Does N-acetylcysteine reduce exacerbations in COPD patients?",
    "Is metformin effective for type 2 diabetes management?",
    "Does aspirin reduce the risk of cardiovascular events?",
    "Is antibiotic therapy effective for community-acquired pneumonia?",
    "Does physical activity reduce the risk of type 2 diabetes?",
    "Is corticosteroid therapy beneficial in septic shock?",
    "Does breastfeeding reduce the risk of childhood obesity?",
    "Is laparoscopic surgery better than open surgery for appendicitis?",
    "Does statins therapy reduce mortality in heart failure patients?",
    "Is cognitive behavioural therapy effective for depression?",
    "Does early mobilization improve outcomes in ICU patients?",
    "Is beta-blocker therapy beneficial after myocardial infarction?",
    "Does vitamin D supplementation reduce fracture risk in elderly?",
    "Is chemotherapy effective for non-small cell lung cancer?",
    "Does hand hygiene reduce hospital-acquired infections?",
    "Is insulin therapy necessary for type 1 diabetes?",
    "Does obesity increase the risk of sleep apnea?",
    "Is thrombolysis effective in acute ischemic stroke?",
    "Does smoking cessation reduce cardiovascular disease risk?",
    "Is prophylactic anticoagulation beneficial in hospitalized patients?"
]

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "generation_samples"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Each baseline maps to its function and output filename ────────────────
BASELINES = [
    ("literature_only",    baseline_literature_only,         "baseline_literature_only_20.json"),
    ("patient_only",       baseline_patient_only,            "baseline_patient_only_20.json"),
    ("no_retrieval",       baseline_no_retrieval,            "baseline_no_retrieval_20.json"),
    ("fixed_chunk",        baseline_fixed_chunk_literature,  "baseline_fixed_chunk_20.json"),
]


def run_one_baseline(name, fn, filename):
    out_path = os.path.join(OUT_DIR, filename)
    print(f"\n{'='*60}")
    print(f"BASELINE: {name}  ({len(QUERIES)} queries)")
    print(f"Output  : {out_path}")
    print('='*60)

    results = []
    for i, query in enumerate(QUERIES, 1):
        print(f"\n  [{i}/{len(QUERIES)}] {query[:70]}")
        t0 = time.time()
        try:
            result = fn(query)
            result["query_id"] = i
            result["status"]   = "success"
            result["runtime_seconds"] = round(time.time() - t0, 2)
            print(f"    confidence={result['confidence']}  "
                  f"penalty={'YES' if result['penalty'] else 'no'}  "
                  f"time={result['runtime_seconds']}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {
                "baseline":    name,
                "query_id":    i,
                "query":       query,
                "status":      "error",
                "error":       str(e),
                "confidence":  0,
                "penalty":     False
            }
        results.append(result)

        # Save after every query so progress is never lost
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Summary for this baseline
    success   = sum(1 for r in results if r.get("status") == "success")
    avg_conf  = sum(r.get("confidence", 0) for r in results) / len(results)
    penalised = sum(1 for r in results if r.get("penalty"))
    print(f"\n  DONE: {success}/{len(QUERIES)} succeeded | "
          f"avg confidence={avg_conf:.4f} | penalties={penalised}")
    return results


def main():
    print("Week 10 — Baseline Batch Runner")
    print(f"Running {len(BASELINES)} baselines × {len(QUERIES)} queries\n")

    summary = {}
    for name, fn, filename in BASELINES:
        results = run_one_baseline(name, fn, filename)
        avg_conf  = sum(r.get("confidence", 0) for r in results) / len(results)
        penalised = sum(1 for r in results if r.get("penalty"))
        summary[name] = {
            "avg_confidence": round(avg_conf, 4),
            "penalties":      penalised,
            "output_file":    filename
        }

    print(f"\n{'='*60}")
    print("ALL BASELINES COMPLETE — SUMMARY")
    print('='*60)
    print(f"{'Baseline':<30} {'Avg Conf':>9} {'Penalties':>10}")
    print('-'*55)
    for name, s in summary.items():
        print(f"{name:<30} {s['avg_confidence']:>9.4f} {s['penalties']:>10}")
    print('='*60)
    print(f"\nAll output files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
