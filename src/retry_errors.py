"""
retry_errors.py
Finds any rows with status == "error" in a generation output file
and reprocesses just those, in place. Leaves successful rows untouched.

Usage:
    python src/retry_errors.py <method_name> <output_filename>
Example:
    python src/retry_errors.py baseline_dual_source_random_patient baseline_dual_source_random_patient_val200.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one_method import load_val_questions

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "generation_samples")

METHOD_FUNCS = {
    "dual_source":                          None,  # special-cased below
    "baseline_literature_only":             "baselines.baseline_literature_only",
    "baseline_patient_only":                "baselines.baseline_patient_only",
    "baseline_no_retrieval":                "baselines.baseline_no_retrieval",
    "baseline_fixed_chunk":                 "baselines.baseline_fixed_chunk_literature",
    "baseline_dual_source_random_patient":  "baselines.baseline_dual_source_random_patient",
}

def main():
    if len(sys.argv) != 3:
        print("Usage: python retry_errors.py <method_name> <output_filename>")
        sys.exit(1)
    method_name, filename = sys.argv[1], sys.argv[2]
    out_path = os.path.join(OUT_DIR, filename)

    with open(out_path) as f:
        results = json.load(f)

    error_indices = [i for i, r in enumerate(results) if r.get("status") == "error"]
    if not error_indices:
        print("No error rows found. Nothing to retry.")
        return
    print(f"Found {len(error_indices)} error row(s) at index/indices: {error_indices}")

    # Get fn
    if method_name == "dual_source":
        from pipeline import Pipeline
        pipe = Pipeline()
        fn = lambda q: pipe.run(q)
    else:
        mod_name, fn_name = METHOD_FUNCS[method_name].split(".")
        mod = __import__(mod_name)
        fn = getattr(mod, fn_name)

    questions = load_val_questions()
    pmid_to_q = {q["pmid"]: q for q in questions}

    for i in error_indices:
        pmid = results[i]["pmid"]
        q = pmid_to_q[pmid]
        print(f"  Retrying pmid={pmid}: {q['question'][:70]}...")
        t0 = time.time()
        try:
            result = fn(q["question"])
            result["pmid"]          = q["pmid"]
            result["gold_label"]    = q["gold_label"]
            result["gold_contexts"] = q["gold_contexts"]
            result["status"]        = "success"
            result["runtime_s"]     = round(time.time() - t0, 2)
            results[i] = result
            print(f"    Success: {result['answer_extracted']}")
        except Exception as e:
            print(f"    STILL FAILING: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    success = sum(1 for r in results if r.get("status") == "success")
    print(f"Final: {success}/{len(results)} successful")

if __name__ == "__main__":
    main()