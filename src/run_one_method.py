"""
run_one_method.py
Runs exactly ONE method on all 200 questions, in its own fresh process.
Called as a subprocess by run_val200_generate.py — never run multiple
methods in the same interpreter.

Usage:
    python run_one_method.py <method_name>
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "generation_samples")
os.makedirs(OUT_DIR, exist_ok=True)


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
    return questions


def run_method(name, fn, questions, out_path):
    print(f"\n{'='*60}\nMETHOD: {name}  ({len(questions)} questions)\nOutput: {out_path}\n{'='*60}")
    existing = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(questions):
            print(f"  Already complete ({len(existing)} questions) — skipping.")
            return
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
                "pmid": q["pmid"], "question": q["question"],
                "gold_label": q["gold_label"], "gold_contexts": q["gold_contexts"],
                "answer": "abstain", "answer_raw": "", "answer_extracted": "abstain",
                "extraction_method": "error", "confidence": 0.5,
                "s_al": 0.5, "s_ap": 0.5, "a_lp": 0.5, "penalty": False,
                "status": "error", "error": str(e),
            }
        results.append(result)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\n  Done: {success}/{len(questions)} successful")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_one_method.py <method_name>")
        sys.exit(1)
    method_name = sys.argv[1]
    questions = load_val_questions()

    if method_name == "dual_source":
        from pipeline import Pipeline
        pipe = Pipeline()
        run_method("dual_source", lambda q: pipe.run(q), questions,
                    os.path.join(OUT_DIR, "dual_source_val200.json"))
    elif method_name == "baseline_literature_only":
        from baselines import baseline_literature_only
        run_method("baseline_literature_only", baseline_literature_only, questions,
                    os.path.join(OUT_DIR, "baseline_literature_only_val200.json"))
    elif method_name == "baseline_patient_only":
        from baselines import baseline_patient_only
        run_method("baseline_patient_only", baseline_patient_only, questions,
                    os.path.join(OUT_DIR, "baseline_patient_only_val200.json"))
    elif method_name == "baseline_no_retrieval":
        from baselines import baseline_no_retrieval
        run_method("baseline_no_retrieval", baseline_no_retrieval, questions,
                    os.path.join(OUT_DIR, "baseline_no_retrieval_val200.json"))
    elif method_name == "baseline_fixed_chunk":
        from baselines import baseline_fixed_chunk_literature
        run_method("baseline_fixed_chunk", baseline_fixed_chunk_literature, questions,
                    os.path.join(OUT_DIR, "baseline_fixed_chunk_val200.json"))
    elif method_name == "baseline_dual_source_random_patient":
        from baselines import baseline_dual_source_random_patient
        run_method("baseline_dual_source_random_patient", baseline_dual_source_random_patient, questions,
                    os.path.join(OUT_DIR, "baseline_dual_source_random_patient_val200.json"))
    else:
        print(f"Unknown method: {method_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()