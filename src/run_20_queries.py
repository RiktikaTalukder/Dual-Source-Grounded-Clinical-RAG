"""
run_20_queries.py
Week 9 - Riktika (M1)

Runs dual_source_rag on 20 PubMedQA questions.
Saves outputs to results/generation_samples/dual_source_20.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import dual_source_rag

# 20 real PubMedQA-style clinical questions
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

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "generation_samples", "dual_source_20.json"
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def main():
    print(f"Running dual_source_rag on {len(QUERIES)} queries...")
    print(f"Output will be saved to: {OUTPUT_PATH}\n")

    results = []
    for i, query in enumerate(QUERIES):
        print(f"\n{'='*60}")
        print(f"Query {i+1}/{len(QUERIES)}: {query}")
        print('='*60)
        try:
            result = dual_source_rag(query, top_k_lit=3, top_k_pat=3)
            result["query_id"] = i + 1
            result["status"] = "success"
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            result = {
                "query_id": i + 1,
                "query": query,
                "status": "error",
                "error": str(e)
            }
        results.append(result)

        # Save after every query so we don't lose progress if it crashes
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Saved {i+1}/{len(QUERIES)} to file]")

    # Final summary
    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {success}/{len(QUERIES)} succeeded")
    avg_conf = sum(r.get("confidence", 0) for r in results if r.get("status") == "success") / max(success, 1)
    avg_time = sum(r.get("runtime_seconds", 0) for r in results if r.get("status") == "success") / max(success, 1)
    print(f"Average confidence : {round(avg_conf, 4)}")
    print(f"Average runtime    : {round(avg_time, 2)}s per query")
    print(f"Output saved to    : {OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()
