"""
split_pubmedqa.py
Week 10 — Farhana (M2)

Splits the PubMedQA expert-labelled dataset (ori_pqal.json, 1000 questions)
into:
    - 200 validation questions  → data/pubmedqa/val_ids.json
    - 800 test questions        → data/pubmedqa/test_ids.json

Uses seed=42 for reproducibility (same split every time you run this).

The split saves only the question IDs (keys), not the full data.
This keeps MIMIC-adjacent data out of git while still being reproducible.

Usage:
    python src/split_pubmedqa.py
"""

import json
import os
import random

# ── Config ────────────────────────────────────────────────────────────────
PQAL_PATH  = "data/pubmedqa/raw/ori_pqal.json"   # 1000 expert-labelled questions
OUT_DIR = "data/pubmedqa/processed"
VAL_PATH   = os.path.join(OUT_DIR, "val_ids.json")
TEST_PATH  = os.path.join(OUT_DIR, "test_ids.json")

VAL_SIZE   = 200
TEST_SIZE  = 800
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────


def main():
    # 1. Load the expert-labelled dataset
    print(f"Loading PubMedQA from: {PQAL_PATH}")
    with open(PQAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ids = list(data.keys())
    total   = len(all_ids)
    print(f"Total questions found: {total}")

    if total < VAL_SIZE + TEST_SIZE:
        raise ValueError(
            f"Not enough questions! Found {total}, need {VAL_SIZE + TEST_SIZE}."
        )

    # 2. Shuffle with fixed seed so split is reproducible
    random.seed(SEED)
    random.shuffle(all_ids)

    val_ids  = all_ids[:VAL_SIZE]
    test_ids = all_ids[VAL_SIZE: VAL_SIZE + TEST_SIZE]

    # 3. Quick sanity check — no overlap between val and test
    overlap = set(val_ids) & set(test_ids)
    assert len(overlap) == 0, f"ERROR: {len(overlap)} IDs appear in both splits!"

    # 4. Save splits
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=2)

    # 5. Print summary with label distribution for each split
    def label_dist(ids):
        counts = {"yes": 0, "no": 0, "maybe": 0}
        for qid in ids:
            label = data[qid].get("final_decision", "unknown").lower()
            if label in counts:
                counts[label] += 1
        return counts

    val_dist  = label_dist(val_ids)
    test_dist = label_dist(test_ids)

    print("\n" + "="*50)
    print("SPLIT COMPLETE")
    print("="*50)
    print(f"Validation set : {len(val_ids):>4} questions → {VAL_PATH}")
    print(f"  Labels: yes={val_dist['yes']}, no={val_dist['no']}, "
          f"maybe={val_dist['maybe']}")
    print(f"Test set       : {len(test_ids):>4} questions → {TEST_PATH}")
    print(f"  Labels: yes={test_dist['yes']}, no={test_dist['no']}, "
          f"maybe={test_dist['maybe']}")
    print(f"Seed used      : {SEED}")
    print(f"Overlap check  : PASSED (0 shared IDs)")
    print("="*50)
    print("\nDone! Use val_ids.json for weight tuning (Week 11).")
    print("Use test_ids.json for main experiment (Week 13+).")


if __name__ == "__main__":
    main()
