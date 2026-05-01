"""
evaluate_recall.py
Week 6 – Farhana (M2)
Computes Recall@5 and Recall@10 on 50 PubMedQA questions.
Saves results to /results/recall_baseline.csv
"""

import json
import csv
import os
import random
from pmc_retriever import retrieve_literature   # our function from Step 5

PUBMEDQA_PATH = "data/pubmedqa/raw/ori_pqal.json"  # adjust path if different
RESULTS_PATH  = "results/recall_baseline.csv"
NUM_QUESTIONS = 50
RANDOM_SEED   = 42

def load_pubmedqa_sample(path, n, seed):
    """Load n random questions from PubMedQA expert-labeled set."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ids = list(data.keys())
    random.seed(seed)
    sampled_ids = random.sample(all_ids, n)

    questions = []
    for qid in sampled_ids:
        entry = data[qid]
        questions.append({
            "id":       qid,
            "question": entry["QUESTION"],
            "context":  entry["CONTEXTS"],   # list of gold passages
            "answer":   entry["final_decision"]
        })
    return questions

def compute_recall_at_k(retrieved_passages, gold_contexts, k):
    """
    Recall@k = fraction of gold context passages that appear in top-k retrieved.
    We use a simple lexical overlap check (shared words).
    """
    retrieved_k = [r["passage"].lower() for r in retrieved_passages[:k]]

    hits = 0
    for gold in gold_contexts:
        gold_lower = gold.lower()
        # Check if any retrieved passage overlaps significantly with this gold passage
        gold_words = set(gold_lower.split())
        for ret in retrieved_k:
            ret_words = set(ret.split())
            overlap   = len(gold_words & ret_words) / max(len(gold_words), 1)
            if overlap > 0.3:   # 30% word overlap threshold
                hits += 1
                break

    return hits / max(len(gold_contexts), 1)

def main():
    os.makedirs("results", exist_ok=True)

    print(f"Loading {NUM_QUESTIONS} PubMedQA questions...")
    questions = load_pubmedqa_sample(PUBMEDQA_PATH, NUM_QUESTIONS, RANDOM_SEED)

    rows = []
    recall5_scores  = []
    recall10_scores = []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{NUM_QUESTIONS}] {q['question'][:80]}...")

        # Retrieve top-10 (so we can compute both @5 and @10)
        retrieved = retrieve_literature(q["question"], k=10)

        r5  = compute_recall_at_k(retrieved, q["context"], k=5)
        r10 = compute_recall_at_k(retrieved, q["context"], k=10)

        recall5_scores.append(r5)
        recall10_scores.append(r10)

        rows.append({
            "question_id": q["id"],
            "question":    q["question"][:100],
            "recall@5":    round(r5, 4),
            "recall@10":   round(r10, 4),
            "answer":      q["answer"]
        })

    avg_r5  = sum(recall5_scores)  / len(recall5_scores)
    avg_r10 = sum(recall10_scores) / len(recall10_scores)

    print(f"\n{'='*50}")
    print(f"  Average Recall@5  : {avg_r5:.4f}")
    print(f"  Average Recall@10 : {avg_r10:.4f}")
    print(f"{'='*50}")

    # Save to CSV
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        # Add summary row
        writer.writerow({
            "question_id": "AVERAGE",
            "question":    "",
            "recall@5":    round(avg_r5, 4),
            "recall@10":   round(avg_r10, 4),
            "answer":      ""
        })

    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()