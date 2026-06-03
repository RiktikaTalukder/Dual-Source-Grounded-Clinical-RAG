"""
evaluate_recall.py
Week 6 – Farhana (M2)
Computes Recall@5 and Recall@10 on 50 PubMedQA questions.
Saves results to /results/recall_baseline.csv

Recall method: semantic similarity using BiomedBERT embeddings
(same model as pmc_retriever.py — pritamdeka/S-PubMedBert-MS-MARCO).
A retrieved passage counts as a hit if its cosine similarity to any
gold context passage is >= 0.5.

Note on cross-corpus overlap: the 50 questions sampled here use seed=42
from the full 1000-question pool. The test split (test_ids.json) was also
created with seed=42. There may be overlap between these 50 questions and
the test set — this is a known limitation to note in thesis §5.
"""

import json
import csv
import os
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from pmc_retriever import retrieve_literature

_base         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBMEDQA_PATH = os.path.join(_base, "data", "pubmedqa", "raw", "ori_pqal.json")
RESULTS_PATH  = os.path.join(_base, "results", "recall_baseline.csv")
NUM_QUESTIONS = 50
RANDOM_SEED   = 42

# Semantic similarity threshold: a retrieved passage is a hit if
# cosine similarity to any gold context passage >= this value
SIMILARITY_THRESHOLD = 0.5

# Same model as pmc_retriever.py so the comparison is fair
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Load embedding model once
print("Loading BiomedBERT for semantic recall evaluation...")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Model loaded.")


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of text strings, return normalised numpy array."""
    return _embed_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )


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


def compute_recall_at_k(retrieved_passages: list, gold_contexts: list, k: int) -> float:
    """
    Semantic Recall@k.

    A gold context passage counts as 'found' if any of the top-k retrieved
    passages has cosine similarity >= SIMILARITY_THRESHOLD with it.

    Uses BiomedBERT embeddings (same model as pmc_retriever.py) so that
    the similarity measure is consistent with how retrieval itself works.

    Parameters
    ----------
    retrieved_passages : list of dicts, each with key 'passage' (from retrieve_literature)
    gold_contexts      : list of str — the ground-truth context passages from PubMedQA
    k                  : int — how many retrieved passages to consider

    Returns
    -------
    float — fraction of gold contexts that were semantically matched (0.0 to 1.0)
    """
    if not gold_contexts or not retrieved_passages:
        return 0.0

    # Get the top-k retrieved passage texts
    retrieved_texts = [r["passage"] for r in retrieved_passages[:k]]

    # Embed retrieved passages and gold contexts
    retrieved_vecs = embed_texts(retrieved_texts)   # shape: (k, dim)
    gold_vecs      = embed_texts(gold_contexts)     # shape: (n_gold, dim)

    # Similarity matrix: (n_gold, k) — cosine sim for each gold-retrieved pair
    sim_matrix = np.dot(gold_vecs, retrieved_vecs.T)

    # A gold context is 'hit' if any retrieved passage exceeds the threshold
    hits = 0
    for gold_idx in range(len(gold_contexts)):
        if np.any(sim_matrix[gold_idx] >= SIMILARITY_THRESHOLD):
            hits += 1

    return hits / len(gold_contexts)


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

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
    print(f"  Method            : semantic (BiomedBERT cosine >= {SIMILARITY_THRESHOLD})")
    print(f"{'='*50}")

    # Save to CSV
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
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