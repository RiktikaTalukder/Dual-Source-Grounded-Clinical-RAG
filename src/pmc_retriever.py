"""
pmc_retriever.py
Week 6 – Farhana (M2)
Retrieves top-k PMC literature passages for a given query.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_PATH = "data/indexes/pmc_articles.index"
TEXTS_PATH = "data/indexes/pmc_texts.json"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
# ─────────────────────────────────────────────────────────────────────────────

# Load once at module level (so it doesn't reload every function call)
_model = None
_index = None
_texts = None

def _load_resources():
    global _model, _index, _texts
    if _model is None:
        print("Loading BiomedBERT model...")
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        print("Loading FAISS index...")
        _index = faiss.read_index(INDEX_PATH)
    if _texts is None:
        print("Loading passage texts...")
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            _texts = json.load(f)
    print("All resources loaded.")

def retrieve_literature(query: str, k: int = 5) -> list:
    """
    Given a clinical query string, return the top-k most relevant PMC passages.

    Parameters
    ----------
    query : str   — the clinical question or text to search for
    k     : int   — how many results to return (default 5)

    Returns
    -------
    list of dicts, each with keys: 'rank', 'score', 'passage'
    """
    _load_resources()

    # Embed the query the same way we embedded the articles
    query_vec = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search the FAISS index
    scores, indices = _index.search(query_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        results.append({
            "rank":    rank,
            "score":   float(score),
            "passage": _texts[idx]
        })

    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the risk factors for sepsis in ICU patients?",
        "How does heart failure affect kidney function?",
        "What medications are used to treat type 2 diabetes?"
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print(f"{'='*60}")
        results = retrieve_literature(q, k=5)
        for r in results:
            print(f"\n  Rank {r['rank']} | Score: {r['score']:.4f}")
            print(f"  {r['passage'][:300]}...")