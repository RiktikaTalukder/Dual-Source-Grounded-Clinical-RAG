"""
pmc_embedder.py
Week 6 – Farhana (M2)
Embeds PMC articles using BiomedBERT and builds a FAISS index.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
PMC_DATA_PATH   = "data/pmc_literature/pmc_sample/"       # folder with your 500 PMC JSON files
INDEX_SAVE_PATH = "data/indexes/pmc_articles.index"
TEXTS_SAVE_PATH = "data/indexes/pmc_texts.json"   # we'll save raw texts too
MODEL_NAME      = "pritamdeka/S-PubMedBert-MS-MARCO"
BATCH_SIZE      = 32
# ────────────────────────────────────────────────────────────────────────────

def load_pmc_articles(folder_path):
    """Load all JSON files from the PMC sample folder."""
    articles = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                articles.append(data)
    print(f"Loaded {len(articles)} PMC articles.")
    return articles

def build_passage_texts(articles):
    """
    For each article, combine: title + abstract + first 3 body paragraphs.
    This is what we will embed.
    """
    passages = []
    for art in articles:
        title    = art.get("title", "")
        abstract = art.get("abstract", "")
        body     = art.get("body_paragraphs", [])
        first3   = " ".join(body[:3]) if body else ""
        passage  = f"{title}. {abstract} {first3}".strip()
        passages.append(passage)
    return passages

def embed_and_index(passages, model_name, batch_size):
    """Embed all passages and build a FAISS flat index."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(passages)} passages in batches of {batch_size}...")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # cosine similarity works best with normalized vectors
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim (for normalized vectors)
    index.add(embeddings)
    print(f"FAISS index built. Total vectors: {index.ntotal}, Dimension: {dim}")
    return index, embeddings

def main():
    os.makedirs("data/indexes", exist_ok=True)

    articles = load_pmc_articles(PMC_DATA_PATH)
    passages = build_passage_texts(articles)

    index, _ = embed_and_index(passages, MODEL_NAME, BATCH_SIZE)

    # Save FAISS index
    faiss.write_index(index, INDEX_SAVE_PATH)
    print(f"Index saved to {INDEX_SAVE_PATH}")

    # Save texts so we can retrieve them later by index position
    with open(TEXTS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)
    print(f"Passage texts saved to {TEXTS_SAVE_PATH}")

if __name__ == "__main__":
    main()