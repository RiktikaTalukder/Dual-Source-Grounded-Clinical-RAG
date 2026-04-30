"""
chunking_baselines.py
Week 5 — Riktika
3 chunking strategies for MIMIC discharge notes.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os

# ── Load ClinicalBERT ──────────────────────────────────────────────
print("Loading ClinicalBERT... (this takes ~1 minute first time)")
model = SentenceTransformer("medicalai/ClinicalBERT")
print("Model loaded!")


# ── Helper: split text into word-based chunks ──────────────────────
def split_into_words(text):
    return text.split()


# ── Baseline A: Fixed-size chunks (512 tokens, 10% overlap) ───────
def chunk_fixed(text, size=512, overlap=0.10):
    """Cut text into fixed chunks of `size` words with 10% overlap."""
    words = split_into_words(text)
    step = int(size * (1 - overlap))   # how far to move each time
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


# ── Baseline B: Sliding-window (256 tokens, 50% overlap) ──────────
def chunk_sliding(text, size=256, overlap=0.50):
    """Sliding window — smaller chunks, lots of overlap."""
    words = split_into_words(text)
    step = int(size * (1 - overlap))
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


# ── Baseline C: Retrieval-guided dynamic chunk selection ───────────
def chunk_dynamic(text, query, top_n=3, size=256):
    """
    Split into small chunks, embed them + the query,
    then return the top_n chunks most similar to the query.
    """
    # 1. Make candidate chunks (sliding window, no overlap)
    words = split_into_words(text)
    candidates = []
    for start in range(0, len(words), size):
        chunk = words[start : start + size]
        if chunk:
            candidates.append(" ".join(chunk))

    if not candidates:
        return []

    # 2. Embed query and all candidate chunks
    query_emb = model.encode([query], normalize_embeddings=True)
    chunk_embs = model.encode(candidates, normalize_embeddings=True)

    # 3. Cosine similarity = dot product (because embeddings are normalized)
    scores = np.dot(chunk_embs, query_emb.T).flatten()

    # 4. Pick top_n chunks
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [candidates[i] for i in top_indices]


# ── Build FAISS index from MIMIC notes ────────────────────────────
def build_faiss_index(notes_list, strategy="fixed"):
    """
    Takes a list of note strings.
    Chunks them all, embeds them, builds a FAISS index.
    Returns (index, all_chunks_list)
    """
    all_chunks = []

    print(f"Chunking {len(notes_list)} notes using strategy: {strategy}")
    for note in notes_list:
        if strategy == "fixed":
            all_chunks.extend(chunk_fixed(note))
        elif strategy == "sliding":
            all_chunks.extend(chunk_sliding(note))
        else:
            raise ValueError("Use 'fixed' or 'sliding' for FAISS index building")

    print(f"Total chunks: {len(all_chunks)} — now embedding...")
    embeddings = model.encode(all_chunks, batch_size=32,
                               show_progress_bar=True,
                               normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine sim (normalized)
    index.add(embeddings.astype("float32"))

    print(f"FAISS index built! {index.ntotal} vectors, dim={dim}")
    return index, all_chunks


# ── Save index to disk ─────────────────────────────────────────────
def save_index(index, all_chunks, index_path, chunks_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    pd.Series(all_chunks).to_csv(chunks_path, index=False, header=["chunk"])
    print(f"Saved index → {index_path}")
    print(f"Saved chunks → {chunks_path}")


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_note = (
        "The patient is a 65-year-old male admitted with chest pain. "
        "History of hypertension and diabetes. ECG showed ST elevation. "
        "Started on aspirin and heparin. Cardiology consulted. "
        "Patient underwent PCI successfully. Discharged in stable condition. "
    ) * 30   # repeat to make it long enough to chunk

    print("\n--- Baseline A (Fixed 512, 10% overlap) ---")
    chunks_a = chunk_fixed(sample_note)
    print(f"Number of chunks: {len(chunks_a)}")
    print(f"First chunk preview: {chunks_a[0][:100]}...")

    print("\n--- Baseline B (Sliding 256, 50% overlap) ---")
    chunks_b = chunk_sliding(sample_note)
    print(f"Number of chunks: {len(chunks_b)}")

    print("\n--- Baseline C (Dynamic, query-guided) ---")
    query = "what medication was given for chest pain?"
    chunks_c = chunk_dynamic(sample_note, query, top_n=3)
    print(f"Top 3 chunks returned: {len(chunks_c)}")
    print(f"Best match preview: {chunks_c[0][:150]}...")
