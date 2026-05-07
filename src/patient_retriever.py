"""
patient_retriever.py
Retrieves top-K similar historical patient cases given a query note.
Similarity = 0.6 × ClinicalBERT cosine sim + 0.4 × ICD Jaccard overlap
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME    = "medicalai/ClinicalBERT"
INDEX_PATH = os.path.join(_base, "data", "indexes", "mimic_patients.index")
import os
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(_base, "data", "patient_metadata.csv") if os.path.exists(os.path.join(_base, "data", "patient_metadata.csv")) else os.path.join(_base, "data", "mimic", "processed", "patient_metadata.csv")
NOTES_DIR     = "data/mimic/mimic_sample"
ALPHA         = 0.6   # weight for embedding similarity
BETA          = 0.4   # weight for ICD Jaccard overlap
TOP_K         = 3
# ────────────────────────────────────────────────────────────────────────

def load_resources():
    print("Loading ClinicalBERT model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading patient metadata...")
    meta = pd.read_csv(METADATA_PATH)
    meta["icd_set"] = meta["icd_codes_top5"].fillna("").apply(
        lambda x: set(str(x).split(",")) - {""}
    )

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("All resources loaded.\n")
    return model, meta, index


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def retrieve(query_text: str, query_icd: set,
             model, meta, index, top_k: int = TOP_K):
    # 1. Embed query
    q_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")

    # 2. FAISS cosine search (index stores L2-normalised vectors)
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(q_vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        row = meta.iloc[idx]
        cos_sim  = float(dist)                        # already normalised → dot = cosine
        icd_sim  = jaccard(query_icd, row["icd_set"])
        combined = ALPHA * cos_sim + BETA * icd_sim
        results.append({
            "rank_score"     : combined,
            "cos_sim"        : cos_sim,
            "icd_jaccard"    : icd_sim,
            "subject_id"     : row["subject_id"],
            "age"            : row["age"],
            "gender"         : row["gender"],
            "admission_type" : row["admission_type"],
            "icd_codes"      : row["icd_codes_top5"],
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:top_k]


def build_patient_index(model, meta):
    """Embed one representative note per patient and save FAISS index."""
    print("Building patient FAISS index from mimic_sample notes...")
    texts = []
    valid_indices = []

    for i, row in meta.iterrows():
        # Try to find a note file for this subject
        note_files = [f for f in os.listdir(NOTES_DIR)
                      if f.endswith(".txt") and not f.endswith("_meta.json")]
        if i < len(note_files):
            note_path = os.path.join(NOTES_DIR, sorted(note_files)[i])
            with open(note_path, "r", encoding="utf-8") as f:
                texts.append(f.read()[:1000])   # first 1000 chars as representative
            valid_indices.append(i)
        else:
            texts.append(f"Patient age {row['age']} gender {row['gender']} "
                         f"admission {row['admission_type']} ICD {row['icd_codes_top5']}")
            valid_indices.append(i)

    print(f"Embedding {len(texts)} patient records...")
    embeddings = model.encode(texts, normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # Inner Product = cosine on normalised vecs
    index.add(embeddings)

    os.makedirs("data/indexes", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Patient index saved → {INDEX_PATH}  ({index.ntotal} vectors)")
    return index


# ── Main demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Build index if it doesn't exist yet
    if not os.path.exists(INDEX_PATH):
        print("Index not found — building it now...")
        tmp_model = SentenceTransformer(MODEL_NAME)
        tmp_meta  = pd.read_csv(METADATA_PATH)
        tmp_meta["icd_set"] = tmp_meta["icd_codes_top5"].fillna("").apply(
            lambda x: set(str(x).split(",")) - {""}
        )
        build_patient_index(tmp_model, tmp_meta)

    model, meta, index = load_resources()

    # 5 sample queries
    sample_queries = [
        ("Patient admitted with severe sepsis and multi-organ failure, requiring ICU care.",
         {"99591", "99592", "99811"}),
        ("Elderly patient with acute heart failure and fluid overload.",
         {"42831", "40291", "5849"}),
        ("Patient with COPD exacerbation and respiratory failure needing ventilator support.",
         {"49121", "51881", "9670"}),
        ("Type 2 diabetes patient with hyperglycaemic crisis and ketoacidosis.",
         {"25010", "25001", "2762"}),
        ("Post-operative patient with wound infection and fever after abdominal surgery.",
         {"9985", "5990", "78650"}),
    ]

    for i, (query_text, query_icd) in enumerate(sample_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query_text}")
        print(f"Query ICD codes: {query_icd}")
        print(f"{'='*60}")
        results = retrieve(query_text, query_icd, model, meta, index)
        for rank, r in enumerate(results, 1):
            print(f"\n  Rank {rank} | Combined Score: {r['rank_score']:.4f} "
                  f"(cos={r['cos_sim']:.4f}, icd_jacc={r['icd_jaccard']:.4f})")
            print(f"  Subject: {r['subject_id']} | Age: {r['age']} | "
                  f"Gender: {r['gender']} | Admission: {r['admission_type']}")
            print(f"  ICD codes: {r['icd_codes']}")
