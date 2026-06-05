"""
patient_retriever.py
Week 14 — Phase B Core Fixes

Changes from previous version:
- Embedding model: ClinicalBERT → S-PubMedBert-MS-MARCO (unified model)
- jaccard(): ICD codes normalized to first 3 chars before comparison
- build_patient_index(): embeds bhc field instead of raw note first 1000 chars
- ICD sets: asserted to contain only strings, never integers
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
_base         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_base, "src"))
from config import MODEL_REVISIONS

MODEL_NAME    = "pritamdeka/S-PubMedBert-MS-MARCO"
INDEX_PATH    = os.path.join(_base, "data", "indexes", "mimic_patients.index")
METADATA_PATH = os.path.join(_base, "data", "mimic", "processed", "patient_metadata.csv")
NOTES_DIR     = os.path.join(_base, "data", "mimic", "mimic_sample")
ALPHA         = 0.6   # weight for embedding similarity
BETA          = 0.4   # weight for ICD Jaccard overlap
TOP_K         = 3
# ─────────────────────────────────────────────────────────────────────────────


def normalize_icd(code: str) -> str:
    """Normalize ICD-10 code to first 3 characters (category level)."""
    return str(code).strip()[:3]


def load_resources():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISIONS["pritamdeka/S-PubMedBert-MS-MARCO"])

    print("Loading patient metadata...")
    STRATIFIED_PATH = os.path.join(_base, "data", "mimic", "processed", "patient_metadata_stratified.csv")
    meta_path = STRATIFIED_PATH if os.path.exists(STRATIFIED_PATH) else METADATA_PATH
    meta = pd.read_csv(meta_path, dtype=str)

    # Build normalized ICD sets (3-char prefixes, strings only)
    def parse_icd_set(raw):
        codes = set(str(raw).split(",")) - {"", "nan"}
        normalized = {normalize_icd(c) for c in codes}
        assert all(isinstance(c, str) for c in normalized), \
            "ICD set contains non-string values!"
        return normalized

    meta["icd_set"] = meta["icd_codes_top5"].fillna("").apply(parse_icd_set)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("All resources loaded.\n")
    return model, meta, index


def jaccard(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity between two ICD code sets.
    Both sets must contain only strings normalized to 3-char prefixes.
    """
    assert all(isinstance(c, str) for c in set_a), "set_a contains non-strings"
    assert all(isinstance(c, str) for c in set_b), "set_b contains non-strings"

    # Normalize both sets to 3-char prefixes
    a = {normalize_icd(c) for c in set_a}
    b = {normalize_icd(c) for c in set_b}

    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def retrieve(query_text: str, query_icd: set,
             model, meta, index, top_k: int = TOP_K):
    """Retrieve top-k similar patients given query text and ICD hints."""

    # Normalize query ICD codes to 3-char prefixes
    query_icd_norm = {normalize_icd(c) for c in query_icd}

    # Embed query
    q_vec = model.encode(
        [query_text], normalize_embeddings=True
    ).astype("float32")

    # FAISS cosine search
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(q_vec, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        row      = meta.iloc[idx]
        cos_sim  = float(dist)
        icd_sim  = jaccard(query_icd_norm, row["icd_set"])
        combined = ALPHA * cos_sim + BETA * icd_sim
        results.append({
            "rank_score"      : combined,
            "cos_sim"         : cos_sim,
            "icd_jaccard"     : icd_sim,
            "subject_id"      : row["subject_id"],
            "hadm_id"         : row["hadm_id"],
            "age"             : row["age"],
            "gender"          : row["gender"],
            "admission_type"  : row["admission_type"],
            "icd_codes"       : row["icd_codes_top5"],
            "discharge_location": row.get("discharge_location", ""),
            "bhc_preview"     : str(row.get("bhc", "") or "")[:200],
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:top_k]


def build_patient_index(model, meta):
    """
    Embed one representative text per patient and save FAISS index.
    Priority: bhc field (Brief Hospital Course, up to 800 chars)
    Fallback: first 500 chars of raw .txt note file
    Final fallback: demographic string
    """
    print("Building patient FAISS index...")
    texts = []
    used_bhc      = 0
    used_note     = 0
    used_fallback = 0

    note_files = sorted([
        f for f in os.listdir(NOTES_DIR)
        if f.endswith(".txt")
    ])
    note_map = {f: os.path.join(NOTES_DIR, f) for f in note_files}

    for i, row in meta.iterrows():
        bhc = str(row.get("bhc", "")).strip()

        if bhc and bhc != "nan":
            # Primary: use BHC field
            texts.append(bhc)
            used_bhc += 1
        else:
            # Fallback 1: try to find a note file for this subject
            subject_id = str(row["subject_id"]).strip()
            matched = [f for f in note_files if subject_id in f]
            if matched:
                note_path = note_map[matched[0]]
                with open(note_path, "r", encoding="utf-8") as f:
                    texts.append(f.read()[:500])
                used_note += 1
            else:
                # Fallback 2: demographic string
                texts.append(
                    f"Patient age {row.get('age','')} "
                    f"gender {row.get('gender','')} "
                    f"admission {row.get('admission_type','')} "
                    f"ICD {row.get('icd_codes_top5','')}"
                )
                used_fallback += 1

    print(f"  Embedding sources:")
    print(f"    BHC field      : {used_bhc}")
    print(f"    Note file      : {used_note}")
    print(f"    Demographic str: {used_fallback}")
    print(f"  Total            : {len(texts)}")

    print(f"\nEmbedding {len(texts)} patient records...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Patient index saved → {INDEX_PATH} ({index.ntotal} vectors)")
    return index


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Rebuilding mimic_patients.index with S-PubMedBert + BHC embeddings...")
    tmp_model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISIONS["pritamdeka/S-PubMedBert-MS-MARCO"])

    # Load ONLY patients whose notes exist in mimic_sample/
    # This gives us the ~637 stratified corpus patients, not all 431,231
    note_files = [f for f in os.listdir(NOTES_DIR) if f.endswith(".txt")]
    note_hadm_ids = set()
    for fname in note_files:
        # filename format: note_{note_id}.txt
        # we need to map note_id → hadm_id via discharge.csv
        pass

    # Simpler: load full metadata but filter to rows that have bhc content
    # (bhc was only populated for our 637 stratified notes)
    tmp_meta = pd.read_csv(METADATA_PATH, dtype=str)
    tmp_meta["icd_set"] = tmp_meta["icd_codes_top5"].fillna("").apply(
        lambda x: {normalize_icd(c)
                   for c in (set(str(x).split(",")) - {"", "nan"})}
    )

    # Filter to only patients with BHC content OR a matching note file
    has_bhc = tmp_meta["bhc"].fillna("").str.strip().str.len() > 0

    # Also find hadm_ids from note filenames via discharge mapping
    import pandas as _pd
    discharge_path = os.path.join(_base, "data", "mimic", "processed", "discharge.csv")
    discharge_map = _pd.read_csv(
        discharge_path, dtype=str, low_memory=False,
        usecols=["note_id", "hadm_id"]
    )
    note_ids_in_sample = [
        f.replace("note_", "").replace(".txt", "")
        for f in note_files
    ]
    hadm_ids_in_sample = set(
        discharge_map[discharge_map["note_id"].isin(note_ids_in_sample)]["hadm_id"]
    )

    has_note = tmp_meta["hadm_id"].isin(hadm_ids_in_sample)
    tmp_meta_filtered = tmp_meta[has_bhc | has_note].reset_index(drop=True)
    print(f"Filtered to {len(tmp_meta_filtered)} patients with notes/BHC content")

    build_patient_index(tmp_model, tmp_meta_filtered)

    # Save filtered metadata for retrieval use
    FILTERED_META_PATH = os.path.join(
        _base, "data", "mimic", "processed", "patient_metadata_stratified.csv"
    )
    tmp_meta_filtered.to_csv(FILTERED_META_PATH, index=False)
    print(f"✅ Filtered metadata saved → {FILTERED_META_PATH}")
    print()

    # Load and run 5 test queries using filtered metadata
    print("Loading resources for test queries...")
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISIONS["pritamdeka/S-PubMedBert-MS-MARCO"])
    meta  = pd.read_csv(FILTERED_META_PATH, dtype=str)
    meta["icd_set"] = meta["icd_codes_top5"].fillna("").apply(
        lambda x: {normalize_icd(c)
                   for c in (set(str(x).split(",")) - {"", "nan"})}
    )
    index = faiss.read_index(INDEX_PATH)

    test_queries = [
        ("Patient with type 2 diabetes and hyperglycaemia requiring insulin.",
         {"E11"}),
        ("Elderly patient with acute myocardial infarction and chest pain.",
         {"I21", "I50"}),
        ("Patient with COPD exacerbation and respiratory failure.",
         {"J44", "J96"}),
        ("Patient admitted with pneumonia and sepsis requiring ICU care.",
         {"J18", "A41"}),
        ("Patient with chronic kidney disease and fluid overload.",
         {"N18", "N19"}),
    ]

    for i, (query_text, query_icd) in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query_text}")
        print(f"Query ICD hints: {query_icd}")
        print(f"{'='*60}")
        results = retrieve(query_text, query_icd, model, meta, index)
        for rank, r in enumerate(results, 1):
            print(f"\n  Rank {rank} | Score: {r['rank_score']:.4f} "
                  f"(cos={r['cos_sim']:.4f}, icd={r['icd_jaccard']:.4f})")
            print(f"  Subject: {r['subject_id']} | Age: {r['age']} | "
                  f"Gender: {r['gender']}")
            print(f"  ICD codes: {r['icd_codes']}")
            print(f"  Discharge: {r['discharge_location']}")
            print(f"  BHC: {r['bhc_preview']}...")