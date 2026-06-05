"""
build_stratified_corpus.py
Week 12 — Phase A Data Rebuild

Reads diagnoses_icd_combined.csv, filters to ICD-10 codes only,
samples up to 100 admissions per clinical topic (700 total),
extracts discharge notes from discharge.csv,
saves one .txt file per admission to data/mimic/mimic_sample/
"""

import os
import re
import pandas as pd
import random

random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
_base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(_base, "data", "mimic", "processed")
SAMPLE_DIR   = os.path.join(_base, "data", "mimic", "mimic_sample")
DIAG_PATH    = os.path.join(PROC_DIR, "diagnoses_icd.csv")
DISCHARGE_PATH = os.path.join(PROC_DIR, "discharge.csv")
ADMISSIONS_PATH = os.path.join(PROC_DIR, "admissions.csv")

os.makedirs(SAMPLE_DIR, exist_ok=True)

# ── 7 Clinical Topic Clusters (ICD-10 prefixes only) ─────────────────────────
TOPIC_ICD_PREFIXES = {
    "cardiovascular": ["I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"],
    "diabetes":       ["E10", "E11", "E12", "E13", "E14"],
    "respiratory":    ["J0", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"],
    "neurological":   ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"],
    "infections":     ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
                       "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
    "oncology":       ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                       "D0", "D1", "D2", "D3", "D4"],
    "renal":          ["N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07",
                       "N08", "N09", "N10", "N11", "N12", "N13", "N14", "N15",
                       "N16", "N17", "N18", "N19", "N20", "N21", "N22", "N23",
                       "N24", "N25", "N26", "N27", "N28", "N29", "N30", "N31",
                       "N32", "N33", "N34", "N35", "N36", "N37", "N38", "N39"],
}

MAX_PER_TOPIC = 100

def matches_topic(icd_code: str, prefixes: list) -> bool:
    """Check if an ICD-10 code starts with any of the topic prefixes."""
    code = str(icd_code).strip()
    return any(code.startswith(p) for p in prefixes)

def clean_text(text: str) -> str:
    """Basic PHI safety cleaning (MIMIC is pre-deidentified by PhysioNet)."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Step 1: Load diagnoses (ICD-10 only) ─────────────────────────────────────
print("Loading diagnoses_icd_combined.csv ...")
diag = pd.read_csv(DIAG_PATH, dtype=str)
print(f"  Total rows loaded: {len(diag):,}")

# Keep only ICD-10
diag = diag[diag["icd_version"].astype(str).str.strip() == "10"].copy()
print(f"  After ICD-10 filter: {len(diag):,} rows")

# ── Step 2: Assign each hadm_id to topic groups ───────────────────────────────
print("\nAssigning admissions to clinical topics ...")
topic_hadm = {topic: set() for topic in TOPIC_ICD_PREFIXES}

for _, row in diag.iterrows():
    code = str(row["icd_code"]).strip()
    hadm = str(row["hadm_id"]).strip()
    for topic, prefixes in TOPIC_ICD_PREFIXES.items():
        if matches_topic(code, prefixes):
            topic_hadm[topic].add(hadm)

for topic, hadms in topic_hadm.items():
    print(f"  {topic}: {len(hadms)} admissions found")

# ── Step 3: Sample up to 100 per topic, collect unique hadm_ids ──────────────
print("\nSampling up to 100 admissions per topic ...")
selected_hadm = set()
topic_counts  = {}

for topic, hadms in topic_hadm.items():
    hadm_list = list(hadms)
    random.shuffle(hadm_list)
    chosen = hadm_list[:MAX_PER_TOPIC]
    selected_hadm.update(chosen)
    topic_counts[topic] = len(chosen)
    print(f"  {topic}: selected {len(chosen)}")

print(f"\nTotal unique admissions selected: {len(selected_hadm)}")

# ── Step 4: Load discharge notes and extract selected ones ───────────────────
print("\nLoading discharge.csv (this may take a moment) ...")
discharge = pd.read_csv(DISCHARGE_PATH, dtype=str, low_memory=False)
print(f"  Loaded {len(discharge):,} discharge notes")
print(f"  Columns: {list(discharge.columns)}")

# Filter to selected hadm_ids
discharge["hadm_id"] = discharge["hadm_id"].astype(str).str.strip()
selected_df = discharge[discharge["hadm_id"].isin(selected_hadm)].copy()
print(f"  Matched {len(selected_df):,} notes for selected admissions")

# ── Step 5: Save one .txt file per admission ──────────────────────────────────
print("\nSaving discharge note .txt files ...")
saved = 0
skipped = 0

# Use note_id if available, else hadm_id
id_col = "note_id" if "note_id" in selected_df.columns else "hadm_id"

for _, row in selected_df.iterrows():
    note_text = str(row.get("text", "")).strip()
    if not note_text or note_text == "nan":
        skipped += 1
        continue
    note_text = clean_text(note_text)
    hadm_id   = str(row["hadm_id"]).strip()
    note_id   = str(row[id_col]).strip()
    filename  = os.path.join(SAMPLE_DIR, f"note_{note_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(note_text)
    saved += 1

print(f"\n✅ Done!")
print(f"   Notes saved : {saved}")
print(f"   Skipped (empty): {skipped}")
print(f"   Output folder: {SAMPLE_DIR}")
print(f"\nTopic breakdown:")
for topic, count in topic_counts.items():
    print(f"  {topic}: {count}")