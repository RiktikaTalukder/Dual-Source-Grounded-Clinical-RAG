"""
extract_bhc.py
Week 12 — Phase A Data Rebuild

Reads each .txt discharge note in data/mimic/mimic_sample/,
extracts the 'Brief Hospital Course:' section (up to 800 chars),
and adds a 'bhc' column to patient_metadata.csv.

Also rebuilds patient_metadata.csv with:
  - icd_codes_top5: ICD-10 only, zero-padded strings
  - discharge_location: from admissions.csv
  - bhc: extracted Brief Hospital Course text
"""

import os
import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_base        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR     = os.path.join(_base, "data", "mimic", "processed")
SAMPLE_DIR   = os.path.join(_base, "data", "mimic", "mimic_sample")
DIAG_PATH    = os.path.join(PROC_DIR, "diagnoses_icd.csv")
ADMISSIONS_PATH = os.path.join(PROC_DIR, "admissions.csv")
PATIENTS_PATH   = os.path.join(PROC_DIR, "patients.csv")
DISCHARGE_PATH  = os.path.join(PROC_DIR, "discharge.csv")
OUT_PATH     = os.path.join(PROC_DIR, "patient_metadata.csv")

BHC_MAX_CHARS = 800

# ── Step 1: Rebuild patient_metadata.csv from scratch ────────────────────────
print("=" * 60)
print("STEP 1: Rebuilding patient_metadata.csv")
print("=" * 60)

# Load tables
print("Loading admissions.csv ...")
admissions = pd.read_csv(ADMISSIONS_PATH, dtype=str)

print("Loading patients.csv ...")
patients = pd.read_csv(PATIENTS_PATH, dtype=str)

print("Loading diagnoses_icd.csv (ICD-10 only) ...")
diag = pd.read_csv(DIAG_PATH, dtype=str)

# Filter ICD-10 only
diag = diag[diag["icd_version"].str.strip() == "10"].copy()

# Sort by seq_num and take top 5 per admission
diag["seq_num"] = diag["seq_num"].astype(int)
diag = diag.sort_values(["hadm_id", "seq_num"])

# Store codes as zero-padded strings (never integers)
diag["icd_code"] = diag["icd_code"].astype(str).str.strip()

top5 = (
    diag.groupby("hadm_id")["icd_code"]
    .apply(lambda x: ",".join(list(x)[:5]))
    .reset_index()
    .rename(columns={"icd_code": "icd_codes_top5"})
)

# Build metadata: admissions + patients + ICD top5
print("Joining tables ...")
meta = admissions[["subject_id", "hadm_id", "admission_type", "discharge_location"]].copy()
meta = meta.merge(
    patients[["subject_id", "gender", "anchor_age"]].rename(
        columns={"anchor_age": "age"}),
    on="subject_id", how="left"
)
meta = meta.merge(top5, on="hadm_id", how="left")
meta["icd_codes_top5"] = meta["icd_codes_top5"].fillna("")

print(f"  Metadata rows: {len(meta):,}")
print(f"  Columns: {list(meta.columns)}")

# ── Step 2: Extract BHC from .txt files ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Extracting Brief Hospital Course sections")
print("=" * 60)

def extract_bhc(text: str) -> str:
    """
    Extract text between 'Brief Hospital Course:' and the next
    major heading 'Medications on Admission:'.
    Returns up to BHC_MAX_CHARS characters.
    If not found, returns empty string.
    """
    # Case-insensitive search for the BHC heading
    pattern = re.compile(
        r'brief hospital course[:\s]*(.*?)(?=medications on admission|discharge medications|$)',
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        bhc_text = match.group(1).strip()
        # Collapse extra whitespace
        bhc_text = re.sub(r'\s+', ' ', bhc_text)
        return bhc_text[:BHC_MAX_CHARS]
    return ""

# Get all .txt files in mimic_sample
txt_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt")]
print(f"Found {len(txt_files)} .txt note files in mimic_sample/")

# Build a mapping: note_id → bhc text
# Also build note_id → hadm_id from discharge.csv
print("Loading discharge.csv to map note_id → hadm_id ...")
discharge = pd.read_csv(DISCHARGE_PATH, dtype=str, low_memory=False,
                        usecols=["note_id", "hadm_id"])
discharge["note_id"] = discharge["note_id"].astype(str).str.strip()
discharge["hadm_id"] = discharge["hadm_id"].astype(str).str.strip()
note_to_hadm = dict(zip(discharge["note_id"], discharge["hadm_id"]))

# Extract BHC for each file
bhc_records = []
found_count = 0
not_found_count = 0

for fname in txt_files:
    # Extract note_id from filename: "note_{note_id}.txt"
    note_id = fname.replace("note_", "").replace(".txt", "").strip()
    hadm_id = note_to_hadm.get(note_id, None)

    fpath = os.path.join(SAMPLE_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    bhc = extract_bhc(text)
    if bhc:
        found_count += 1
    else:
        not_found_count += 1

    bhc_records.append({
        "note_id": note_id,
        "hadm_id": hadm_id,
        "bhc": bhc
    })

bhc_df = pd.DataFrame(bhc_records)
bhc_df["hadm_id"] = bhc_df["hadm_id"].astype(str).str.strip()

print(f"  BHC found     : {found_count}")
print(f"  BHC not found : {not_found_count}")
success_rate = found_count / len(txt_files) * 100
print(f"  Success rate  : {success_rate:.1f}%")

# ── Step 3: Merge BHC into metadata ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Merging BHC into patient_metadata.csv")
print("=" * 60)

# Aggregate BHC per hadm_id (take first non-empty)
bhc_per_hadm = (
    bhc_df[bhc_df["bhc"] != ""]
    .groupby("hadm_id")["bhc"]
    .first()
    .reset_index()
)

meta["hadm_id"] = meta["hadm_id"].astype(str).str.strip()
meta = meta.merge(bhc_per_hadm, on="hadm_id", how="left")
meta["bhc"] = meta["bhc"].fillna("")

# Keep only columns we need, in correct order
meta = meta[[
    "subject_id", "hadm_id", "age", "gender",
    "admission_type", "discharge_location",
    "icd_codes_top5", "bhc"
]]

# Save
meta.to_csv(OUT_PATH, index=False)
print(f"✅ patient_metadata.csv saved: {len(meta):,} rows")
print(f"   Columns: {list(meta.columns)}")
print(f"   Path: {OUT_PATH}")

# ── Step 4: Verify on sample rows ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Sample verification")
print("=" * 60)
sample = meta[meta["bhc"] != ""].head(3)
for _, row in sample.iterrows():
    print(f"\n  hadm_id: {row['hadm_id']}")
    print(f"  ICD codes: {row['icd_codes_top5']}")
    print(f"  Discharge location: {row['discharge_location']}")
    print(f"  BHC preview: {row['bhc'][:150]}...")

print("\n✅ extract_bhc.py complete!")