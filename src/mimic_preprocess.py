import pandas as pd
import os
import re
from tqdm import tqdm

DATA_DIR = "data"
SAMPLE_DIR = f"{DATA_DIR}/mimic_sample"
os.makedirs(SAMPLE_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """Extra safety cleaning (MIMIC is already de-identified)"""
    text = re.sub(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\bMRN\s*[:#]?\s*\d+\b', '[MRN]', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading tables...")
discharge = pd.read_csv(f"{DATA_DIR}/discharge.csv.gz", low_memory=False)

# === 1. Create 200 cleaned sample notes ===
print("Creating 200 cleaned sample notes...")
sample_df = discharge.sample(n=200, random_state=42).reset_index(drop=True)

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    clean_note = clean_text(row['text'])
    filename = f"{SAMPLE_DIR}/note_{row['note_id']}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(clean_note)
    # Save metadata
    meta = row[['subject_id', 'hadm_id', 'note_id']].to_dict()
    pd.DataFrame([meta]).to_json(f"{SAMPLE_DIR}/note_{row['note_id']}_meta.json", orient="records")

print(f"✅ Saved 200 cleaned notes to {SAMPLE_DIR}/")

# === 2. Build patient_metadata.csv ===
print("Building patient_metadata.csv...")
patients = pd.read_csv(f"{DATA_DIR}/patients.csv.gz", low_memory=False)
admissions = pd.read_csv(f"{DATA_DIR}/admissions.csv.gz", low_memory=False)
diagnoses = pd.read_csv(f"{DATA_DIR}/diagnoses_icd.csv.gz", low_memory=False)

top_icd = (diagnoses.sort_values('seq_num')
           .groupby(['subject_id', 'hadm_id'])['icd_code']
           .apply(lambda x: ','.join(x.head(5)))
           .reset_index())

meta = (admissions[['subject_id', 'hadm_id', 'admission_type']]
        .merge(patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')
        .merge(top_icd, on=['subject_id', 'hadm_id'], how='left'))

meta = meta.rename(columns={'anchor_age': 'age'})
meta['icd_codes_top5'] = meta['icd_code'].fillna('')
meta = meta[['subject_id', 'age', 'gender', 'admission_type', 'icd_codes_top5']]
meta.to_csv(f"{DATA_DIR}/patient_metadata.csv", index=False)

print(f"✅ patient_metadata.csv created with {len(meta):,} rows")
print(meta.head())
