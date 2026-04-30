"""
build_mimic_index.py
Loads real MIMIC notes and builds the FAISS chunk index.
Run ONCE — takes a few minutes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
from chunking_baselines import build_faiss_index, save_index

# ── Load discharge notes from the .gz file in the project root ──
discharge_path = "discharge.csv.gz"

print(f"Loading discharge notes from: {discharge_path}")
discharge = pd.read_csv(discharge_path, low_memory=False)
print(f"Loaded {len(discharge):,} total notes")

# Use a sample of 500 notes (full 331K would take hours)
sample_notes = discharge["text"].dropna().sample(500, random_state=42).tolist()
print(f"Using {len(sample_notes)} sampled notes for index building")

# Build with fixed-size chunking strategy
index, chunks = build_faiss_index(sample_notes, strategy="fixed")

# Save to disk
os.makedirs("data/indexes", exist_ok=True)
save_index(
    index, chunks,
    index_path="data/indexes/mimic_chunks.index",
    chunks_path="data/indexes/mimic_chunks_text.csv"
)

print("\n✅ Done! FAISS index built and saved.")
