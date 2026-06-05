# Environment Setup Guide (Farhana — read this first! 15/03/2026)

## 1\. Clone the repo

git clone https://github.com/RiktikaTalukder/Dual-Source-Grounded-Clinical-RAG.git
cd Dual-Source-Grounded-Clinical-RAG

## 2\. Create \& activate environment (exactly how we fixed it)

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2


conda create -n dual-source-rag python=3.10 -c conda-forge --yes
conda activate dual-source-rag

conda install faiss-cpu numpy pandas scikit-learn matplotlib seaborn -c conda-forge --yes

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers langchain langchain-community wandb huggingface\_hub datasets bert-score

## 3\. Verify

python -c "import torch, transformers, langchain, faiss, sentence\_transformers; print('✅ All core packages ready!')"

Report any issues in /docs/HANDOFF.md

## 4. Index Files

Two FAISS index files exist in `data/indexes/`. They serve different purposes.
Do NOT confuse them — only one is used by the live pipeline.

### mimic_patients.index — PRODUCTION INDEX
- Built from the stratified 938-patient MIMIC corpus (ICD-10 topics, BHC field embeddings)
- Embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Used by: `src/pipeline.py`, `src/patient_retriever.py`, `src/generator.py`
- **This is the index the live system queries at runtime**
- Gitignored — must be rebuilt locally. Rebuild command:
`python src/patient_retriever.py`

### mimic_chunks.index — DEMONSTRATION ONLY
- Built from 500 random notes for chunking strategy comparison
- Used by: `notebooks/02_chunking_comparison.ipynb` ONLY
- NOT used by the live pipeline at any point
- Gitignored — rebuild command (only needed if running notebook 02):
`python src/build_mimic_index.py`

### pmc_articles.index — LITERATURE INDEX
- Built from 499 filtered clinical PMC articles
- Embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Used by: `src/pmc_retriever.py`, `src/pipeline.py`
- Gitignored — must be rebuilt locally. Rebuild command:
`python src/pmc_embedder.py`

### Rebuild order (fresh clone)
Run these two commands in order before using the pipeline:
`python src/patient_retriever.py`
`python src/pmc_embedder.py`

**Note:** MIMIC data files (`data/mimic/`) are never committed to Git (restricted/credentialed data).
You must have MIMIC-IV v2.2 downloaded locally from PhysioNet before rebuilding the patient index.