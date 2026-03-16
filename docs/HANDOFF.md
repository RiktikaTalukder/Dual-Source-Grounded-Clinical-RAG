## Week 2 - Farhana (16/03/2026)

Environment verification completed on Windows.

- Conda environment created successfully: `dual-source-rag`
- Python version: `3.10.20`
- Active Python path: `C:\Users\fakht\miniconda3\envs\dual-source-rag\python.exe`
- Core packages installed successfully
- Verification command passed

### Verification outputs
- `python -c "import torch, transformers, langchain, faiss, sentence_transformers; print('All core packages ready!')"` → passed
- `torch.__version__` → `2.10.0+cpu`
- `faiss` import → passed

### Repository / setup status
- Collaborator access accepted
- Repository cloned successfully
- `docs/SETUP.md` followed and verified on local machine
- Local setup notes updated where needed for smoother Windows reproducibility

### PubMedQA setup
- Created `data/pubmedqa/raw`
- Staged PubMedQA files locally:
  - `ori_pqaa.json`
  - `ori_pqal.json`
  - `ori_pqau.json`
- Implemented `src/load_pubmedqa.py`
- Loader currently supports all three local PubMedQA files through:
  - `load_pubmedqa_file(...)`
  - `load_pubmedqa_all()`
- Each loaded sample includes:
  - `id`
  - `split`
  - `question`
  - `contexts`
  - `label`
  - `long_answer`
- Verification added in script main block:
  - checks each file exists
  - prints per-file sample counts
  - prints total combined sample count
  - prints one example loaded item
- Verified local loading successfully

### WandB setup
- WandB local setup tested with a dummy run
- `src/test_wandb.py` used earlier for verification
- Teammate should create a WandB account and send me the email used for the account so I can add them to the team/workspace

### MIMIC-IV-Note local staging
- Access to MIMIC-IV-Note obtained
- Relevant local files downloaded:
  - `discharge.csv.gz`
  - `discharge_detail.csv.gz`
- These are the current priority files for discharge-note-based work
- Raw restricted files are local only and are not committed to GitHub
- No preprocessing done yet

### PMC preparation
- Added initial helper script: `src/download_pmc.py`
- Current script prepares the local PMC folder structure and writes a small sample manifest for later download/testing
- Full PMC OA bulk download and parsing are not completed yet

### Git / data handling policy
- Updated `.gitignore` to exclude:
  - `data/`
  - `wandb/`
  - experiment outputs and large archives
- Raw datasets are kept local only
- Restricted MIMIC files must not be uploaded or redistributed through GitHub

### Current status at handoff
Completed:
- local environment verification
- PubMedQA local staging and loader implementation
- WandB test setup
- MIMIC discharge-note raw file staging
- PMC local structure preparation
- repository cleanup / ignore rules

Not completed yet:
- full PMC OA download and parsing
- MIMIC preprocessing
- schema inspection and preprocessing scripts for restricted datasets

### Ready for next step
Week 3 can continue with:
- MIMIC schema/column inspection
- discharge note preprocessing
- patient/admission linkage planning
- literature corpus setup for retrieval