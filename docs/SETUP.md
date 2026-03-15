# Environment Setup Guide (Farhana — read this first!)

## 1. Clone the repo
git clone https://github.com/RiktikaTalukder/Dual-Source-Grounded-Clinical-RAG.git
cd Dual-Source-Grounded-Clinical-RAG

## 2. Create & activate environment (exactly how we fixed it)
conda create -n dual-source-rag python=3.10 -c conda-forge --yes
conda activate dual-source-rag

conda install faiss-cpu numpy pandas scikit-learn matplotlib seaborn -c conda-forge --yes

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers langchain langchain-community wandb huggingface_hub datasets bert-score

## 3. Verify
python -c "import torch, transformers, langchain, faiss, sentence_transformers; print('✅ All core packages ready!')"

Report any issues in /docs/HANDOFF.md
