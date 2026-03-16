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

