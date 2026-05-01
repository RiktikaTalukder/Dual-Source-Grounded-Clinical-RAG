## Week 6 Handoff — Farhana (M2) — 01 May 2026

### What I did this week:
- Reviewed 02_chunking_comparison.ipynb and added observations
- Fixed PMC article quality by tightening clinical keyword filter in pmc_parser.py
  (now requires 2+ keyword matches instead of any 1)
- Switched embedding model from BiomedNLP-PubMedBERT (not fine-tuned for retrieval)
  to pritamdeka/S-PubMedBert-MS-MARCO (fine-tuned for semantic search)
- Rebuilt FAISS index with improved clinical article set
- Wrote evaluate_recall.py and ran evaluation on 50 PubMedQA questions

### Results:
- Recall@5  = 0.8722
- Recall@10 = 0.8946

### What these scores suggest:
- Recall@5 of 0.87 means the retriever finds relevant evidence in the top 5
  results for ~87% of clinical questions. The small gap between Recall@5 and
  Recall@10 (only 0.02) indicates relevant articles are ranking near the top,
  not buried lower in the list. This is a strong single-source PMC baseline.
  Future phases should test whether adding MIMIC-IV patient cases (dual-source)
  pushes these numbers higher and improves answer grounding.

### How to continue next week (Riktika):
- IMPORTANT: pmc_articles.index is not committed (binary file, too large for Git).
  Run `python src/pmc_embedder.py` first to rebuild the index before anything else.
- After rebuilding, run: python src/pmc_retriever.py to verify retrieval works
- Embedding model in use: pritamdeka/S-PubMedBert-MS-MARCO
- FAISS index: data/indexes/pmc_articles.index
- Recall baseline results: results/recall_baseline.csv
- Both pmc_embedder.py and pmc_retriever.py must use the same model name

### What is incomplete:
- Recall evaluation uses lexical overlap (30% word overlap threshold) as a 
  proxy for relevance — not semantic similarity. This is a known limitation
  and can be refined in later phases using BERTScore.