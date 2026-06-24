# PubMedQA Local Staging (Week 2)

## Local folder structure
data/pubmedqa/
└── raw/
    ├── ori_pqaa.json
    ├── ori_pqal.json
    └── ori_pqau.json

## Notes
- These three JSON files are staged locally for PubMedQA loading and later preprocessing.
- The loader script is `src/load_pubmedqa.py`.
- The loader currently reads all three files and combines them into one in-memory dataset.
- Each loaded sample includes:
  - `id`
  - `split`
  - `question`
  - `contexts`
  - `label`
  - `long_answer`

## Verification
Run:
python src/load_pubmedqa.py

Expected behavior:
- prints sample counts for pqaa, pqal, and pqau
- prints total sample count
- prints one example item

## Git policy
- Raw dataset files should remain local unless the team explicitly decides otherwise.
- Large data files should not be committed accidentally.