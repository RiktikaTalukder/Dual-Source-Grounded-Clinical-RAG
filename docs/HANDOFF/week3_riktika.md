## Week 3 - Riktika (30/03/2026)

### Completed

- Explored full MIMIC-IV schema in `notebooks/01_mimic_schema_exploration.ipynb`
- Successfully ran `src/mimic_preprocess.py`
- Created `data/mimic_sample/` with 200 cleaned discharge notes + metadata JSON
- Built `data/patient_metadata.csv`
- Total discharge notes: 331,793

### Outputs

- Preprocessed sample dataset
- Patient metadata CSV
- Schema exploration notebook

### Preprocessing description (Thesis §5.1)

> Discharge notes from MIMIC-IV-Note were loaded from discharge.csv.gz (331,793 notes). Residual PHI patterns were replaced with placeholders. Text was normalized by collapsing multiple whitespaces. A subset of 200 notes was sampled (random_state=42) and saved with metadata. Patient metadata was constructed by joining patients, admissions, and diagnoses tables, extracting top-5 ICD codes per admission.

### Notes

- Jupyter working directory issue fixed using `os.chdir()`

### Next Steps (Week 4 - Farhana)

```bash
jupyter notebook notebooks/01_data_eda.ipynb

