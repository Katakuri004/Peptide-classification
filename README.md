# Frog code – antibacterial peptide prediction

Pipeline for training and submitting an ensemble meta-model for antibacterial peptide (ABP) classification.

## Canonical pipeline (run in order)

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_eda.ipynb` | EDA and data prep |
| 2 | `02_features.ipynb` | Feature engineering |
| 4 | `04_cluster_splits.ipynb` | Cluster-based train/val splits |
| 5 | `05_kmer_tfidf.ipynb` | k-mer TF-IDF + TF-IDF LogReg/SVM OOF |
| 6 | `06_plm_embeddings.ipynb` | ESM2 embeddings + ESM LogReg/MLP OOF |
| 7 | `07_tuning.ipynb` | Tune LGBM and XGBoost |
| 8 | `08_ensemble.ipynb` | Stack OOF → meta-model (4 or 6 features), save `final_meta_model.pkl` |
| 9 | `09_calibrate_submit.ipynb` | Calibrate (isotonic), build test stack, predict, write submission CSV |

- **With ESM**: When `data/processed/oof_esm.csv` and ESM arrays exist, the stack uses **6 base models** (TF-IDF LR, TF-IDF SVM, LGBM, XGBoost, ESM LR, ESM MLP).  
- **Without ESM**: Stack uses **4 base models** (TF-IDF LR, TF-IDF SVM, LGBM, XGBoost).

The **canonical submission** is the one produced in `09_calibrate_submit.ipynb`: meta-model predictions on the 4- or 6-feature test stack, then isotonic calibration, then CSV export.

## Data (key artifacts)

- `data/processed/`: OOF CSVs (`oof_tfidf.csv`, `oof_tuned_trees.csv`, `oof_esm.csv`), ESM arrays (`X_train_esm.npy`, `X_test_esm.npy`), `final_meta_model.pkl`, `train_clusters.csv`, etc.

## Other notebooks

- `11_antibp2_features.ipynb`: AntiBP2-style features (optional).
- `12_finetune_esm.ipynb`: ESM fine-tuning (optional).
- `notebooks/archive/`: Older or superseded notebooks (e.g. baseline, ablation, evaluation).

## Docs

- `docs/submission_0.86_analysis_and_improvements.md`: Analysis of CV vs leaderboard gap and improvement ideas.
