# Why the submission scored 0.86 AUC-ROC (vs ~0.90 CV) and how to improve it

## 1. Why CV was higher than leaderboard

### 1.1 Train vs test distribution gap (main cause)

- **Cluster-based CV** (used in 08_ensemble and in base model OOF) keeps whole sequence clusters in one fold, so it already reduces leakage and gives a more realistic ~0.90 AUC.
- The **test set** is a separate holdout. It can differ from the training distribution in:
  - **Sequence / peptide families**: Test may include families or organisms under-represented in train (e.g. different antibacterial peptide classes).
  - **Length / composition**: Different length or amino-acid composition distributions would hurt handcrafted features (AAC, DPC, stats, k-mers) and TF-IDF.
- The **AntiBP2 reference** (ref-paper.txt) reports a similar drop: ~92% accuracy in 5-fold CV vs **87.55% on an independent dataset** (Table 2). So a ~4–5% drop from CV to held-out test is expected in this domain.

### 1.2 Calibration fit on train only

- Isotonic calibration in `09_calibrate_submit.ipynb` is fit on **train OOF** predictions.
- If test score distribution or difficulty differs from train, the same calibration curve may not transfer, which can slightly hurt AUC or probability quality on the leaderboard.

### 1.3 Ensemble submission uses only 4 base models

- In `09_calibrate_submit.ipynb`, the submission stack uses only: **TF-IDF LogReg, TF-IDF SVM, Tuned LGBM, Tuned XGBoost**.
- In `08_ensemble.ipynb`, when ESM OOF exists, the meta-model is trained on **6** inputs (including ESM2 LogReg and ESM2 MLP). The saved `final_meta_model.pkl` must then be trained on the **same** 4 columns used at submit time; otherwise you get a shape mismatch. If the best CV run used 6 models but the submitted file was built with a 4-model meta, that could also explain part of the gap.

### 1.4 Possible overfitting

- Trees (LGBM, XGBoost) and the meta-learner are tuned and trained on the same cluster splits. Some overfitting to those splits is possible, so a small drop on a truly unseen test set is normal.

---

## 2. Improvement plan (to increase AUC-ROC)

### 2.1 Align meta-model and submission (quick check)

- **Ensure the same inputs at train and test.**  
  If ESM OOF is used in 08, either:
  - Include **ESM test predictions** in the submission stack and keep a 6-feature meta-model, or  
  - Explicitly train and save a **4-feature** meta-model (no ESM) and use that in 09.
- Verify that `final_meta_model.pkl` was trained on exactly the same columns (and in the same order) as `X_test_meta` in 09.

### 2.2 Add / improve sequence-aware and robust features

- **Include ESM (or other PLM) in the submission** if it’s not there: ESM captures evolutionary/structural signal that handcrafted features don’t. In 08, ESM2 LogReg and ESM2 MLP had solid OOF AUC; adding them to the stack can help.
- **Try domain-specific features** from `11_antibp2_features.ipynb` (e.g. N/C-terminal or composition features from AntiBP2) and feed them into the ensemble or as extra columns in the dense feature set.
- **More robust physicochemical features**: e.g. length bins, isoelectric point, instability index, or other stats that generalize across peptide families.

### 2.3 Better generalization to unseen sequences

- **Regularization**: Slightly increase regularization for the meta-learner (e.g. smaller `C` in LogisticRegression) and for tree models (e.g. stronger L2, lower `max_depth` or `num_leaves`) to reduce overfitting to train clusters.
- **Feature selection**: Drop noisy or highly train-specific features (e.g. rare k-mers) to reduce overfitting.
- **Pseudo-labeling or semi-supervised**: Use confident test predictions to iteratively retrain (with care to avoid leakage and overfitting).

### 2.4 Calibration and probability quality

- **Calibrate per-fold or on a holdout**: Fit the isotonic (or Platt) calibrator on a dedicated validation fold or a small holdout instead of the full OOF, so calibration is less tied to the exact train distribution.
- **Cross-validated calibration**: Fit calibrators inside each CV fold and average or use a single calibrator fit on OOF from a different split scheme.

### 2.5 Data and split strategy

- **Analyze train vs test distribution**: Compare sequence length, AAC, and (if possible) family or source between train and test to confirm distribution shift and target features that are more stable.
- **Stronger cluster split**: Use a stricter cluster threshold (e.g. 0.2 instead of 0.3) so that “similar” sequences are never split across train/val, giving a harder but more realistic CV estimate and potentially more robust models.

### 2.6 Model and ensemble diversity

- **Include ESM in the stack** (if not already) for diversity.
- **Try other PLMs or embeddings** (e.g. different ESM sizes or ProtTrans) and add them as extra base models.
- **Blend with a simple average** of base model probabilities and compare to stacking in 08; sometimes a simple average generalizes better than a learned meta-model on small or shifted test sets.

### 2.7 Tuning and validation

- **Tune on cluster-based CV only**: Ensure Optuna (or any tuner) in 07 uses `StratifiedGroupKFold` with cluster labels so that selected hyperparameters are optimized for the same evaluation setting as the final submission.
- **Report cluster CV AUC for the exact 4- or 6-model stack** used in 09 so you have a single number to compare to the leaderboard.

---

## 3. Suggested order of actions

1. **Verify pipeline consistency**: Meta-model trained on same 4 (or 6) features as submission; add ESM to submission if it improves CV.
2. **Add ESM to the submission stack** (if currently omitted) and re-run 09.
3. **Strengthen regularization** (meta-learner and trees) and re-evaluate with cluster CV.
4. **Improve calibration** (e.g. CV-based or holdout calibrator).
5. **Add domain features** (AntiBP2-style) and/or more robust physicochemical features.
6. **Analyze train vs test** (length, composition) and adjust features/splits accordingly.

After each step, compare **cluster CV AUC** to the **leaderboard AUC**; the gap should shrink as the pipeline and features become more consistent and robust to distribution shift.
