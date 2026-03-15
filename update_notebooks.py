import nbformat as nbf
import os

def update_07():
    nb = nbf.read('notebooks/07_tuning.ipynb', as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 1. Add ESM Embeddings to features
            if "X_train = pd.read_pickle('../data/processed/X_train_features.pkl')" in cell.source:
                if 'X_train_esm.npy' not in cell.source:
                    addition = """
esm_train = np.load('../data/processed/X_train_esm.npy')
X_train_features = pd.read_pickle('../data/processed/X_train_features.pkl')

print(f'Handcrafted shape: {X_train_features.shape}')
print(f'ESM shape: {esm_train.shape}')

if len(X_train_features) == len(esm_train):
    X_train_combined = np.hstack([X_train_features.values, esm_train])
    X_train = pd.DataFrame(X_train_combined)
else:
    print('Warning: Length mismatch, using only handcrafted features.')
    X_train = X_train_features
"""
                    cell.source = cell.source.replace(
                        "X_train = pd.read_pickle('../data/processed/X_train_features.pkl')",
                        addition
                    )
            
            # 2. Add reg_lambda and min_split_gain
            if "def objective_lgb(trial):" in cell.source:
                if "'reg_lambda'" not in cell.source:
                    cell.source = cell.source.replace(
                        "'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),",
                        "'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),\n        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),\n        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),"
                    )
            
            # 3. Increase optuna trials and add TPESampler
            if "study_lgb.optimize(objective_lgb, n_trials=30" in cell.source:
                cell.source = cell.source.replace(
                    "study_lgb = optuna.create_study(direction='maximize')",
                    "from optuna.samplers import TPESampler\nstudy_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(n_startup_trials=20, seed=42))"
                )
                cell.source = cell.source.replace("n_trials=30", "n_trials=100")
                
            if "study_xgb.optimize(objective_xgb, n_trials=30" in cell.source:
                cell.source = cell.source.replace(
                    "study_xgb = optuna.create_study(direction='maximize')",
                    "study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(n_startup_trials=20, seed=42))"
                )
                cell.source = cell.source.replace("n_trials=30", "n_trials=100")

    with open('notebooks/07_tuning.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Updated 07_tuning.ipynb")

def update_08():
    nb = nbf.read('notebooks/08_ensemble.ipynb', as_version=4)
    esm_block = """
# 3. ESM PLM Meta (Phase 6)
if os.path.exists('../data/processed/oof_esm.csv'):
    df_esm = pd.read_csv('../data/processed/oof_esm.csv')
    oof_dfs.append(df_esm[['esm_lr_pred', 'esm_mlp_pred']])
    model_aucs['ESM2 LogReg'] = roc_auc_score(y_true, df_esm['esm_lr_pred'])
    model_aucs['ESM2 MLP']    = roc_auc_score(y_true, df_esm['esm_mlp_pred'])
    print('Loaded ESM OOFs.')
"""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'oof_dfs = []' in cell.source:
            if 'oof_esm.csv' not in cell.source:
                cell.source = cell.source.replace('# Combine all', esm_block + '\n# Combine all')

    with open('notebooks/08_ensemble.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Updated 08_ensemble.ipynb")

if __name__ == '__main__':
    update_07()
    update_08()
