import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text1 = """\
# Evaluation of Hierarchical Retrieval Framework
This notebook evaluates the performance of the newly implemented Hierarchical Retriever (`src.retrieval`) against baseline ML models (e.g. TF-IDF + Logistic Regression).
"""

code1 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.append('../src')
from retrieval import HierarchicalRetriever
"""

text2 = """\
## 1. Data Loading
We load the `train.csv` which contains labeled peptide sequences for evaluation. We will sample a subset to keep execution fast.
"""

code2 = """\
# Load evaluation data
df_eval = pd.read_csv('../Data/train.csv')
# Rename the column if it has a leading '#'
df_eval = df_eval.rename(columns={'# Sequence': 'Sequence', 'Sequences': 'Sequence'})

# Sample a subset to evaluate on (e.g., 500 records) to keep text distance calculation fast
df_eval = df_eval.sample(n=500, random_state=42).reset_index(drop=True)

# Instantiate the retriever (using our sourced external database)
retriever = HierarchicalRetriever(db_path="../Data/external/peptide_database.csv", similarity_threshold=0.5)

print(f"Loaded {len(df_eval)} labeled sequences for evaluation.")
"""

text3 = """\
## 2. Evaluate Hierarchical Retriever
We will pass the evaluation sequences through the retriever and see which level resolves the query and what the predicted label is.
"""

code3 = """\
import time
start_time = time.time()

levels_used = []
y_true = []
y_pred_retriever = []

for idx, row in df_eval.iterrows():
    seq = row['Sequence']
    true_label = row['Label']
    
    # Map -1 to 0 for consistency with our peptide_database which uses 0 and 1
    if true_label == -1:
        true_label = 0
        
    y_true.append(true_label)
    
    pred_label, level = retriever.retrieve(seq)
    levels_used.append(level)
    y_pred_retriever.append(pred_label)
    
print(f"Retrieval finished in {time.time() - start_time:.2f} seconds.")

# Fallback label handling (if model fallback happens, we predict 0 by default for metric calc)
y_pred_retriever_clean = [1 if p == 1 else 0 for p in y_pred_retriever]

retriever_acc = accuracy_score(y_true, y_pred_retriever_clean)
retriever_f1 = f1_score(y_true, y_pred_retriever_clean)
print(f"Retriever Accuracy: {retriever_acc:.4f}, F1: {retriever_f1:.4f}")
"""

text4 = """\
## 3. Compare with Baseline ML (TF-IDF + Logistic Regression)
We'll train a baseline model on the rest of `train.csv` and evaluate on the same 500 samples.
"""

code4 = """\
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Prepare train data (excluding the 500 used for eval)
train_full = pd.read_csv('../Data/train.csv')
train_full = train_full.rename(columns={'# Sequence': 'Sequence', 'Sequences': 'Sequence'})

train_full['Label'] = train_full['Label'].replace(-1, 0)
# Drop the eval rows
train_baseline = train_full.drop(df_eval.index)

# Character level TF-IDF (k-mers of length 1 to 3)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_baseline_train = vectorizer.fit_transform(train_baseline['Sequence'])
y_baseline_train = train_baseline['Label']

X_baseline_eval = vectorizer.transform(df_eval['Sequence'])

clf = LogisticRegression(random_state=42)
clf.fit(X_baseline_train, y_baseline_train)

y_pred_baseline = clf.predict(X_baseline_eval)

baseline_acc = accuracy_score(y_true, y_pred_baseline)
baseline_f1 = f1_score(y_true, y_pred_baseline)
print(f"Baseline Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")
"""

text5 = """## 4. Visualizations"""

code5 = """\
# 1. Level Usage Distribution
level_counts = pd.Series(levels_used).value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=level_counts.index, y=level_counts.values, palette='viridis')
plt.title('Hierarchical Retriever: Coverage by Level')
plt.ylabel('Number of Sequences')
plt.show()

# 2. Performance Comparison
metrics = ['Accuracy', 'F1-Score']
retriever_scores = [retriever_acc, retriever_f1]
baseline_scores = [baseline_acc, baseline_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, retriever_scores, width, label='Hierarchical Retriever', color='royalblue')
rects2 = ax.bar(x + width/2, baseline_scores, width, label='TF-IDF + LogReg', color='darkorange')

ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: Retriever vs Baseline')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

fig.tight_layout()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5)
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/evaluation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated successfully at notebooks/evaluation.ipynb")
