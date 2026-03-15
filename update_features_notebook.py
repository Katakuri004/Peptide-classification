import nbformat as nbf
import os

def update_02_features():
    nb = nbf.read('notebooks/02_features.ipynb', as_version=4)
    
    # Check if we already injected it
    already_injected = any('iFeatureOmegaCLI' in cell.source for cell in nb.cells)
    if already_injected:
        print("iFeatureOmegaCLI already integrated in 02_features.ipynb. Skipping...")
        return
        
    markdown_cell = nbf.v4.new_markdown_cell("""\
## Step 2.X — Advanced Physicochemical Features
Using `iFeatureOmegaCLI` to generate CTD (Composition, Transition, Distribution) and PseAAC features.
(Assumes `pip install iFeatureOmegaCLI` is installed)
""")

    code_cell = nbf.v4.new_code_cell("""\
# If not installed, uncomment:
# !pip install iFeatureOmegaCLI

import os
from iFeatureOmegaCLI import iProtein
import pandas as pd

# We need to temporarily write sequences to a FASTA file for iProtein to consume
def sequences_to_fasta(sequences, labels, filepath):
    with open(filepath, 'w') as f:
        for i, (seq, label) in enumerate(zip(sequences, labels)):
            f.write(f'>Seq_{i}|{label}\\n{seq}\\n')

train_df = pd.read_csv('../data/processed/train_clean.csv')
fasta_path = '../data/processed/temp_train.fasta'
sequences_to_fasta(train_df['Sequence'].values, train_df['Label'].values, fasta_path)

# Initialize iProtein
obj = iProtein(file=fasta_path, para=None)

# Compute CTD (Composition, Transition, Distribution) -> 147 features
obj.get_descriptor('CTD')
ctd_features = obj.encodings
print(f'CTD features shape: {ctd_features.shape}')

# Compute PseAAC (Pseudo Amino Acid Composition)
try:
    obj.get_descriptor('PseAAC')
    pseaac_features = obj.encodings
    print(f'PseAAC features shape: {pseaac_features.shape}')
except Exception as e:
    print(f"Warning: Could not compute PseAAC, error: {e}")

# Clean up temp
if os.path.exists(fasta_path):
    os.remove(fasta_path)
    
print('Ready to concatenate these features to X_train_features!')
""")
    
    # Insert at the end before storing
    nb.cells.extend([markdown_cell, code_cell])
    
    with open('notebooks/02_features.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Updated notebooks/02_features.ipynb with iFeatureOmega blocks.")

if __name__ == "__main__":
    update_02_features()
