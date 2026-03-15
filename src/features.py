"""
Peptide feature extraction utilities.

Reusable functions for computing:
- Amino Acid Composition (AAC)
- Dipeptide Composition (DPC)
- Sequence-level physicochemical stats
- k-mer binary fingerprints
"""

import numpy as np
import pandas as pd
from collections import Counter

STANDARD_AAS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

# ─── Physicochemical property tables ───
# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# Molecular weight of amino acids (Da)
MOLECULAR_WEIGHT = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.0, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
}

# Charge at pH 7 (approximate)
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

AROMATIC_AAS = set('FWY')
HYDROPHOBIC_AAS = set('AILMFVW')


def compute_aac(sequences: pd.Series) -> np.ndarray:
    """Amino Acid Composition — 20 features (frequency of each AA)."""
    n = len(sequences)
    features = np.zeros((n, 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        length = len(seq)
        if length == 0:
            continue
        counts = Counter(seq)
        for aa, idx in AA_TO_IDX.items():
            features[i, idx] = counts.get(aa, 0) / length
    return features


def compute_dpc(sequences: pd.Series) -> np.ndarray:
    """Dipeptide Composition — 400 features (frequency of each dipeptide)."""
    dipeptides = [a + b for a in STANDARD_AAS for b in STANDARD_AAS]
    dp_to_idx = {dp: i for i, dp in enumerate(dipeptides)}
    
    n = len(sequences)
    features = np.zeros((n, 400), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        length = len(seq)
        if length < 2:
            continue
        num_dp = length - 1
        for j in range(num_dp):
            dp = seq[j:j+2]
            if dp in dp_to_idx:
                features[i, dp_to_idx[dp]] += 1.0 / num_dp
    return features


def compute_seq_stats(sequences: pd.Series) -> np.ndarray:
    """
    Sequence-level physicochemical stats — 7 features:
    [length, net_charge, hydrophobic_ratio, avg_hydrophobicity,
     avg_molecular_weight, aromaticity, charge_density]
    """
    n = len(sequences)
    features = np.zeros((n, 7), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        length = len(seq)
        if length == 0:
            continue
        
        net_charge = sum(CHARGE.get(aa, 0) for aa in seq)
        hydrophobic_count = sum(1 for aa in seq if aa in HYDROPHOBIC_AAS)
        avg_hydro = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq])
        avg_mw = np.mean([MOLECULAR_WEIGHT.get(aa, 100) for aa in seq])
        aromatic_count = sum(1 for aa in seq if aa in AROMATIC_AAS)
        
        features[i] = [
            length,
            net_charge,
            hydrophobic_count / length,      # hydrophobic ratio
            avg_hydro,                         # avg hydrophobicity
            avg_mw,                            # avg molecular weight
            aromatic_count / length,           # aromaticity
            net_charge / length,               # charge density
        ]
    return features


SEQ_STAT_NAMES = [
    'length', 'net_charge', 'hydrophobic_ratio', 'avg_hydrophobicity',
    'avg_molecular_weight', 'aromaticity', 'charge_density'
]


def compute_kmer_fingerprint(sequences: pd.Series, k: int = 3) -> np.ndarray:
    """
    k-mer binary fingerprint — binary presence/absence of each k-mer.
    Returns a sparse-like dense matrix of shape (n, 20^k) for small k,
    or a hashed version for larger k.
    """
    if k > 3:
        # Use hashing for k > 3 to keep memory reasonable
        return _compute_kmer_hashed(sequences, k, n_features=4096)
    
    # Build all possible k-mers
    from itertools import product
    all_kmers = [''.join(p) for p in product(STANDARD_AAS, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    n_kmers = len(all_kmers)
    
    n = len(sequences)
    features = np.zeros((n, n_kmers), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = 1
    return features


def _compute_kmer_hashed(sequences: pd.Series, k: int, n_features: int = 4096) -> np.ndarray:
    """Hashed k-mer fingerprint for larger k values."""
    n = len(sequences)
    features = np.zeros((n, n_features), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            idx = hash(kmer) % n_features
            features[i, idx] = 1
    return features


AAC_NAMES = [f'aac_{aa}' for aa in STANDARD_AAS]

def get_dipeptide_names():
    return [f'dpc_{a}{b}' for a in STANDARD_AAS for b in STANDARD_AAS]
