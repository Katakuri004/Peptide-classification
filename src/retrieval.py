import pandas as pd
from typing import Optional, Tuple

class HierarchicalRetriever:
    """
    A 3-level hierarchical retrieval framework for peptide classification.
    Level 1: Exact Sequence Match.
    Level 2: Homology/Similarity Search using Levenshtein distance.
    Level 3: Fallback (returns None to indicate ML model should be used).
    """
    def __init__(self, db_path: str, similarity_threshold: float = 0.5):
        """
        Args:
            db_path: Path to the CSV knowledge base with 'Sequence' and 'Label'.
            similarity_threshold: Minimum similarity score (0 to 1) to consider a match in Level 2.
        """
        self.db_path = db_path
        self.threshold = similarity_threshold
        
        # Load database
        try:
            self.db = pd.read_csv(db_path)
            # Create a dictionary for O(1) Level 1 lookups
            self.exact_dict = dict(zip(self.db['Sequence'], self.db['Label']))
        except FileNotFoundError:
            print(f"Warning: Database {db_path} not found. Retrieval will fall back to ML model.")
            self.db = pd.DataFrame(columns=['Sequence', 'Label'])
            self.exact_dict = {}

    def _get_kmers(self, seq: str, k: int = 3) -> set:
        """Extracts overlapping k-mers from a sequence."""
        if len(seq) < k:
            return {seq}
        return set([seq[i:i+k] for i in range(len(seq) - k + 1)])

    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate Jaccard similarity using 3-mers (0 to 1)."""
        if not seq1 or not seq2:
            return 0.0
            
        set1 = self._get_kmers(seq1)
        set2 = self._get_kmers(seq2)
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def retrieve(self, sequence: str) -> Tuple[Optional[int], str]:
        """
        Hierarchical retrieval for a single sequence.
        Returns:
            Tuple of (Predicted Label, 'Source Level')
            or (None, 'Model Fallback') if no match is found.
        """
        # Level 1: Exact Match
        if sequence in self.exact_dict:
            return self.exact_dict[sequence], 'Level 1 (Exact Match)'
            
        # Level 2: Homology Search
        # A simple linear scan. In a production system, use FAISS or MMseqs2.
        best_score = 0.0
        best_label = None
        
        for index, row in self.db.iterrows():
            db_seq = row['Sequence']
            score = self._calculate_similarity(sequence, db_seq)
            if score > best_score:
                best_score = score
                best_label = row['Label']
                
        if best_score >= self.threshold:
            return best_label, f'Level 2 (Homology: {best_score:.2f})'
            
        # Level 3: Fallback to ML Model
        return None, 'Level 3 (Model Fallback)'

if __name__ == '__main__':
    # Simple test
    retriever = HierarchicalRetriever(db_path="Data/external/peptide_database.csv", similarity_threshold=0.5)
    print("Test exact match (if db exists):")
    # You would test with actual exact sequences
    print("Fallback test:", retriever.retrieve("AWKWAKWAKAWKWAW"))
