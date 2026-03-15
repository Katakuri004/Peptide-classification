import os
import requests
import pandas as pd
from typing import List, Dict

# Create directories if they don't exist
os.makedirs("Data/external", exist_ok=True)
os.makedirs("Data/processed", exist_ok=True)

def fetch_uniprot_stream(query: str, label: int, max_records: int = 25000) -> List[Dict]:
    """Fetches protein sequences from UniProt using stream API for large datasets."""
    print(f"Streaming data for query (up to {max_records}): {query}")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "tsv",
        "fields": "sequence"
    }
    
    sequences = []
    # Use streaming to not overload memory
    with requests.get(url, params=params, stream=True) as response:
        response.raise_for_status()
        header_skipped = False
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8').strip()
            # Skip the first TSV header line
            if not header_skipped:
                header_skipped = True
                continue
                
            # Basic validation
            if 0 < len(line_str) < 200:
                 sequences.append({
                     "Sequence": line_str,
                     "Label": label
                 })
            
            if len(sequences) >= max_records:
                break
                
    print(f"Fetched {len(sequences)} valid short sequences for query.")
    return sequences

def main():
    print("Starting data sourcing...")
    
    # 1. Fetch Positive samples (Antimicrobial)
    # Using keyword KW-0929 (Antimicrobial). Dropping 'reviewed:true' to get >25k sequences.
    pos_data = fetch_uniprot_stream("keyword:KW-0929", label=1, max_records=25000)
    
    # 2. Fetch Negative samples
    # Using NOT Antimicrobial to get random proteins (from Human to satisfy API rate limits), limiting to short ones
    neg_data = fetch_uniprot_stream("length:[10 TO 150] AND taxonomy_id:9606 AND NOT keyword:KW-0929", label=0, max_records=25000)
    
    # Combine
    all_data = pos_data + neg_data
    df = pd.DataFrame(all_data)
    
    # Dedup
    df = df.drop_duplicates(subset=["Sequence"])
    
    output_path = "Data/external/peptide_database.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    main()
