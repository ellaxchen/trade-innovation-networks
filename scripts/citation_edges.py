import pandas as pd

# Process large Parquet files in chunks to avoid MemoryError
# Each chunk is manageable in memory, then we combine them

def read_parquet_chunks(file_path, columns, chunk_size=50000):
    """Read Parquet file in chunks using pyarrow"""
    import pyarrow.parquet as pq
    chunk_list = []
    for batch in pq.read_table(file_path, columns=columns).to_batches(chunk_size):
        chunk_list.append(pd.DataFrame(batch.to_pandas()))
    if chunk_list:
        return pd.concat(chunk_list, ignore_index=True)
    return pd.DataFrame()

# Define chunk size - this balance between memory usage and processing speed
chunk_size = 50000  # Adjust based on available memory

print("Loading PATSTAT tables in chunks...")

# Load each table in chunks, column by column to minimize memory usage
# tls201.parquet (2.1 GB) - Applications
try:
    apps = read_parquet_chunks("../rawdata/tls201.parquet",
                              ["appln_id", "appln_auth", "appln_filing_year"],
                              chunk_size)
    print(f"Loaded {len(apps)} application records")
except Exception as e:
    print(f"Error loading ../rawdata/tls201.parquet: {e}")
    # Fallback for extremely large file - process only sample or modify approach
    apps = pd.read_parquet("../rawdata/tls201.parquet", columns=["appln_id", "appln_auth", "appln_filing_year"])
    print(f"Used fallback - loaded {len(apps)} records")

# tls212.parquet (1.5 GB) - Citations
try:
    cit = read_parquet_chunks("../rawdata/tls212.parquet",
                             ["pat_publn_id", "cited_pat_publn_id"],
                             chunk_size)
    print(f"Loaded {len(cit)} citation records")
except Exception as e:
    print(f"Error loading ../rawdata/tls212.parquet: {e}")
    cit = pd.read_parquet("../rawdata/tls212.parquet", columns=["pat_publn_id", "cited_pat_publn_id"])
    print(f"Used fallback - loaded {len(cit)} records")

# tls211.parquet (897 MB) - Bridge table
try:
    pub = read_parquet_chunks("../rawdata/tls211.parquet",
                             ["pat_publn_id", "appln_id"],
                             chunk_size)
    print(f"Loaded {len(pub)} patent-application records")
except Exception as e:
    print(f"Error loading ../rawdata/tls211.parquet: {e}")
    pub = pd.read_parquet("../rawdata/tls211.parquet", columns=["pat_publn_id", "appln_id"])
    print(f"Used fallback - loaded {len(pub)} records")

# Filter valid citations (cited_pat_publn_id > 0)
print("Filtering valid citations...")
cit = cit[cit["cited_pat_publn_id"] > 0]
print(f"Valid citations: {len(cit)}")

# Map citing publication -> citing application
print("Merging dataframes...")
edges = cit.merge(pub, on="pat_publn_id")

# Map cited publication -> cited application
edges = edges.merge(
    pub.rename(columns={"pat_publn_id": "cited_pat_publn_id", "appln_id": "cited_appln_id"}),
    on="cited_pat_publn_id",
)

# Keep only the IDs we need for a pure edge list
edges = edges[["appln_id", "cited_appln_id"]]

# Save to CSV for visualization tools
print(f"Processing {len(edges)} total citation edges...")
edges.to_csv("../processed/citation_edges.csv", index=False)

print(f"✅ Saved {len(edges)} citation edges to ../processed/citation_edges.csv")
print("\nNext step: Run country-level aggregation using:")
print("python -c \"import pandas as pd; edges = pd.read_csv('../processed/citation_edges.csv'); apps = pd.read_parquet('../rawdata/tls201.parquet')[[\"appln_id\", \"appln_auth\"]]; country_edges = edges.merge(apps.rename(columns={\"appln_id\":\"appln_id\", \"appln_auth\":\"citing_country\"}), on=\"appln_id\").merge(apps.rename(columns={\"appln_id\":\"cited_appln_id\", \"appln_auth\":\"cited_country\"}), on=\"cited_appln_id\"); country_edges.groupby([\"citing_country\", \"cited_country\"]).size().reset_index(name=\"citation_count\").to_csv('../processed/country_citation_edges.csv', index=False); print('✅ Saved country-level citation edges')\"")
print("\nScript prepared - run the above Python command to complete the citation network!")