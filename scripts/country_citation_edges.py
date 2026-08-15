"""
Generate country-level citation edges from patent-level citation_edges.csv.
Uses chunked reading + pandas merge for memory efficiency.
"""
import pandas as pd
from collections import Counter

print("Loading country mapping...")
# Load as Series: appln_id -> appln_auth (compact, uses numpy arrays internally)
apps = pd.read_parquet("../rawdata/tls201.parquet", columns=["appln_id", "appln_auth"])
apps = apps.set_index("appln_id")["appln_auth"]
print(f"Loaded {len(apps):,} applications, memory: {apps.memory_usage(deep=True) / 1e9:.2f} GB")

print("Processing citation edges in chunks (1M rows each)...")
CHUNK_SIZE = 1_000_000
country_counts = Counter()
chunks_done = 0
total_rows = 0

for chunk in pd.read_csv("../processed/citation_edges.csv", chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    # Vectorized map using pandas Series (way faster than Python dict)
    citing = chunk["appln_id"].map(apps)
    cited = chunk["cited_appln_id"].map(apps)

    mask = citing.notna() & cited.notna()
    if mask.sum() == 0:
        continue

    # Vectorized groupby count (fast, C-level)
    valid = pd.DataFrame({"citing": citing[mask].values, "cited": cited[mask].values})
    chunk_counts = valid.groupby(["citing", "cited"]).size()

    # Accumulate into Python Counter (only unique country pairs, max ~40K)
    for (c1, c2), n in chunk_counts.items():
        country_counts[(c1, c2)] += n

    chunks_done += 1
    if chunks_done % 20 == 0:
        print(f"  {chunks_done} chunks / {total_rows:,} rows -> {len(country_counts):,} unique country pairs")

print(f"\nDone: {chunks_done} chunks, {total_rows:,} total rows")
print(f"Unique country pairs: {len(country_counts):,}")

# Convert to DataFrame and save
rows = [{"citing_country": c1, "cited_country": c2, "citation_count": n}
        for (c1, c2), n in country_counts.most_common()]
result = pd.DataFrame(rows)
result.to_csv("../processed/country_citation_edges.csv", index=False)

print(f"Saved {len(result):,} rows to ../processed/country_citation_edges.csv")
print(f"\nTop 10 knowledge flows:")
for i, row in result.head(10).iterrows():
    print(f"  {row.citing_country:>3} -> {row.cited_country:<3}  {row.citation_count:>12,}")
