"""
Generate year-filtered country-level citation edges.
Groups by (citing_country, cited_country, citing_year) so each
year can be compared directly to trade snapshots.

Output: country_citation_edges_by_year.csv
Columns: citing_country, cited_country, year, citation_count
"""
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

print("Loading country + year mapping from PATSTAT applications...")
apps = pd.read_parquet("../rawdata/tls201.parquet",
                       columns=["appln_id", "appln_auth", "appln_filing_year"])
apps = apps.dropna(subset=["appln_filing_year"])
apps["appln_filing_year"] = apps["appln_filing_year"].astype(int)
print(f"Loaded {len(apps):,} applications with filing years")

# Build lookup Series (appln_id -> value)
country_map = apps.set_index("appln_id")["appln_auth"]
year_map = apps.set_index("appln_id")["appln_filing_year"]
del apps

print("Processing citation edges in chunks (1M rows each)...")
CHUNK_SIZE = 1_000_000
# (citing_country, cited_country, year) -> count
yearly_counts = defaultdict(int)
chunks_done = 0
total_rows = 0

for chunk in pd.read_csv("../processed/citation_edges.csv", chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    # Map both sides to country and year
    citing_cty = chunk["appln_id"].map(country_map)
    cited_cty = chunk["cited_appln_id"].map(country_map)
    citing_yr = chunk["appln_id"].map(year_map)

    mask = citing_cty.notna() & cited_cty.notna() & citing_yr.notna()
    if mask.sum() == 0:
        continue

    valid = pd.DataFrame({
        "citing_country": citing_cty[mask].values,
        "cited_country": cited_cty[mask].values,
        "year": citing_yr[mask].values,
    })

    # Vectorized groupby
    chunk_counts = valid.groupby(["citing_country", "cited_country", "year"]).size()

    for (c1, c2, yr), n in chunk_counts.items():
        yearly_counts[(c1, c2, yr)] += n

    chunks_done += 1
    if chunks_done % 20 == 0:
        print(f"  {chunks_done} chunks / {total_rows:,} rows -> {len(yearly_counts):,} unique (country, country, year) triples")

print(f"\nDone: {chunks_done} chunks, {total_rows:,} total rows")
print(f"Unique (country, country, year) triples: {len(yearly_counts):,}")

# Convert to DataFrame
rows = [{"citing_country": c1, "cited_country": c2, "year": yr, "citation_count": n}
        for (c1, c2, yr), n in sorted(yearly_counts.items(),
                                       key=lambda x: x[1], reverse=True)]
result = pd.DataFrame(rows)
result.to_csv("../processed/country_citation_edges_by_year.csv", index=False)

print(f"\nSaved {len(result):,} rows to ../processed/country_citation_edges_by_year.csv")
print(f"Year range: {result.year.min()} - {result.year.max()}")
print(f"Countries: {len(set(result.citing_country) | set(result.cited_country))}")

# Show a sample
print("\nTop 10 country-pair-year entries:")
for _, r in result.head(10).iterrows():
    print(f"  {r.citing_country:>3} -> {r.cited_country:<3}  yr={r.year}  {r.citation_count:>12,}")
