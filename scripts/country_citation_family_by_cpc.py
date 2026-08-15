"""
Family-level country citation edges split by CPC technology section.
Uses tls228 (family→family edges) instead of tls212 (application→application).
Avoids multi-filing inflation and the China 2010 coverage artifact.

Architecture:
  1. Build family→(country, year, cpc_section) lookup from tls201 + tls224
  2. Process tls228 in batches, map both sides
  3. Aggregate by (citing_country, cited_country, cpc_section, year)

Output: country_citation_family_by_cpc.csv
"""
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
import sys

CHUNK_SIZE = 2_000_000
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# ============================================================
# 1. Build family→(country, year, cpc_section) lookup
# ============================================================
print("[1/3] Building family→country+CPC lookup...")

# Load app-level metadata
apps = pd.read_parquet(
    "../rawdata/tls201.parquet",
    columns=["appln_id", "docdb_family_id", "appln_auth", "appln_filing_year"]
)
apps = apps.dropna(subset=["docdb_family_id", "appln_filing_year"])
apps["appln_filing_year"] = apps["appln_filing_year"].astype(int)
apps = apps[(apps["appln_filing_year"] >= 1800) & (apps["appln_filing_year"] <= 2020)]
print(f"  {len(apps):,} applications")

# Load CPC classes
cpc = pd.read_parquet(
    "../rawdata/tls224.parquet",
    columns=["appln_id", "cpc_class_symbol"]
)
cpc["cpc_section"] = cpc["cpc_class_symbol"].str[0]
cpc = cpc[["appln_id", "cpc_section"]].drop_duplicates(subset="appln_id", keep="first")
print(f"  {len(cpc):,} apps with CPC assignments ({len(cpc)/len(apps)*100:.1f}% coverage)")

# Merge CPC into apps
apps = apps.merge(cpc, on="appln_id", how="left")
del cpc

# For each family: take earliest filing, get its country + CPC section
apps = apps.sort_values(["docdb_family_id", "appln_filing_year"])
family_lookup = apps.groupby("docdb_family_id").first()
del apps

# Filter non-countries
family_lookup = family_lookup[~family_lookup["appln_auth"].isin(NON_COUNTRIES)]
print(f"  {len(family_lookup):,} families with country+CPC assignments")

# CPC coverage at family level
cpc_coverage = family_lookup["cpc_section"].notna().mean() * 100
print(f"  CPC coverage: {cpc_coverage:.1f}% of families have CPC data")

# Section distribution
print(f"  CPC sections: {sorted(family_lookup['cpc_section'].dropna().unique())}")

# Create lookup Series
country_map = family_lookup["appln_auth"]
year_map = family_lookup["appln_filing_year"]
cpc_map = family_lookup["cpc_section"]
del family_lookup

# ============================================================
# 2. Process tls228 edges
# ============================================================
print("\n[2/3] Processing family-level citation edges with CPC...")
counts = defaultdict(int)
chunks_done = 0
total_rows = 0
valid_rows = 0

pf = pq.ParquetFile("../rawdata/tls228.parquet")

for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
    chunk = batch.to_pandas()
    total_rows += len(chunk)

    citing_cty = chunk["docdb_family_id"].map(country_map)
    cited_cty = chunk["cited_docdb_family_id"].map(country_map)
    citing_yr = chunk["docdb_family_id"].map(year_map)
    citing_cpc = chunk["docdb_family_id"].map(cpc_map)

    mask = (citing_cty.notna() & cited_cty.notna() &
            citing_yr.notna() & citing_cpc.notna())

    if mask.sum() == 0:
        chunks_done += 1
        continue

    valid_rows += mask.sum()
    valid = pd.DataFrame({
        "citing_country": citing_cty[mask].values,
        "cited_country": cited_cty[mask].values,
        "cpc_section": citing_cpc[mask].values,
        "year": citing_yr[mask].values,
    })

    for (c1, c2, cp, yr), n in valid.groupby(
        ["citing_country", "cited_country", "cpc_section", "year"]
    ).size().items():
        counts[(c1, c2, cp, yr)] += n

    chunks_done += 1
    if chunks_done % 10 == 0:
        print(f"  {chunks_done} batches / {total_rows:,} rows → {len(counts):,} unique")

print(f"\n  Done: {chunks_done} batches, {total_rows:,} total, {valid_rows:,} mapped with CPC")
print(f"  Unique (country, country, section, year) combos: {len(counts):,}")

# ============================================================
# 3. Save
# ============================================================
print("\n[3/3] Saving...")
rows = [{"citing_country": c1, "cited_country": c2,
         "cpc_section": cp, "year": yr, "citation_count": n}
        for (c1, c2, cp, yr), n in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
result = pd.DataFrame(rows)

# Filter to reasonable year range
result = result[(result["year"] >= 1995) & (result["year"] <= 2017)]

result.to_csv("../processed/country_citation_family_by_cpc.csv", index=False)
print(f"  Saved {len(result):,} rows → ../processed/country_citation_family_by_cpc.csv")
print(f"  Year range: {result['year'].min()} – {result['year'].max()}")

# Per-section summary
print(f"\n=== Family-level citation volume by CPC section ===")
for sec in sorted(result["cpc_section"].unique()):
    sub = result[result["cpc_section"] == sec]
    total = sub["citation_count"].sum()
    cross = sub[sub["citing_country"] != sub["cited_country"]]["citation_count"].sum()
    print(f"  {sec}: {total:>15,} total, {cross:>12,} cross-border ({cross/total*100:.1f}%)")

print(f"\nDone. Family-level CPC edges saved.")
