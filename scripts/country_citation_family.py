"""
Rebuild citation network at DOCDB family level (one node = one invention).
Avoids inflating countries that split inventions into many filings (e.g., Japan).

Compared to the application-level version (country_citation_edges.py /
country_citation_by_year.py), this:
- Uses tls228 (pre-built family→family edges) — no join needed
- Assigns each family ONE country (earliest filing's appln_auth)
- Assigns each family ONE year (earliest filing year)
- Outputs: country_citation_family_by_year.csv

Usage:
  python country_citation_family.py
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

CHUNK_SIZE = 2_000_000
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# ============================================================
# 1. Build family → (country, year) lookup from tls201
# ============================================================
print("[1/4] Building family→country lookup from tls201...")
apps = pd.read_parquet(
    "../rawdata/tls201.parquet",
    columns=["docdb_family_id", "appln_auth", "appln_filing_year"]
)
apps = apps.dropna(subset=["docdb_family_id", "appln_filing_year"])
apps["appln_filing_year"] = apps["appln_filing_year"].astype(int)

# Filter to reasonable year range
apps = apps[(apps["appln_filing_year"] >= 1800) & (apps["appln_filing_year"] <= 2020)]

# For each family, take the earliest filing year and corresponding country
apps = apps.sort_values(["docdb_family_id", "appln_filing_year"])
family_lookup = apps.groupby("docdb_family_id").first()
del apps

# Filter out non-country offices
family_lookup = family_lookup[~family_lookup["appln_auth"].isin(NON_COUNTRIES)]

print(f"  {len(family_lookup):,} families with country assignments")
print(f"  Year range: {family_lookup['appln_filing_year'].min()} – {family_lookup['appln_filing_year'].max()}")
top_fams = family_lookup["appln_auth"].value_counts().head(10)
print(f"  Top countries by family count: {dict(top_fams)}")

# Create lookup Series for fast mapping
country_map = family_lookup["appln_auth"]
year_map = family_lookup["appln_filing_year"]
del family_lookup

# ============================================================
# 2. Check China's application-level vs family-level stats
# ============================================================
print("\n[2/4] China diagnostic: comparing app-level vs family-level...")

# Quick load of app-level data for comparison
cite_app = pd.read_csv("../processed/country_citation_edges_by_year.csv")
cn_app_citing = cite_app[
    (cite_app["citing_country"] == "CN") &
    (cite_app["year"].between(2005, 2015))
].groupby("year")["citation_count"].sum()
print("  CN citing (app-level) 2005-2015:")
for yr, val in cn_app_citing.items():
    print(f"    {yr}: {val:>15,}")
del cite_app

# ============================================================
# 3. Process tls228 edges in chunks
# ============================================================
print("\n[3/4] Processing family-level citation edges...")
counts = defaultdict(int)
chunks_done = 0
total_rows = 0
valid_rows = 0

for chunk in pd.read_parquet(
    "../rawdata/tls228.parquet",
    columns=["docdb_family_id", "cited_docdb_family_id"]
):
    # pd.read_parquet on a large file returns a DataFrame; we process in chunks manually
    pass

# Parquet doesn't support chunksize natively; use pyarrow batch reading
import pyarrow.parquet as pq

pf = pq.ParquetFile("../rawdata/tls228.parquet")
batch_size = 2_000_000

for batch in pf.iter_batches(batch_size=batch_size):
    chunk = batch.to_pandas()
    total_rows += len(chunk)

    # Map citing family → country/year
    citing_cty = chunk["docdb_family_id"].map(country_map)
    citing_yr = chunk["docdb_family_id"].map(year_map)
    # Map cited family → country (year not needed for grouping)
    cited_cty = chunk["cited_docdb_family_id"].map(country_map)

    mask = citing_cty.notna() & cited_cty.notna() & citing_yr.notna()
    if mask.sum() == 0:
        chunks_done += 1
        continue

    valid_rows += mask.sum()
    valid = pd.DataFrame({
        "citing_country": citing_cty[mask].values,
        "cited_country": cited_cty[mask].values,
        "year": citing_yr[mask].values,
    })

    # Group and accumulate
    for (c1, c2, yr), n in valid.groupby(
        ["citing_country", "cited_country", "year"]
    ).size().items():
        counts[(c1, c2, yr)] += n

    chunks_done += 1
    if chunks_done % 10 == 0:
        print(f"  {chunks_done} batches / {total_rows:,} rows → {len(counts):,} unique")

print(f"\n  Done: {chunks_done} batches, {total_rows:,} total, {valid_rows:,} mapped")
print(f"  Unique (country, country, year) combos: {len(counts):,}")

# ============================================================
# 4. Save and report
# ============================================================
print("\n[4/4] Saving family-level citation edges...")
rows = [
    {"citing_country": c1, "cited_country": c2,
     "year": yr, "citation_count": n}
    for (c1, c2, yr), n in sorted(counts.items(), key=lambda x: x[1], reverse=True)
]
result = pd.DataFrame(rows)

# Filter to reasonable year range
result = result[
    (result["year"] >= 1995) & (result["year"] <= 2017)
]
result.to_csv("../processed/country_citation_family_by_year.csv", index=False)
print(f"  Saved {len(result):,} rows → ../processed/country_citation_family_by_year.csv")
print(f"  Year range: {result['year'].min()} – {result['year'].max()}")

# ---- Family-level totals ----
print(f"\n=== Family-level citation totals ===")
total_cites = result["citation_count"].sum()
cross_cites = result[result["citing_country"] != result["cited_country"]]["citation_count"].sum()
print(f"  Total: {total_cites:,.0f}")
print(f"  Cross-border: {cross_cites:,.0f} ({cross_cites/total_cites*100:.1f}%)")

# Top cited countries (family-level)
cited = result.groupby("cited_country")["citation_count"].sum().sort_values(ascending=False)
print(f"\n  Top 10 cited countries (family-level):")
for c, v in cited.head(10).items():
    print(f"    {c}: {v:>15,}")

# ---- China diagnostic ----
print(f"\n=== China citing totals: app-level vs family-level ===")
cn_family_citing = result[result["citing_country"] == "CN"].groupby("year")["citation_count"].sum()
print(f"  {'Year':<6} {'App-level':>15} {'Family-level':>15} {'Ratio':>8}")
print(f"  {'-'*6} {'-'*15} {'-'*15} {'-'*8}")

# Re-load app-level for comparison
cite_app = pd.read_csv("../processed/country_citation_edges_by_year.csv")
cn_app = cite_app[cite_app["citing_country"] == "CN"].groupby("year")["citation_count"].sum()

for yr in range(2000, 2018):
    app_val = cn_app.get(yr, 0)
    fam_val = cn_family_citing.get(yr, 0)
    if app_val > 0 and fam_val > 0:
        ratio = app_val / fam_val
        flag = "  ⚠" if yr >= 2010 and ratio > 5 else ""
        print(f"  {yr:<6} {app_val:>15,} {fam_val:>15,} {ratio:>7.1f}x{flag}")

# Also check: what share of China's app-level citations are within-family?
print(f"\n=== Interpretation ===")
# For China in 2010-2015, compare trends
for yr in [2010, 2011, 2012, 2013, 2014, 2015]:
    app_val = cn_app.get(yr, 0)
    fam_val = cn_family_citing.get(yr, 0)
    if app_val > 0:
        ratio = app_val / fam_val
        print(f"  {yr}: app-level = {app_val:,}, family-level = {fam_val:,}, ratio = {ratio:.1f}x")

print(f"\nDone. Family-level edges saved.")
print(f"Next: update lead_lag_analysis.py and visualizations to use family-level data.")
