"""
Generate country-level citation edges split by CPC technology section.
Joins tls224 (CPC classes) to assign each citing patent to a technology.

CPC sections: A=Human necessities, B=Operations/transport, C=Chemistry,
              D=Textiles, E=Construction, F=Engineering, G=Physics,
              H=Electricity, Y=General/misc

Output: country_citation_by_cpc.csv
Columns: citing_country, cited_country, cpc_section, year, citation_count
"""
import pandas as pd
from collections import defaultdict
import sys

print("Loading CPC classes (tls224)...")
# Read just the columns we need; drop duplicates (keep first section per patent)
cpc = pd.read_parquet("../rawdata/tls224.parquet", columns=["appln_id", "cpc_class_symbol"])
cpc["cpc_section"] = cpc["cpc_class_symbol"].str[0]
cpc = cpc[["appln_id", "cpc_section"]].drop_duplicates(subset="appln_id", keep="first")
print(f"  {len(cpc):,} unique patents with CPC sections")

print("Loading application metadata (tls201)...")
apps = pd.read_parquet("../rawdata/tls201.parquet",
                       columns=["appln_id", "appln_auth", "appln_filing_year"])
apps = apps.dropna(subset=["appln_filing_year"])
apps["appln_filing_year"] = apps["appln_filing_year"].astype(int)
print(f"  {len(apps):,} applications with filing years")

# Merge CPC into apps (left join — not all apps have CPC)
apps = apps.merge(cpc, on="appln_id", how="left")
cpc_coverage = apps["cpc_section"].notna().mean() * 100
print(f"  CPC coverage: {cpc_coverage:.1f}% of applications have CPC data")

# Build lookup Series: appln_id -> (country, year, cpc_section)
country_map = apps.set_index("appln_id")["appln_auth"]
year_map = apps.set_index("appln_id")["appln_filing_year"]
cpc_map = apps.set_index("appln_id")["cpc_section"]
del apps, cpc

# ---------- process citation edges ----------
print("\nProcessing citation edges in chunks (1M rows each)...")
CHUNK_SIZE = 1_000_000
# (citing_cty, cited_cty, cpc_section, year) -> count
counts = defaultdict(int)
chunks_done = 0
total_rows = 0
valid_rows = 0

for chunk in pd.read_csv("../processed/citation_edges.csv", chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    citing_cty = chunk["appln_id"].map(country_map)
    cited_cty = chunk["cited_appln_id"].map(country_map)
    citing_yr = chunk["appln_id"].map(year_map)
    citing_cpc = chunk["appln_id"].map(cpc_map)

    mask = (citing_cty.notna() & cited_cty.notna() &
            citing_yr.notna() & citing_cpc.notna())

    if mask.sum() == 0:
        continue

    valid_rows += mask.sum()
    valid = pd.DataFrame({
        "citing_country": citing_cty[mask].values,
        "cited_country": cited_cty[mask].values,
        "cpc_section": citing_cpc[mask].values,
        "year": citing_yr[mask].values,
    })

    chunk_counts = valid.groupby(
        ["citing_country", "cited_country", "cpc_section", "year"]
    ).size()

    for (c1, c2, cp, yr), n in chunk_counts.items():
        counts[(c1, c2, cp, yr)] += n

    chunks_done += 1
    if chunks_done % 20 == 0:
        print(f"  {chunks_done} chunks / {total_rows:,} rows -> {len(counts):,} unique combinations")

print(f"\nDone: {chunks_done} chunks, {total_rows:,} total, {valid_rows:,} with CPC")
print(f"Unique (country, country, section, year) combos: {len(counts):,}")

# ---------- save ----------
rows = [{"citing_country": c1, "cited_country": c2,
         "cpc_section": cp, "year": yr, "citation_count": n}
        for (c1, c2, cp, yr), n in sorted(counts.items(),
                                           key=lambda x: x[1], reverse=True)]
result = pd.DataFrame(rows)
result.to_csv("../processed/country_citation_by_cpc.csv", index=False)

print(f"\nSaved {len(result):,} rows to ../processed/country_citation_by_cpc.csv")
print(f"Year range: {result.year.min()} - {result.year.max()}")
print(f"CPC sections: {sorted(result.cpc_section.unique())}")

# Per-section summary
print("\n=== Citation volume by CPC section ===")
for sec in sorted(result.cpc_section.unique()):
    sub = result[result.cpc_section == sec]
    total = sub.citation_count.sum()
    cross = sub[sub.citing_country != sub.cited_country].citation_count.sum()
    print(f"  {sec}: {total:>15,} total, {cross:>12,} cross-border ({cross/total*100:.1f}%)")
