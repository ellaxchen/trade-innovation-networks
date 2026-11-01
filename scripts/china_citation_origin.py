"""
China 2010 citation-origin tabulation.

The paper previously attributed China's app-level citing jump (2,148 → 1,215,925
from 2009 → 2010) to "EPO completing its indexing of Chinese documents".
Redo the tabulation directly from tls212 to pin down the actual mechanism.

For every tls212 citation row, map the citing publication through tls211 to
(publn_auth, publn_year) and tabulate citations by CN publication year ×
citn_origin (and citn_gener_auth). A US tabulation is included as a control.

Hypothesis to verify: tls212 contains ZERO citations from CN publications
before 2010, even though CN applications in tls201 grow smoothly through 2010.
From 2010 on, CN-publication citations appear and are overwhelmingly
citn_origin = SEA with citn_gener_auth = CN — i.e., CNIPA only began sharing
examiner (search-report) citations with EPO in 2010. Indexing was never the
issue; the pre-2010 absence was on the citation side, not the application side.

Outputs:
  ../processed/citation_origin_tabulation.csv   (publn_country, year, citn_origin, citn_gener_auth, citations)
  ../processed/cn_appln_counts_by_year.csv      (CN application counts from tls201, filing year)

Usage: python china_citation_origin.py
"""
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

CHUNK_SIZE = 1_000_000
CONTROL_COUNTRIES = ["CN", "US"]  # CN = case, US = control

# ============================================================
# 1. tls211: citing publication -> (authority, publication year)
# ============================================================
print("[1/3] Loading tls211 publication metadata...")
pub = pd.read_parquet(
    "../rawdata/tls211.parquet",
    columns=["pat_publn_id", "publn_auth", "publn_date"]
)
pub["publn_year"] = pd.to_numeric(pub["publn_date"].astype(str).str[:4], errors="coerce")
pub = pub.dropna(subset=["publn_year"])
pub["publn_year"] = pub["publn_year"].astype(int)
pub = pub[(pub["publn_year"] >= 1995) & (pub["publn_year"] <= 2017)]
print(f"  {len(pub):,} publications 1995-2017")
print(f"  publn_auth values: {sorted(pub['publn_auth'].unique())[:20]}...")

auth_map = pub.set_index("pat_publn_id")["publn_auth"]
year_map = pub.set_index("pat_publn_id")["publn_year"]
del pub

# ============================================================
# 2. CN application counts from tls201 (the smooth baseline)
# ============================================================
print("\n[2/3] Counting CN applications per year from tls201...")
cn_apps = defaultdict(int)
pf201 = pq.ParquetFile("../rawdata/tls201.parquet")
for batch in pf201.iter_batches(batch_size=CHUNK_SIZE, columns=["appln_auth", "appln_filing_year"]):
    df = batch.to_pandas()
    mask = (df["appln_auth"] == "CN") & df["appln_filing_year"].between(1995, 2017)
    for yr, n in df.loc[mask].groupby("appln_filing_year").size().items():
        cn_apps[int(yr)] += int(n)
app_rows = pd.DataFrame(
    [{"year": yr, "cn_applications": n} for yr, n in sorted(cn_apps.items())]
)
app_rows.to_csv("../processed/cn_appln_counts_by_year.csv", index=False)
print("  CN applications by year (tls201, filing year):")
for _, r in app_rows.iterrows():
    print(f"    {r['year']}: {r['cn_applications']:>12,}")
del cn_apps

# ============================================================
# 3. tls212: tabulate citations by citing-publication year × origin
# ============================================================
print("\n[3/3] Tabulating tls212 citations by publication year x citn_origin...")
counts = defaultdict(int)   # (publn_auth, year, citn_origin, citn_gener_auth) -> n
total_rows = 0

pf212 = pq.ParquetFile("../rawdata/tls212.parquet")
n_batches = 0
for batch in pf212.iter_batches(
    batch_size=CHUNK_SIZE,
    columns=["pat_publn_id", "citn_origin", "citn_gener_auth"]
):
    chunk = batch.to_pandas()
    total_rows += len(chunk)

    auth = chunk["pat_publn_id"].map(auth_map)
    yr = chunk["pat_publn_id"].map(year_map)
    keep = auth.notna() & yr.notna() & auth.isin(CONTROL_COUNTRIES)
    if keep.sum() == 0:
        n_batches += 1
        continue

    sub = pd.DataFrame({
        "publn_auth": auth[keep].values,
        "year": yr[keep].values.astype(int),
        "citn_origin": chunk["citn_origin"][keep].values,
        "citn_gener_auth": chunk["citn_gener_auth"][keep].values,
    })
    for (c, y, o, g), n in sub.groupby(
        ["publn_auth", "year", "citn_origin", "citn_gener_auth"]
    ).size().items():
        counts[(c, y, o, g)] += int(n)

    n_batches += 1
    if n_batches % 20 == 0:
        print(f"  {n_batches} batches / {total_rows:,} rows processed")

print(f"\n  Done: {total_rows:,} total citation rows scanned")
rows = [
    {"publn_country": c, "year": y, "citn_origin": o,
     "citn_gener_auth": g, "citations": n}
    for (c, y, o, g), n in sorted(counts.items())
]
result = pd.DataFrame(rows)
result.to_csv("../processed/citation_origin_tabulation.csv", index=False)
print(f"  Saved {len(result):,} rows -> ../processed/citation_origin_tabulation.csv")

# ============================================================
# 4. Report
# ============================================================
print("\n=== CN citations in tls212 by citing-publication year ===")
cn = result[result["publn_country"] == "CN"].groupby("year")["citations"].sum()
for yr in range(1995, 2018):
    if yr in cn.index or yr >= 2005:
        print(f"  {yr}: {cn.get(yr, 0):>12,}")

print("\n=== CN by citn_origin, 2008-2012 (the discontinuity) ===")
cn_wide = result[result["publn_country"] == "CN"].pivot_table(
    index="year", columns="citn_origin", values="citations", aggfunc="sum", fill_value=0)
print(cn_wide.loc[2008:2012].to_string())

print("\n=== citn_gener_auth for CN citations, 2010-2012 ===")
cn_gen = result[result["publn_country"] == "CN"].pivot_table(
    index="year", columns="citn_gener_auth", values="citations", aggfunc="sum", fill_value=0)
print(cn_gen.loc[2010:2012].to_string())

print("\n=== US control: citations by citing-publication year, 2005-2012 ===")
us = result[result["publn_country"] == "US"].groupby("year")["citations"].sum()
for yr in range(2005, 2013):
    print(f"  {yr}: {us.get(yr, 0):>12,}")

print("\nDone.")
