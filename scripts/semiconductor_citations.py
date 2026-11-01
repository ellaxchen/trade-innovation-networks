"""
Semiconductor (H01L) family-level citation edges — BOTH country definitions.

Case study for the paper's closing section: who leads in semiconductors?
Builds the H01L knowledge network at DOCDB family level, 1995-2017.

Country definitions (output files):
  appln_auth  (filing office, proxy)      -> semiconductor_citations_family_by_year.csv
  inventor country (tls206/tls207)        -> semiconductor_citations_family_by_year_inventor.csv
      Inventor country = country of the first-listed inventor (lowest
      invt_seq_nr) with a non-blank person_ctry_code, built by
      build_inventor_country.py. Coverage is limited by PATSTAT person
      records (~34% of applications in this edition); report it as such.

CPC note: this spring-2018 PATSTAT edition contains ZERO H10 rows; H01L is
the complete semiconductor class available here.

Architecture (mirrors country_citation_family_by_cpc.py, but at CPC CLASS
granularity — cpc_class_symbol startswith "H01L" — instead of section letter):
  1. tls201 + tls224 → family → (country, year, is_H01L) lookup.
     Country/year = earliest filing (canonical). A family is H01L if ANY
     member application carries an H01L CPC class (the earliest filing often
     predates CPC assignment, so max() across members is more faithful than
     first()).
  2. tls228 chunks → keep edges where the CITING family is H01L (semiconductor
     patents building on prior art; cited side unrestricted — semiconductor
     patents cite physics, chemistry, etc.).
  3. Aggregate (citing_country, cited_country, year) for both definitions.

Usage: python semiconductor_citations.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

CHUNK_SIZE = 2_000_000
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# ============================================================
# 1. Family → (country, inventor country, year, is_H01L) lookup
# ============================================================
print("[1/3] Building family→country+year+H01L lookup...")

apps = pd.read_parquet(
    "../rawdata/tls201.parquet",
    columns=["appln_id", "docdb_family_id", "appln_auth", "appln_filing_year"]
)
apps = apps.dropna(subset=["docdb_family_id", "appln_filing_year"])
apps["appln_filing_year"] = apps["appln_filing_year"].astype(int)
apps = apps[(apps["appln_filing_year"] >= 1800) & (apps["appln_filing_year"] <= 2020)]
print(f"  {len(apps):,} applications")

cpc = pd.read_parquet(
    "../rawdata/tls224.parquet",
    columns=["appln_id", "cpc_class_symbol"]
)
cpc = cpc[cpc["cpc_class_symbol"].str.startswith("H01L", na=False)]
cpc["is_h01l"] = True
cpc = cpc[["appln_id", "is_h01l"]].drop_duplicates(subset="appln_id")
print(f"  {len(cpc):,} applications with an H01L class")

apps = apps.merge(cpc, on="appln_id", how="left")
apps["is_h01l"] = apps["is_h01l"].fillna(False)
del cpc

print("[1/3] Attaching inventor country...", flush=True)
inv = pd.read_parquet("../processed/appln_inventor_country.parquet")
inv["inventor_country"] = inv["inventor_country"].astype("category")
apps = apps.merge(inv, on="appln_id", how="left")
del inv
inv_hit = apps["inventor_country"].notna().sum()
print(f"  inventor country available for {inv_hit:,} / {len(apps):,} applications "
      f"({inv_hit/len(apps)*100:.1f}%)", flush=True)

# Earliest filing decides country/year; any-member H01L flag.
# The two country lookups are built separately: the base lookup runs on all
# 94M applications (2-key sort), the inventor lookup on the 32M applications
# with a known inventor country. This is much faster than one 3-key sort and,
# critically, keeps EP/WO-filed families for the inventor definition.
print("  Building base family lookup...", flush=True)
apps["appln_filing_year"] = apps["appln_filing_year"].astype("int16")
apps["appln_auth"] = apps["appln_auth"].astype("category")
apps = apps.drop(columns=["appln_id"])
apps_sorted = apps.sort_values(["docdb_family_id", "appln_filing_year"])
fam = apps_sorted["docdb_family_id"].to_numpy()
first_mask = np.empty(len(fam), dtype=bool)
first_mask[0] = True
first_mask[1:] = fam[1:] != fam[:-1]
base = apps_sorted.loc[
    first_mask, ["docdb_family_id", "appln_auth", "appln_filing_year"]
].set_index("docdb_family_id")
del apps_sorted, fam, first_mask
h01l_ids = apps.loc[apps["is_h01l"], "docdb_family_id"].unique()
base["is_h01l"] = base.index.isin(h01l_ids)
del h01l_ids
print(f"  base lookup: {len(base):,} families", flush=True)

print("  Building inventor family lookup...", flush=True)
inv_apps = apps.loc[
    apps["inventor_country"].notna(),
    ["docdb_family_id", "appln_filing_year", "inventor_country"],
]
del apps
inv_apps = inv_apps.sort_values(["docdb_family_id", "appln_filing_year"])
fam = inv_apps["docdb_family_id"].to_numpy()
first_mask = np.empty(len(fam), dtype=bool)
first_mask[0] = True
first_mask[1:] = fam[1:] != fam[:-1]
inv_fam = inv_apps.loc[
    first_mask, ["docdb_family_id", "inventor_country"]
].set_index("docdb_family_id")["inventor_country"]
del inv_apps, fam, first_mask
print(f"  inventor lookup: {len(inv_fam):,} families", flush=True)

family_lookup = base.join(inv_fam, how="left")
del base, inv_fam

# Do NOT drop EP/WO families here: those are filing offices, but their
# inventors may be US/DE/JP/etc. The auth map drops them (not countries);
# the inventor map keeps them and uses the inventor's country.
family_lookup["appln_auth"] = family_lookup["appln_auth"].astype("string")
family_lookup.loc[
    family_lookup["appln_auth"].isin(NON_COUNTRIES), "appln_auth"] = pd.NA
family_lookup["inventor_country"] = family_lookup["inventor_country"].astype("string")
family_lookup.loc[
    family_lookup["inventor_country"].isin(NON_COUNTRIES), "inventor_country"] = pd.NA

n_h01l = family_lookup["is_h01l"].sum()
print(f"  {len(family_lookup):,} families total; "
      f"{n_h01l:,} ({n_h01l/len(family_lookup)*100:.1f}%) are H01L")
print(f"  filing-office country: "
      f"{family_lookup['appln_auth'].notna().sum():,} families "
      f"({family_lookup['appln_auth'].notna().mean()*100:.1f}%)")
print(f"  inventor country: "
      f"{family_lookup['inventor_country'].notna().sum():,} families "
      f"({family_lookup['inventor_country'].notna().mean()*100:.1f}%)")

auth_map = family_lookup["appln_auth"]
inv_map = family_lookup["inventor_country"]
year_map = family_lookup["appln_filing_year"]
h01l_map = family_lookup["is_h01l"]
del family_lookup

# ============================================================
# 2. Process tls228 edges: citing family must be H01L
# ============================================================
print("\n[2/3] Processing tls228 edges (citing family = H01L)...")
auth_counts = defaultdict(int)
inv_counts = defaultdict(int)
total_rows = 0
h01l_rows = 0
auth_valid = 0
inv_valid = 0

pf = pq.ParquetFile("../rawdata/tls228.parquet")
n_batches = 0
for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
    chunk = batch.to_pandas()
    total_rows += len(chunk)

    citing_h01l = chunk["docdb_family_id"].map(h01l_map)
    keep = citing_h01l.fillna(False)
    h01l_rows += keep.sum()
    if keep.sum() == 0:
        n_batches += 1
        continue

    sub = chunk[keep.values]
    citing_yr = sub["docdb_family_id"].map(year_map)
    keep2 = citing_yr.notna()

    # --- filing-office version ---
    ca = sub["docdb_family_id"].map(auth_map)
    cb = sub["cited_docdb_family_id"].map(auth_map)
    mask_a = keep2 & ca.notna() & cb.notna()
    auth_valid += mask_a.sum()
    va = pd.DataFrame({
        "citing_country": ca[mask_a].values,
        "cited_country": cb[mask_a].values,
        "year": citing_yr[mask_a].values.astype(int),
    })
    for (c1, c2, yr), n in va.groupby(
        ["citing_country", "cited_country", "year"]
    ).size().items():
        auth_counts[(c1, c2, yr)] += int(n)
    del va, ca, cb

    # --- inventor-country version ---
    ia = sub["docdb_family_id"].map(inv_map)
    ib = sub["cited_docdb_family_id"].map(inv_map)
    mask_i = keep2 & ia.notna() & ib.notna()
    inv_valid += mask_i.sum()
    vi = pd.DataFrame({
        "citing_country": ia[mask_i].values,
        "cited_country": ib[mask_i].values,
        "year": citing_yr[mask_i].values.astype(int),
    })
    for (c1, c2, yr), n in vi.groupby(
        ["citing_country", "cited_country", "year"]
    ).size().items():
        inv_counts[(c1, c2, yr)] += int(n)
    del vi, ia, ib

    n_batches += 1
    if n_batches % 10 == 0:
        print(f"  {n_batches} batches / {total_rows:,} rows → "
              f"{h01l_rows:,} H01L citing edges")

print(f"\n  Done: {total_rows:,} total family edges, {h01l_rows:,} H01L-citing")
print(f"  filing-office mapped: {auth_valid:,} ({auth_valid/h01l_rows*100:.1f}%)")
print(f"  inventor mapped:      {inv_valid:,} ({inv_valid/h01l_rows*100:.1f}%)")

# ============================================================
# 3. Save
# ============================================================
print("\n[3/3] Saving...")

def save(counts, path):
    rows = [{"citing_country": c1, "cited_country": c2, "year": yr,
             "citation_count": n}
            for (c1, c2, yr), n in sorted(
                counts.items(), key=lambda x: x[1], reverse=True)]
    df = pd.DataFrame(rows)
    df = df[(df["year"] >= 1995) & (df["year"] <= 2017)]
    df.to_csv(path, index=False)
    total = df["citation_count"].sum()
    cross = df[df["citing_country"] != df["cited_country"]]["citation_count"].sum()
    print(f"  Saved {len(df):,} rows -> {path}")
    print(f"    H01L citations 1995-2017: {total:,} total, {cross:,} cross-border "
          f"({cross/total*100:.1f}%)")
    return df

auth_df = save(auth_counts, "../processed/semiconductor_citations_family_by_year.csv")
inv_df = save(inv_counts,
              "../processed/semiconductor_citations_family_by_year_inventor.csv")

for label, df in [("filing office", auth_df), ("inventor country", inv_df)]:
    cited = df.groupby("cited_country")["citation_count"].sum().sort_values(ascending=False)
    print(f"\n  Top 10 cited countries ({label}):")
    for c, v in cited.head(10).items():
        print(f"    {c}: {v:>15,}")

print("\nDone.")
