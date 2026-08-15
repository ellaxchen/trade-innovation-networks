"""
Level B lead/lag analysis: does trade centrality lead or lag knowledge
centrality WITHIN a sector? Runs the lead/lag machinery from
lead_lag_analysis.py on industry-filtered edges.

For each sector (HS2 trade chapter ↔ CPC citation section):
  1. Build trade edges for each year 1995-2013, filtered to the HS2 chapter
  2. Build citation edges for each year 1995-2013, filtered to the CPC section
     (family-level data)
  3. Compute PageRank (alpha=0.85) for both networks per year
  4. Cross-correlate each country's trade PR vs cite PR at lags -5..+5
  5. Report best lag per country — does China's dual-rise survive at sector level?

Usage:
  python level_b_lead_lag.py                        # 4 default sectors
  python level_b_lead_lag.py --sector electronics   # single sector
  python level_b_lead_lag.py --list-sectors
"""
import pandas as pd
import numpy as np
import networkx as nx
import argparse
import sys
from math import pi
import matplotlib.pyplot as plt

SECTORS = {
    "electronics": {"hs2": 85, "cpc": "H", "label": "Electronics (HS85 ↔ CPC H)"},
    "vehicles":    {"hs2": 87, "cpc": "B", "label": "Vehicles (HS87 ↔ CPC B)"},
    "pharma":      {"hs2": 30, "cpc": "C", "label": "Pharma (HS30 ↔ CPC C)"},
    "machinery":   {"hs2": 84, "cpc": "F", "label": "Machinery (HS84 ↔ CPC F)"},
}

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
ALPHA = 0.85
YEARS = list(range(1995, 2014))  # citations stop at 2013
LAGS = list(range(-5, 6))
KEY_COUNTRIES = ["CN", "US", "DE", "JP", "KR", "FR", "GB", "TW"]

parser = argparse.ArgumentParser()
parser.add_argument("--sector", type=str, default=None)
parser.add_argument("--list-sectors", action="store_true")
args = parser.parse_args()

if args.list_sectors:
    for k, s in SECTORS.items():
        print(f"  {k:<14} {s['label']}")
    sys.exit(0)

sectors_to_run = ([args.sector] if args.sector else list(SECTORS.keys()))
for s in sectors_to_run:
    if s not in SECTORS:
        print(f"Unknown sector '{s}'. Use --list-sectors.")
        sys.exit(1)

# ---- country codes ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia",
         "Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

# ============================================================
# 1. Load trade data once, aggregate per (year, hs2, source, target)
# ============================================================
print("[1/3] Loading trade data (all years, all sectors)...")
trade_cols = ["year", "exporter", "importer", "hs2", "trade_value"]
trade_raw = pd.read_parquet("../rawdata/baci_panel_hs6.parquet", columns=trade_cols)
trade_raw = trade_raw[trade_raw["year"].between(1995, 2013)]
trade_raw["source"] = trade_raw["exporter"].map(baci_to_iso)
trade_raw["target"] = trade_raw["importer"].map(baci_to_iso)
trade_raw["source"] = trade_raw["source"].fillna(
    trade_raw["exporter"].map(dict(zip(cc["country_code"], cc["country_name"]))).map(name_to_iso))
trade_raw["target"] = trade_raw["target"].fillna(
    trade_raw["importer"].map(dict(zip(cc["country_code"], cc["country_name"]))).map(name_to_iso))
trade_raw = trade_raw.dropna(subset=["source", "target"])
trade_raw = trade_raw[~trade_raw["source"].isin(NON_COUNTRIES) &
                      ~trade_raw["target"].isin(NON_COUNTRIES)]
trade_raw = trade_raw[trade_raw["source"] != trade_raw["target"]]
# Aggregate to (year, hs2, source, target)
trade_agg = trade_raw.groupby(
    ["year", "hs2", "source", "target"], as_index=False)["trade_value"].sum()
del trade_raw
print(f"  {len(trade_agg):,} aggregated sector flows")

# ============================================================
# 2. Load family-level CPC citations once
# ============================================================
print("[2/3] Loading family-level CPC citations...")
cite_all = pd.read_csv("../processed/country_citation_family_by_cpc.csv")
cite_all = cite_all[cite_all["year"].between(1995, 2013)]
cite_all = cite_all[~cite_all["citing_country"].isin(NON_COUNTRIES) &
                    ~cite_all["cited_country"].isin(NON_COUNTRIES)]
cite_all = cite_all[cite_all["citing_country"] != cite_all["cited_country"]]
print(f"  {len(cite_all):,} cross-border citation rows")

# ============================================================
# 3. Per-sector lead/lag
# ============================================================
print("[3/3] Computing sector lead/lag...\n")

summary_rows = []

for sector_key in sectors_to_run:
    S = SECTORS[sector_key]
    hs2, cpc = S["hs2"], S["cpc"]

    # Trade edges for this HS2
    trade_s = trade_agg[trade_agg["hs2"] == hs2]
    # Citation edges for this CPC
    cite_s = cite_all[cite_all["cpc_section"] == cpc]

    # PageRank per year (vectorized edge building — no iterrows)
    trade_pr = {}
    cite_pr = {}
    for yr in YEARS:
        t_yr = trade_s[trade_s["year"] == yr]
        if len(t_yr):
            Gt = nx.DiGraph()
            Gt.add_weighted_edges_from(
                t_yr[["source", "target", "trade_value"]].itertuples(index=False, name=None))
            trade_pr[yr] = nx.pagerank(Gt, alpha=ALPHA, weight="weight")
        else:
            trade_pr[yr] = {}

        c_yr = cite_s[cite_s["year"] == yr]
        if len(c_yr):
            Gc = nx.DiGraph()
            Gc.add_weighted_edges_from(
                c_yr[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))
            cite_pr[yr] = nx.pagerank(Gc, alpha=ALPHA, weight="weight")
        else:
            cite_pr[yr] = {}

    # Rank countries by combined trade + citation VOLUME (not year-presence,
    # which ties at the max for nearly everyone and picks arbitrarily)
    trade_vol = trade_s.groupby("source")["trade_value"].sum().add(
        trade_s.groupby("target")["trade_value"].sum(), fill_value=0)
    cite_vol = cite_s.groupby("citing_country")["citation_count"].sum().add(
        cite_s.groupby("cited_country")["citation_count"].sum(), fill_value=0)
    combined = trade_vol.add(cite_vol, fill_value=0)
    # Keep only countries with enough year coverage in BOTH networks
    enough_coverage = {}
    for c in combined.index:
        t_cnt = sum(1 for yr in YEARS if c in trade_pr.get(yr, {}))
        c_cnt = sum(1 for yr in YEARS if c in cite_pr.get(yr, {}))
        if t_cnt >= 5 and c_cnt >= 5:
            enough_coverage[c] = True
    top_countries = [c for c in combined.sort_values(ascending=False).index
                     if c in enough_coverage][:10]

    print(f"=== {S['label']} ===")
    print(f"  Trade edges: {len(trade_s):,}  |  Citation edges: {len(cite_s):,}")
    print(f"  Top {len(top_countries)} countries in both networks")
    print(f"  {'Country':<18}{'Best lag':<10}{'r':>8}  Interpretation")

    sector_table = []
    for country in top_countries:
        ts = [trade_pr[yr].get(country, 0) for yr in YEARS]
        cs = [cite_pr[yr].get(country, 0) for yr in YEARS]
        if np.std(ts) < 1e-12 or np.std(cs) < 1e-12:
            continue

        corrs = {}
        for k in LAGS:
            # canonical convention (lead_lag_analysis.py):
            # positive k = corr(trade[t], cite[t+k]) = trade leads
            # negative k = corr(cite[t], trade[t+|k|]) = knowledge leads
            if k > 0:
                x = np.array(ts[:-k]); y = np.array(cs[k:])
            elif k < 0:
                x = np.array(ts[-k:]); y = np.array(cs[:k])
            else:
                x = np.array(ts); y = np.array(cs)
            if len(x) >= 10:
                corrs[k] = np.corrcoef(x, y)[0, 1]

        best_lag = max(corrs, key=lambda k: abs(corrs[k]))
        r = corrs[best_lag]
        if r > 0:
            interp = "trade leads" if best_lag > 0 else ("knowledge leads" if best_lag < 0 else "synchronous")
        else:
            interp = "opposite movement" if abs(r) > 0.4 else "weak/no signal"

        name = iso_to_name.get(country, country)
        print(f"  {name:<18}{best_lag:>+5}    {r:>+8.3f}  {interp}")
        sector_table.append({"sector": sector_key, "country": country,
                             "name": name, "best_lag": best_lag, "r": r})
        summary_rows.append((sector_key, country, name, best_lag, r))
    print()

# ============================================================
# 4. Summary: China across sectors
# ============================================================
print("=== SUMMARY: does the country-level result survive at sector level? ===")
print(f"  Country-level (Step 6): CN knowledge-leads-trade at -5 (r=0.990)")
print()
cn_rows = [r for r in summary_rows if r[0:2] == ("electronics", "CN")]
all_cn = [r for r in summary_rows if r[1] == "CN"]
print(f"  China by sector:")
for r in all_cn:
    sector_label = SECTORS[r[0]]["label"]
    print(f"    {sector_label:<34} lag={r[3]:+d}  r={r[4]:+.3f}")

print()
print("=== Key questions answered ===")
for r in summary_rows:
    if r[1] in KEY_COUNTRIES and abs(r[4]) >= 0.5:
        sector_label = SECTORS[r[0]]["label"]
        print(f"  {r[2]:<14} in {sector_label:<30} lag={r[3]:+d} r={r[4]:+.3f}")

# Save summary CSV
out = pd.DataFrame(summary_rows, columns=["sector", "country", "name", "best_lag", "r"])
out.to_csv("../processed/level_b_lead_lag_summary.csv", index=False)
print(f"\nSaved → ../processed/level_b_lead_lag_summary.csv")
