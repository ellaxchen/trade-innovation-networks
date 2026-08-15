"""
Lead/lag analysis: does trade centrality lead or lag knowledge centrality?

Loads year-filtered citation data and multi-year trade data, computes
network centrality for each year in the overlap window (1995-2013), and
checks the cross-correlation structure for key countries.

Usage: python lead_lag_analysis.py
"""
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict

OVERLAP_YEARS = list(range(1995, 2014))  # stop at 2013 (last reliable year)
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
KEY_COUNTRIES = ["US", "CN", "JP", "DE", "GB", "FR", "KR"]

# ---- load country mappings ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
# Clean historical names for display
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia",
         "Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

# ---- load year-filtered citation data ----
print("Loading year-filtered citation edges...")
cite_all = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite_all = cite_all[~cite_all["citing_country"].isin(NON_COUNTRIES)
                    & ~cite_all["cited_country"].isin(NON_COUNTRIES)]
# Filter to overlap window
cite_all = cite_all[cite_all["year"].between(OVERLAP_YEARS[0], OVERLAP_YEARS[-1])]
print(f"  {len(cite_all):,} citation edges in overlap window ({OVERLAP_YEARS[0]}-{OVERLAP_YEARS[-1]})")

# ---- build yearly networks and compute centrality ----
trade_centrality = defaultdict(dict)  # trade_centrality[year][country] = value
cite_centrality = defaultdict(dict)   # cite_centrality[year][country] = value

for year in OVERLAP_YEARS:
    # --- Trade network ---
    try:
        trade = pd.read_csv(f"../processed/trade_edges_{year}.csv")
    except FileNotFoundError:
        # Generate on the fly if year not in snapshots
        continue

    Gt = nx.DiGraph()
    for _, r in trade.iterrows():
        iso_e = name_to_iso.get(r["exporter_name"]) or baci_to_iso.get(r["exporter"])
        iso_i = name_to_iso.get(r["importer_name"]) or baci_to_iso.get(r["importer"])
        if iso_e and iso_i:
            Gt.add_edge(iso_e, iso_i, weight=r["trade_value"])

    if Gt.number_of_nodes() == 0:
        continue
    pr_trade = nx.pagerank(Gt, weight="weight")
    for c, v in pr_trade.items():
        trade_centrality[year][c] = v

    # --- Citation network ---
    cite_yr = cite_all[cite_all["year"] == year]
    Gc = nx.DiGraph()
    for _, r in cite_yr.iterrows():
        Gc.add_edge(r["citing_country"], r["cited_country"], weight=r["citation_count"])

    if Gc.number_of_nodes() == 0:
        continue
    pr_cite = nx.pagerank(Gc, weight="weight")
    for c, v in pr_cite.items():
        cite_centrality[year][c] = v

# ---- build time series for key countries ----
print("\n=== Lead/Lag Analysis: PageRank Centrality ===\n")

for country in KEY_COUNTRIES:
    trade_ts = []
    cite_ts = []
    yrs = []
    for y in OVERLAP_YEARS:
        tv = trade_centrality.get(y, {}).get(country)
        cv = cite_centrality.get(y, {}).get(country)
        if tv is not None and cv is not None:
            trade_ts.append(tv)
            cite_ts.append(cv)
            yrs.append(y)

    if len(yrs) < 5:
        continue

    # Normalize to z-scores for comparability
    tz = (np.array(trade_ts) - np.mean(trade_ts)) / np.std(trade_ts)
    cz = (np.array(cite_ts) - np.mean(cite_ts)) / np.std(cite_ts)

    # Cross-correlation: trade vs. citation at various lags
    # Positive lag = trade leads citation (trade at t, citation at t+lag)
    # Negative lag = citation leads trade (citation at t, trade at t+|lag|)
    name = iso_to_name.get(country, country)
    print(f"--- {name} ({country}) ---")
    print(f"  Years: {yrs[0]}-{yrs[-1]} ({len(yrs)} observations)")

    print(f"  Lag  | Correlation | Interpretation")
    print(f"  -----|-------------|---------------")
    results = []
    for lag in range(-5, 6):
        if lag < 0:
            # Citation leads trade: cite[t+lag] vs trade[t]
            corr = np.corrcoef(cz[:lag], tz[-lag:])[0, 1] if abs(lag) < len(yrs) else np.nan
        elif lag > 0:
            # Trade leads citation: trade[t] vs cite[t+lag]
            corr = np.corrcoef(tz[:-lag], cz[lag:])[0, 1] if lag < len(yrs) else np.nan
        else:
            corr = np.corrcoef(tz, cz)[0, 1]

        if not np.isnan(corr):
            results.append((lag, corr))
            direction = "trade LEADS knowledge" if lag > 0 else ("knowledge LEADS trade" if lag < 0 else "synchronous")
            print(f"  {lag:>+4}  |   {corr:>+7.4f}    | {direction}")

    if results:
        best_lag, best_corr = max(results, key=lambda x: abs(x[1]))
        direction = "TRADE LEADS knowledge" if best_lag > 0 else ("KNOWLEDGE LEADS trade" if best_lag < 0 else "SYNCHRONOUS")
        print(f"  => Best fit: lag={best_lag:+d} (r={best_corr:.4f}) — {direction}")
    print()

# ---- overall correlation ----
print("=== Summary: Best-Fit Lag by Country ===\n")
print(f"  {'Country':<20} {'Best Lag':<10} {'Correlation':<12} {'Interpretation'}")
print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*25}")
for country in KEY_COUNTRIES:
    trade_ts = []
    cite_ts = []
    yrs = []
    for y in OVERLAP_YEARS:
        tv = trade_centrality.get(y, {}).get(country)
        cv = cite_centrality.get(y, {}).get(country)
        if tv is not None and cv is not None:
            trade_ts.append(tv)
            cite_ts.append(cv)
            yrs.append(y)

    if len(yrs) < 5:
        continue

    tz = (np.array(trade_ts) - np.mean(trade_ts)) / np.std(trade_ts)
    cz = (np.array(cite_ts) - np.mean(cite_ts)) / np.std(cite_ts)

    best_lag, best_corr = 0, np.corrcoef(tz, cz)[0, 1]
    for lag in range(-5, 6):
        if lag < 0 and abs(lag) < len(yrs):
            corr = np.corrcoef(cz[:lag], tz[-lag:])[0, 1]
        elif lag > 0 and lag < len(yrs):
            corr = np.corrcoef(tz[:-lag], cz[lag:])[0, 1]
        else:
            continue
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag

    name = iso_to_name.get(country, country)
    direction = "Trade leads ->" if best_lag > 0 else ("<- Knowledge leads" if best_lag < 0 else "Synchronous")
    print(f"  {name:<20} {best_lag:+d}         {best_corr:+.4f}        {direction}")

print("\nNote: positive lag = trade centrality changes first, then knowledge follows.")
print("      negative lag = knowledge centrality changes first, then trade follows.")
