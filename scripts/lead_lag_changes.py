"""
Lead/lag on year-over-year CHANGES, not levels.

Level-based cross-correlation between trade and citation PageRank is dominated
by shared smooth trends — both China series rise monotonically, so they
correlate at r ~ 0.99 at EVERY lag and the "best lag" (which lands at the
window edge, ±5) is an artifact of the window, not evidence of timing.

Fix: cross-correlate first differences. A lead now means: changes in one
network's centrality anticipate changes in the other's.

Canonical convention (matches lead_lag_analysis.py):
  positive lag = trade leads (corr(trade[t], cite[t+lag]))
  negative lag = knowledge leads

Usage:
  python lead_lag_changes.py             # country level (CSVs only, fast)
  python lead_lag_changes.py --sectors   # 8 sectors (loads BACI parquet)
"""
import pandas as pd
import numpy as np
import networkx as nx
import argparse
import sys

YEARS = list(range(1995, 2014))
ALPHA = 0.85
LAGS = list(range(-5, 6))
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

SECTORS = {
    "electronics": {"hs2": 85, "cpc": "H", "label": "Electronics (HS85 ↔ CPC H)"},
    "vehicles":    {"hs2": 87, "cpc": "B", "label": "Vehicles (HS87 ↔ CPC B)"},
    "pharma":      {"hs2": 30, "cpc": "C", "label": "Pharma (HS30 ↔ CPC C)"},
    "machinery":   {"hs2": 84, "cpc": "F", "label": "Machinery (HS84 ↔ CPC F)"},
    "oil_fuels":   {"hs2": 27, "cpc": "B", "label": "Oil & Fuels (HS27 ↔ CPC B)"},
    "iron_steel":  {"hs2": 72, "cpc": "C", "label": "Iron & Steel (HS72 ↔ CPC C)"},
    "plastics":    {"hs2": 39, "cpc": "C", "label": "Plastics (HS39 ↔ CPC C)"},
    "apparel":     {"hs2": 62, "cpc": "D", "label": "Apparel (HS62 ↔ CPC D)"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--sectors", action="store_true")
args = parser.parse_args()

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia",
         "Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new


def crosscorr(ts, cs, changes=False):
    """Best lag + r for two series. changes=True → first-difference first.

    Canonical convention: k>0 → corr(trade[t], cite[t+k]) = trade leads.
    """
    if changes:
        ts, cs = np.diff(ts), np.diff(cs)
    best, best_r = 0, np.nan
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(best_r) or abs(r) > abs(best_r):
                best, best_r = k, r
    return best, best_r


def correlogram(ts, cs, changes=False):
    """Full r-by-lag profile (for the China figure)."""
    if changes:
        ts, cs = np.diff(ts), np.diff(cs)
    out = {}
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def page_rank_series(edge_dfs_by_year, weight_col, src_col, dst_col):
    """edge_dfs_by_year: dict year -> DataFrame. Returns dict year -> {node: PR}."""
    pr = {}
    for yr, df in edge_dfs_by_year.items():
        if df is None or len(df) == 0:
            pr[yr] = {}
            continue
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            df[[src_col, dst_col, weight_col]].itertuples(index=False, name=None))
        pr[yr] = nx.pagerank(G, alpha=ALPHA, weight="weight")
    return pr


def run_country():
    print("[1/2] Loading family-level citations...")
    cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
    cite = cite[(cite.year >= 1995) & (cite.year <= 2013)]
    cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]
    cite = cite[cite.citing_country != cite.cited_country]

    print("[2/2] Loading trade snapshots + computing PageRank series...")
    trade_by_yr, cite_by_yr = {}, {}
    trade_vol = pd.Series(dtype=float)
    for yr in YEARS:
        t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
        t["s"] = t["exporter_name"].map(name_to_iso)
        t["t"] = t["importer_name"].map(name_to_iso)
        t = t.dropna(subset=["s", "t"]); t = t[t.s != t.t]
        trade_by_yr[yr] = t[["s", "t", "trade_value"]]
        trade_vol = trade_vol.add(t.groupby("s")["trade_value"].sum(), fill_value=0)
        trade_vol = trade_vol.add(t.groupby("t")["trade_value"].sum(), fill_value=0)
        cite_by_yr[yr] = cite[cite.year == yr][["citing_country", "cited_country", "citation_count"]]

    trade_pr = page_rank_series(trade_by_yr, "trade_value", "s", "t")
    cite_pr = page_rank_series(cite_by_yr, "citation_count", "citing_country", "cited_country")

    # Country selection: same combined-volume top-16 as the paper
    cite_vol = cite.groupby("citing_country")["citation_count"].sum().add(
        cite.groupby("cited_country")["citation_count"].sum(), fill_value=0)
    combined = trade_vol.add(cite_vol, fill_value=0)
    top = combined.sort_values(ascending=False).index[:16].tolist()

    rows = []
    print(f"\n{'Country':<16}{'level lag':>10}{'level r':>9}{'Δ lag':>7}{'Δ r':>8}")
    print("-" * 52)
    for c in top:
        ts = [trade_pr[y].get(c, 0) for y in YEARS]
        cs = [cite_pr[y].get(c, 0) for y in YEARS]
        if np.std(ts) < 1e-12 or np.std(cs) < 1e-12:
            continue
        ll, lr = crosscorr(ts, cs, changes=False)
        dl, dr = crosscorr(ts, cs, changes=True)
        name = iso_to_name.get(c, c)
        print(f"{name:<16}{ll:>+10}{lr:>+9.3f}{dl:>+7}{dr:>+8.3f}")
        rows.append({"country": c, "name": name,
                     "level_lag": ll, "level_r": round(lr, 3),
                     "change_lag": dl, "change_r": round(dr, 3)})

    out = pd.DataFrame(rows)
    out.to_csv("../processed/lead_lag_changes_country.csv", index=False)
    print(f"\nSaved -> ../processed/lead_lag_changes_country.csv")

    # China correlogram for the paper figure
    cn = out[out.country == "CN"]
    if len(cn):
        c = "CN"
        ts = [trade_pr[y].get(c, 0) for y in YEARS]
        cs = [cite_pr[y].get(c, 0) for y in YEARS]
        lev = correlogram(ts, cs, changes=False)
        chg = correlogram(ts, cs, changes=True)
        print("\n=== China correlogram (r by lag) ===")
        print(f"  {'lag':>5} {'levels':>10} {'changes':>10}")
        for k in LAGS:
            print(f"  {k:>+5} {lev.get(k, np.nan):>+10.3f} {chg.get(k, np.nan):>+10.3f}")


def run_sectors():
    print("[1/4] Loading trade data (all years, all sectors)...")
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
    trade_agg = trade_raw.groupby(
        ["year", "hs2", "source", "target"], as_index=False)["trade_value"].sum()
    del trade_raw
    print(f"  {len(trade_agg):,} aggregated sector flows")

    print("[2/4] Loading family-level CPC citations...")
    cite_all = pd.read_csv("../processed/country_citation_family_by_cpc.csv")
    cite_all = cite_all[cite_all["year"].between(1995, 2013)]
    cite_all = cite_all[~cite_all["citing_country"].isin(NON_COUNTRIES) &
                        ~cite_all["cited_country"].isin(NON_COUNTRIES)]
    cite_all = cite_all[cite_all["citing_country"] != cite_all["cited_country"]]

    summary_rows = []
    print("\n[3/4] Per-sector changes-based lead/lag...\n")
    for sector_key, S in SECTORS.items():
        hs2, cpc = S["hs2"], S["cpc"]
        trade_s = trade_agg[trade_agg["hs2"] == hs2]
        cite_s = cite_all[cite_all["cpc_section"] == cpc]

        trade_pr, cite_pr = {}, {}
        for yr in YEARS:
            t_yr = trade_s[trade_s["year"] == yr]
            trade_pr[yr] = page_rank_series({yr: t_yr}, "trade_value", "source", "target")[yr]
            c_yr = cite_s[cite_s["year"] == yr]
            cite_pr[yr] = page_rank_series({yr: c_yr}, "citation_count",
                                           "citing_country", "cited_country")[yr]

        trade_vol = trade_s.groupby("source")["trade_value"].sum().add(
            trade_s.groupby("target")["trade_value"].sum(), fill_value=0)
        cite_vol = cite_s.groupby("citing_country")["citation_count"].sum().add(
            cite_s.groupby("cited_country")["citation_count"].sum(), fill_value=0)
        combined = trade_vol.add(cite_vol, fill_value=0)
        enough_coverage = {}
        for c in combined.index:
            t_cnt = sum(1 for yr in YEARS if c in trade_pr.get(yr, {}))
            c_cnt = sum(1 for yr in YEARS if c in cite_pr.get(yr, {}))
            if t_cnt >= 5 and c_cnt >= 5:
                enough_coverage[c] = True
        top_countries = [c for c in combined.sort_values(ascending=False).index
                         if c in enough_coverage][:10]

        print(f"=== {S['label']} ===")
        print(f"  {'Country':<18}{'level lag':>10}{'level r':>9}{'Δ lag':>7}{'Δ r':>8}")
        for country in top_countries:
            ts = [trade_pr[yr].get(country, 0) for yr in YEARS]
            cs = [cite_pr[yr].get(country, 0) for yr in YEARS]
            if np.std(ts) < 1e-12 or np.std(cs) < 1e-12:
                continue
            ll, lr = crosscorr(ts, cs, changes=False)
            dl, dr = crosscorr(ts, cs, changes=True)
            name = iso_to_name.get(country, country)
            print(f"  {name:<18}{ll:>+10}{lr:>+9.3f}{dl:>+7}{dr:>+8.3f}")
            summary_rows.append({"sector": sector_key, "country": country, "name": name,
                                 "level_lag": ll, "level_r": round(lr, 3),
                                 "change_lag": dl, "change_r": round(dr, 3)})
        print()

    out = pd.DataFrame(summary_rows)
    out.to_csv("../processed/level_b_lead_lag_changes_summary.csv", index=False)
    print(f"[4/4] Saved -> ../processed/level_b_lead_lag_changes_summary.csv")


if __name__ == "__main__":
    run_sectors() if args.sectors else run_country()
