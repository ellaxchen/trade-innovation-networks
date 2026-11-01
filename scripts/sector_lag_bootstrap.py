"""
Block bootstrap for the Level B sector lead/lag results.
Tests whether sector-level lag signs (e.g., China electronics +3, machinery +2)
survive year resampling — the same uncertainty check applied to the
country-level lags (lag_bootstrap.py), which failed there.

Usage: python sector_lag_bootstrap.py [--draws 500]
Output: ../processed/sector_lag_bootstrap.csv
"""
import pandas as pd
import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=500)
parser.add_argument("--block", type=int, default=3)
parser.add_argument("--top-n", type=int, default=8)
args = parser.parse_args()

SECTORS = {
    "electronics": {"hs2": 85, "cpc": "H", "label": "Electronics"},
    "vehicles":    {"hs2": 87, "cpc": "B", "label": "Vehicles"},
    "pharma":      {"hs2": 30, "cpc": "C", "label": "Pharma"},
    "machinery":   {"hs2": 84, "cpc": "F", "label": "Machinery"},
    "oil_fuels":   {"hs2": 27, "cpc": "B", "label": "Oil & Fuels"},
    "iron_steel":  {"hs2": 72, "cpc": "C", "label": "Iron & Steel"},
    "plastics":    {"hs2": 39, "cpc": "C", "label": "Plastics"},
    "apparel":     {"hs2": 62, "cpc": "D", "label": "Apparel"},
}
YEARS = list(range(1995, 2014))
LAGS = list(range(-5, 6))
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
rng = np.random.default_rng(42)

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"

print("[1/3] Loading trade data (all sectors, 1995-2013)...")
trade = pd.read_parquet("../rawdata/baci_panel_hs6.parquet",
                        columns=["year", "exporter", "importer", "hs2", "trade_value"])
trade = trade[trade["year"].between(1995, 2013)]
trade["source"] = trade["exporter"].map(baci_to_iso)
trade["target"] = trade["importer"].map(baci_to_iso)
trade = trade.dropna(subset=["source", "target"])
trade = trade[trade.source != trade.target]
trade = trade[~trade.source.isin(NON_COUNTRIES) & ~trade.target.isin(NON_COUNTRIES)]
trade = trade.groupby(["year", "hs2", "source", "target"], as_index=False)["trade_value"].sum()
print(f"  {len(trade):,} aggregated sector flows")

print("[2/3] Loading family-level CPC citations...")
cite = pd.read_csv("../processed/country_citation_family_by_cpc.csv")
cite = cite[cite.year.between(1995, 2013)]
cite = cite[cite.citing_country != cite.cited_country]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]

def best_lag(ts, cs):
    best, best_r = 0, 0.0
    for k in LAGS:
        if k > 0:
            x, y = ts[:-k], cs[k:]
        elif k < 0:
            x, y = ts[-k:], cs[:k]
        else:
            x, y = ts, cs
        if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
            r = np.corrcoef(x, y)[0, 1]
            if abs(r) > abs(best_r):
                best, best_r = k, r
    return best, best_r

def moving_block_sample(n, b):
    idx = []
    while len(idx) < n:
        start = rng.integers(0, n - b + 1)
        idx.extend(range(start, start + b))
    return np.array(idx[:n])

print(f"[3/3] Bootstrap per sector ({args.draws} draws)...\n")
rows = []

for sector_key, S in SECTORS.items():
    trade_s = trade[trade["hs2"] == S["hs2"]]
    cite_s = cite[cite["cpc_section"] == S["cpc"]]

    trade_pr, cite_pr = {}, {}
    for yr in YEARS:
        t = trade_s[trade_s["year"] == yr]
        Gt = nx.DiGraph()
        if len(t):
            Gt.add_weighted_edges_from(t[["source", "target", "trade_value"]].itertuples(index=False, name=None))
        trade_pr[yr] = nx.pagerank(Gt, alpha=0.85, weight="weight") if Gt.number_of_nodes() else {}

        c = cite_s[cite_s["year"] == yr]
        Gc = nx.DiGraph()
        if len(c):
            Gc.add_weighted_edges_from(c[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))
        cite_pr[yr] = nx.pagerank(Gc, alpha=0.85, weight="weight") if Gc.number_of_nodes() else {}

    # top countries by combined volume
    tv = trade_s.groupby("source")["trade_value"].sum().add(
        trade_s.groupby("target")["trade_value"].sum(), fill_value=0)
    cv = cite_s.groupby("citing_country")["citation_count"].sum().add(
        cite_s.groupby("cited_country")["citation_count"].sum(), fill_value=0)
    top = tv.add(cv, fill_value=0).sort_values(ascending=False).index[:args.top_n]

    for country in top:
        ts = np.array([trade_pr[y].get(country, 0) for y in YEARS])
        cs = np.array([cite_pr[y].get(country, 0) for y in YEARS])
        if np.std(ts) < 1e-12 or np.std(cs) < 1e-12:
            continue
        sample_lag, sample_r = best_lag(ts, cs)
        if abs(sample_r) < 0.5:
            continue  # only test meaningful correlations

        lag_counts = {k: 0 for k in LAGS}
        r_at_sample = []
        for _ in range(args.draws):
            yi = moving_block_sample(len(YEARS), args.block)
            bts, bcs = ts[yi], cs[yi]
            bl, _ = best_lag(bts, bcs)
            lag_counts[bl] += 1
            k = sample_lag
            if k > 0:
                x, y = bts[:-k], bcs[k:]
            elif k < 0:
                x, y = bts[-k:], bcs[:k]
            else:
                x, y = bts, bcs
            if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
                r_at_sample.append(np.corrcoef(x, y)[0, 1])

        r_at_sample = np.array(r_at_sample)
        lo, hi = np.percentile(r_at_sample, [2.5, 97.5])
        share = lag_counts[sample_lag] / args.draws
        rows.append({
            "sector": sector_key, "sector_label": S["label"], "country": country,
            "sample_lag": sample_lag, "sample_r": round(sample_r, 3),
            "lag_share": round(share, 3),
            "r_lo": round(float(lo), 3), "r_hi": round(float(hi), 3),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

print(f"{'Sector':<14}{'Country':<10}{'Lag':>5}{'r':>8}{'lag@draws':>10}{'r 95% CI':>20}  survives?")
print("-" * 82)
for r in rows:
    ci = f"[{r['r_lo']:+.3f}, {r['r_hi']:+.3f}]".rjust(20)
    surv = "YES" if (r["lag_share"] >= 0.5 and r["ci_excludes_zero"]) else "no"
    print(f"{r['sector_label']:<14}{r['country']:<10}{r['sample_lag']:>+5}{r['sample_r']:>+8.3f}"
          f"{r['lag_share']:>10.0%}{ci}  {surv}")

out = pd.DataFrame(rows)
out.to_csv("../processed/sector_lag_bootstrap.csv", index=False)
n_surv = sum(1 for r in rows if r["lag_share"] >= 0.5 and r["ci_excludes_zero"])
print(f"\n  {n_surv}/{len(rows)} country-sector lag findings survive resampling")
print(f"Saved → ../processed/sector_lag_bootstrap.csv")
