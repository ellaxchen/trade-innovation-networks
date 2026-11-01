"""
Block bootstrap for the lead/lag cross-correlations.
Resamples years with replacement (moving block, block size 3) to get:
  1. the distribution of the best lag per country (how stable is the lag?)
  2. a 95% percentile CI for r at the sample's best lag (how strong is it?)

Usage: python lag_bootstrap.py [--draws 1000]
"""
import pandas as pd
import numpy as np
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--block", type=int, default=3)
args = parser.parse_args()

YEARS = list(range(1995, 2014))
LAGS = list(range(-5, 6))
KEY_COUNTRIES = ["US", "CN", "JP", "DE", "GB", "FR", "KR"]
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
rng = np.random.default_rng(42)

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany", "Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

print("[1/2] Computing PageRank series 1995-2013...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[(cite.year >= 1995) & (cite.year <= 2013) &
            (cite.citing_country != cite.cited_country)]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]

trade_pr, cite_pr = {}, {}
for yr in YEARS:
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"]); t = t[t.s != t.t]
    G = nx.DiGraph()
    G.add_weighted_edges_from(t[["s", "t", "trade_value"]].itertuples(index=False, name=None))
    trade_pr[yr] = nx.pagerank(G, alpha=0.85, weight="weight")

    c = cite[cite.year == yr]
    G2 = nx.DiGraph()
    G2.add_weighted_edges_from(
        c[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))
    cite_pr[yr] = nx.pagerank(G2, alpha=0.85, weight="weight") if G2.number_of_nodes() else {}

n = len(YEARS)
block = args.block

def best_lag(ts, cs):
    """Canonical: positive lag = trade leads."""
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
    """Resample indices 0..n-1 with moving blocks of size b."""
    idx = []
    while len(idx) < n:
        start = rng.integers(0, n - b + 1)
        idx.extend(range(start, start + b))
    return np.array(idx[:n])

print(f"[2/2] Block bootstrap ({args.draws} draws, block size {block})...\n")

rows = []
for c in KEY_COUNTRIES:
    ts = np.array([trade_pr[y].get(c, 0) for y in YEARS])
    cs = np.array([cite_pr[y].get(c, 0) for y in YEARS])
    sample_lag, sample_r = best_lag(ts, cs)

    lag_counts = {k: 0 for k in LAGS}
    r_at_sample = []
    for _ in range(args.draws):
        yi = moving_block_sample(n, block)
        bts, bcs = ts[yi], cs[yi]
        bl, br = best_lag(bts, bcs)
        lag_counts[bl] += 1
        # r at the SAMPLE lag on the resampled data
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
    p_zero = float(np.mean(r_at_sample <= 0)) if sample_r > 0 else float(np.mean(r_at_sample >= 0))
    # share of draws where the best lag equals the sample lag
    share = lag_counts[sample_lag] / args.draws
    # top-3 lags
    top3 = sorted(lag_counts.items(), key=lambda kv: -kv[1])[:3]

    rows.append({
        "country": c, "name": iso_to_name.get(c, c),
        "sample_lag": sample_lag, "sample_r": round(sample_r, 3),
        "lag_share": round(share, 3),
        "top3": ", ".join(f"{k}({v/args.draws:.0%})" for k, v in top3),
        "r_lo": round(float(lo), 3), "r_hi": round(float(hi), 3),
        "p_signflip": round(p_zero, 3),
    })

print(f"{'Country':<14}{'Lag':>5}{'r':>8}{'lag@draws':>10}{'r 95% CI':>22}{'p(sign flip)':>14}  top-3 lags")
print("-" * 92)
for r in rows:
    ci = f"[{r['r_lo']:+.3f}, {r['r_hi']:+.3f}]".rjust(22)
    print(f"{r['name']:<14}{r['sample_lag']:>+5}{r['sample_r']:>+8.3f}"
          f"{r['lag_share']:>10.0%}{ci}{r['p_signflip']:>14.3f}  {r['top3']}")

pd.DataFrame(rows).to_csv("../processed/lag_bootstrap.csv", index=False)
print("\nSaved → ../processed/lag_bootstrap.csv")
