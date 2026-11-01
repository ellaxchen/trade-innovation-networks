"""
Robustness check: does the lead/lag two-pathway finding survive alternative
centrality measures? Re-runs the lead/lag cross-correlation with PageRank
(baseline), betweenness (unweighted, structural), and eigenvector centrality
(where it converges) on family-level data, 1995-2013.

Usage: python centrality_robustness.py
"""
import pandas as pd
import numpy as np
import networkx as nx

YEARS = list(range(1995, 2014))
LAGS = list(range(-5, 6))
KEY_COUNTRIES = ["US", "CN", "JP", "DE", "GB", "FR", "KR"]
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany", "Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[(cite.year >= 1995) & (cite.year <= 2013)]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]
cite = cite[cite.citing_country != cite.cited_country]

MEASURES = ["pagerank", "betweenness", "eigenvector"]

def centrality(G, measure):
    if G.number_of_nodes() == 0:
        return {}
    if measure == "pagerank":
        return nx.pagerank(G, alpha=0.85, weight="weight")
    if measure == "betweenness":
        return nx.betweenness_centrality(G, normalized=True)
    if measure == "eigenvector":
        try:
            return nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return {}  # non-convergence: record as missing
    raise ValueError(measure)

# ---- build per-year graphs and centrality per measure ----
print("[1/2] Computing centrality series per measure...")
series = {m: {"trade": {}, "cite": {}} for m in MEASURES}
failures = {"eigenvector": 0}

for yr in YEARS:
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"]); t = t[t.s != t.t]
    Gt = nx.DiGraph()
    Gt.add_weighted_edges_from(t[["s", "t", "trade_value"]].itertuples(index=False, name=None))

    c = cite[cite.year == yr]
    Gc = nx.DiGraph()
    Gc.add_weighted_edges_from(
        c[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))

    for m in MEASURES:
        series[m]["trade"][yr] = centrality(Gt, m)
        r = centrality(Gc, m)
        if m == "eigenvector" and not r:
            failures["eigenvector"] += 1
        series[m]["cite"][yr] = r

print(f"  eigenvector non-convergence count: {failures['eigenvector']} / {2 * len(YEARS)}")

# ---- lead/lag per country per measure ----
print("\n[2/2] Cross-correlation at lags -5..+5...\n")

def best_lag(ts, cs):
    """Canonical convention: positive = trade leads."""
    best, best_r = 0, 0
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
            r = np.corrcoef(x, y)[0, 1]
            if abs(r) > abs(best_r):
                best, best_r = k, r
    return best, best_r

rows = []
for country in KEY_COUNTRIES:
    name = iso_to_name.get(country, country)
    row = {"country": country, "name": name}
    for m in MEASURES:
        ts = [series[m]["trade"][yr].get(country, 0) for yr in YEARS]
        cs = [series[m]["cite"][yr].get(country, 0) for yr in YEARS]
        bl, br = best_lag(ts, cs)
        row[m] = bl
        row[m + "_r"] = round(br, 3)
    rows.append(row)

print(f"{'Country':<16}{'PageRank':>22}{'Betweenness':>22}{'Eigenvector':>22}")
print(f"{'':16}{'lag':>10}{'r':>12}{'lag':>10}{'r':>12}{'lag':>10}{'r':>12}")
print("-" * 82)
for r in rows:
    print(f"{r['name']:<16}{r['pagerank']:>+10}{r['pagerank_r']:>+12.3f}"
          f"{r['betweenness']:>+10}{r['betweenness_r']:>+12.3f}"
          f"{r['eigenvector']:>+10}{r['eigenvector_r']:>+12.3f}")

# ---- verdict ----
print("\n=== Does the two-pathway finding survive? ===")
print("Pathway agreement: does each measure agree on direction (sign of lag)?")
for r in rows:
    signs = [np.sign(r[m]) for m in MEASURES]
    agree = "YES" if len(set(signs)) == 1 else "NO"
    print(f"  {r['name']:<16} lags {[r[m] for m in MEASURES]}  direction agreement: {agree}")

pd.DataFrame(rows).to_csv("../processed/centrality_robustness.csv", index=False)
print("\nSaved → ../processed/centrality_robustness.csv")
