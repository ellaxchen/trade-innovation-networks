"""
Per-country crossing analysis: which line (trade PR vs cite PR) moves first,
where they cross, and who ends on top. Family-level citations, 1995-2013.
"""
import pandas as pd
import numpy as np
import networkx as nx

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Rep. of Korea": "South Korea",
         "Russian Federation": "Russia"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[(cite.year >= 1995) & (cite.year <= 2013) &
            (cite.citing_country != cite.cited_country)]

trade_pr, cite_pr = {}, {}
for yr in range(1995, 2014):
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"])
    t = t[t.s != t.t]
    G = nx.DiGraph()
    for _, r in t.iterrows():
        w = G.get_edge_data(r["s"], r["t"], {}).get("weight", 0) + r["trade_value"]
        G.add_edge(r["s"], r["t"], weight=w)
    trade_pr[yr] = nx.pagerank(G, alpha=0.85, weight="weight")

    c = cite[cite.year == yr]
    G2 = nx.DiGraph()
    for _, r in c.iterrows():
        w = G2.get_edge_data(r["citing_country"], r["cited_country"], {}).get("weight", 0) + r["citation_count"]
        G2.add_edge(r["citing_country"], r["cited_country"], weight=w)
    cite_pr[yr] = nx.pagerank(G2, alpha=0.85, weight="weight")

top = ["US", "DE", "CN", "FR", "GB", "JP", "CA", "IT",
       "MX", "NL", "BE", "HK", "ES", "KR", "CH", "SG"]

print(f"{'Country':<16}{'1st cross':<11}{'Start side':<12}{'End side':<11}{'Who rises 1st':<15}")
print("-" * 65)
for c in top:
    name = iso_to_name.get(c, c)
    ts = [trade_pr[y].get(c, 0) for y in range(1995, 2014)]
    cs = [cite_pr[y].get(c, 0) for y in range(1995, 2014)]
    tn = [v / max(ts) if max(ts) > 0 else 0 for v in ts]
    cn = [v / max(cs) if max(cs) > 0 else 0 for v in cs]
    diffs = [tn[i] - cn[i] for i in range(19)]

    first_cross = "-"
    for i in range(1, 19):
        if diffs[i - 1] * diffs[i] < 0:
            first_cross = str(1995 + i)
            break

    side = lambda d: "trade" if d > 0.02 else ("cite" if d < -0.02 else "tied")
    start, end = side(diffs[0]), side(diffs[-1])

    t_rise = 1995 + int(np.argmax(np.diff(tn)))
    c_rise = 1995 + int(np.argmax(np.diff(cn)))
    moves = "trade" if t_rise < c_rise else ("cite" if c_rise < t_rise else "together")

    print(f"{name:<16}{first_cross:<11}{start:<12}{end:<11}{moves:<15}")
