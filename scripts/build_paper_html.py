"""
Build the HTML version of the research paper with embedded interactive charts.
Computes the PageRank series live (family-level, 1995-2013) and embeds:
  F1: 16-panel small multiples (trade vs knowledge centrality, RAW values)
  F3: China 2010 artifact (app vs family citing)
  F3b: China 2010 mechanism (tls212 CN citations by publication year x origin;
       tls201 CN applications, smooth)
  F4: lead/lag diverging bars (levels)
  F4b: China correlogram, levels vs changes
  F4c: lead/lag diverging bars (year-over-year changes)
  F5: WTO DiD bars
  F6: sector lags
  F7: semiconductor bump chart (knowledge rank vs trade rank)
  F8: semiconductor knowledge balance (citations received / made)
plus the paper's tables. Output: ../report/paper.html (local-only).

Usage: python build_paper_html.py
"""
import pandas as pd
import numpy as np
import networkx as nx
import json
import os

YEARS = list(range(1995, 2014))
ALPHA = 0.85
LAGS = list(range(-5, 6))
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
WATCH = {"KR", "TW"}

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia",
         "Rep. of Korea": "South Korea",
         "China, Hong Kong SAR": "Hong Kong"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

print("[1/4] Computing PageRank series (raw values)...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[(cite.year >= 1995) & (cite.year <= 2013) &
            (cite.citing_country != cite.cited_country)]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]

trade_pr, cite_pr = {}, {}
trade_vol = pd.Series(dtype=float)
for yr in YEARS:
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"]); t = t[t.s != t.t]
    G = nx.DiGraph()
    G.add_weighted_edges_from(t[["s", "t", "trade_value"]].itertuples(index=False, name=None))
    trade_pr[yr] = nx.pagerank(G, alpha=ALPHA, weight="weight")
    trade_vol = trade_vol.add(t.groupby("s")["trade_value"].sum(), fill_value=0)
    trade_vol = trade_vol.add(t.groupby("t")["trade_value"].sum(), fill_value=0)

    c = cite[cite.year == yr]
    G2 = nx.DiGraph()
    G2.add_weighted_edges_from(
        c[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))
    cite_pr[yr] = nx.pagerank(G2, alpha=ALPHA, weight="weight") if G2.number_of_nodes() else {}

cite_vol = cite.groupby("citing_country")["citation_count"].sum().add(
    cite.groupby("cited_country")["citation_count"].sum(), fill_value=0)
combined = trade_vol.add(cite_vol, fill_value=0)
top = combined.sort_values(ascending=False).index[:16].tolist()


def best_lag(ts, cs):
    """Canonical convention: positive = trade leads."""
    best, best_r = 0, 0.0
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


def correlogram(ts, cs, changes=False):
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


records = []
cn_ts = cn_cs = None
for c in top:
    ts = [trade_pr[y].get(c, 0) for y in YEARS]
    cs = [cite_pr[y].get(c, 0) for y in YEARS]
    if c == "CN":
        cn_ts, cn_cs = ts, cs
    diffs = [ts[i] - cs[i] for i in range(len(YEARS))]
    avg_diff = float(np.mean(diffs))
    trend = float(np.mean(diffs[-5:]) - np.mean(diffs[:5]))
    pathway = "Trade-leading" if avg_diff > 0.02 else ("Knowledge-leading" if avg_diff < -0.02 else "Balanced")
    bl, br = best_lag(ts, cs)
    records.append({
        "code": c, "name": iso_to_name.get(c, c),
        "trade_series": [round(v, 4) for v in ts],
        "cite_series": [round(v, 4) for v in cs],
        "avg_diff": round(avg_diff, 3), "trend": round(trend, 3),
        "pathway": pathway, "best_lag": bl,
        "best_r": round(br, 3),
    })
order = {"Knowledge-leading": 0, "Balanced": 1, "Trade-leading": 2}
records.sort(key=lambda r: (order[r["pathway"]], -abs(r["avg_diff"])))

print("[2/4] Loading changes-based lead/lag + China tabulation...")
chg = pd.read_csv("../processed/lead_lag_changes_country.csv")
chg_map = {(r.country): (int(r.change_lag), float(r.change_r)) for r in chg.itertuples()}
changes = [{"code": r["code"], "name": r["name"],
            "lag": chg_map.get(r["code"], (0, 0.0))[0],
            "r": chg_map.get(r["code"], (0, 0.0))[1],
            "weak": abs(chg_map.get(r["code"], (0, 0.0))[1]) < 0.4}
           for r in records]

china_correlogram = {"levels": correlogram(cn_ts, cn_cs, False),
                     "changes": correlogram(cn_ts, cn_cs, True)}

# ---- sector lags: full 8-sector grid for four countries from the summary CSV ----
level_lag_map = {(r.country): int(r.level_lag) for r in chg.itertuples()}
sec_df = pd.read_csv("../processed/level_b_lead_lag_summary.csv")
sector_lags = []
for c, name in [("CN", "China"), ("US", "USA"), ("GB", "UK"), ("DE", "Germany")]:
    row = {"name": name, "country": level_lag_map.get(c, 0)}
    for _, r in sec_df[sec_df["country"] == c].iterrows():
        row[r["sector"]] = int(r["best_lag"])
    sector_lags.append(row)

# ---- China 2010 payload (tabulated from raw tls212/tls201) ----
origin = pd.read_csv("../processed/citation_origin_tabulation.csv")
cn_o = origin[origin["publn_country"] == "CN"]
yrs10 = list(range(2005, 2015))
tls212_cn = [int(cn_o[cn_o["year"] == y]["citations"].sum()) for y in yrs10]
sea = [int(cn_o[(cn_o["year"] == y) & (cn_o["citn_origin"] == "SEA")]["citations"].sum()) for y in yrs10]
other = [tls212_cn[i] - sea[i] for i in range(len(yrs10))]
apps10_df = pd.read_csv("../processed/cn_appln_counts_by_year.csv")
apps10 = [int(apps10_df[apps10_df["year"] == y]["cn_applications"].iloc[0]) if (apps10_df["year"] == y).any() else 0 for y in yrs10]

print("[3/4] Loading semiconductor analysis...")
semi_panel = pd.read_csv("../processed/semiconductor_analysis.csv")
cite_s = pd.read_csv("../processed/semiconductor_citations_family_by_year.csv")
cite_s = cite_s[cite_s["year"].between(1995, 2013)]
cite_s = cite_s[~cite_s["citing_country"].isin(NON_COUNTRIES) &
                ~cite_s["cited_country"].isin(NON_COUNTRIES)]
cite_s = cite_s[cite_s["citing_country"] != cite_s["cited_country"]]
trade_s = pd.read_csv("../processed/semiconductor_trade_edges_by_year.csv")
trade_s = trade_s[trade_s["year"].between(1995, 2013)]
trade_s["source"] = trade_s["exporter"].map(baci_to_iso)
trade_s["target"] = trade_s["importer"].map(baci_to_iso)
trade_s = trade_s.dropna(subset=["source", "target"])
trade_s = trade_s[~trade_s["source"].isin(NON_COUNTRIES) &
                  ~trade_s["target"].isin(NON_COUNTRIES)]
trade_s = trade_s[trade_s["source"] != trade_s["target"]]

cite_vol_s = cite_s.groupby("citing_country")["citation_count"].sum().add(
    cite_s.groupby("cited_country")["citation_count"].sum(), fill_value=0)
trade_vol_s = trade_s.groupby("source")["trade_value"].sum().add(
    trade_s.groupby("target")["trade_value"].sum(), fill_value=0)
combined_s = cite_vol_s.add(trade_vol_s, fill_value=0).sort_values(ascending=False)
top12 = list(combined_s.index[:12])
# keep only countries with real presence in BOTH networks (HK's trade volume is
# entrepôt and it barely appears in H01L citations)
presence_s = {}
for c in combined_s.index:
    k_yrs = semi_panel[(semi_panel["country"] == c) & semi_panel["knowledge_rank"].notna()]["year"].nunique()
    t_yrs = semi_panel[(semi_panel["country"] == c) & semi_panel["trade_rank"].notna()]["year"].nunique()
    presence_s[c] = min(k_yrs, t_yrs)
qualified_s = [c for c in top12 if presence_s.get(c, 0) >= 12]
colored8 = [c for c in qualified_s if c not in WATCH][:6] + [c for c in WATCH if c in qualified_s]
gray = [c for c in top12 if c not in colored8]

semi_colored, semi_gray = [], []
for c in colored8 + gray:
    sub = semi_panel[semi_panel["country"] == c].sort_values("year")
    entry = {"code": c, "name": iso_to_name.get(c, c),
             "knowledge_ranks": [None if pd.isna(v) else int(v) for v in sub["knowledge_rank"]],
             "trade_ranks": [None if pd.isna(v) else int(v) for v in sub["trade_rank"]],
             "ratio": [None if pd.isna(v) else round(float(v), 3) for v in sub["cite_ratio"]],
             "export_share_2013": round(float(sub[sub["year"] == 2013]["export_share"].iloc[0] * 100), 1) if len(sub[sub["year"] == 2013]) else None,
             "cite_ratio_2013": round(float(sub[sub["year"] == 2013]["cite_ratio"].iloc[0]), 2) if len(sub[sub["year"] == 2013]) and not pd.isna(sub[sub["year"] == 2013]["cite_ratio"].iloc[0]) else None,
             }
    (semi_colored if c in colored8 else semi_gray).append(entry)

# Spearman between the two rank orders, per year: full sample AND top-15 tier
spearman = []
if os.path.exists("../processed/semiconductor_spearman.csv"):
    sp_df = pd.read_csv("../processed/semiconductor_spearman.csv")
    for _, r in sp_df.iterrows():
        spearman.append({"year": int(r["year"]),
                         "rho": round(float(r["spearman_full"]), 3),
                         "rho_top": round(float(r["spearman_top15"]), 3) if not pd.isna(r["spearman_top15"]) else None,
                         "n": int(r["n_full"]), "n_top": int(r["n_top15"]) if not pd.isna(r["n_top15"]) else None})

# 2013 snapshot table (top 12 by knowledge rank, restricted to the displayed set)
snap = semi_panel[semi_panel["year"] == 2013].dropna(subset=["knowledge_rank", "trade_rank"]).sort_values("knowledge_rank")
snapshot = []
for _, r in snap.iterrows():
    if r["country"] in colored8 + gray:
        snapshot.append({
            "name": iso_to_name.get(r["country"], r["country"]),
            "knowledge_rank": int(r["knowledge_rank"]),
            "trade_rank": int(r["trade_rank"]),
            "export_share": round(float(r["export_share"] * 100), 1),
            "cite_ratio": round(float(r["cite_ratio"]), 2) if not pd.isna(r["cite_ratio"]) else None,
        })

# 2013 facts for the narrative
k13 = {r["country"]: r for _, r in snap.iterrows()}
facts = {}
for c in ["US", "KR", "TW", "CN", "JP", "DE"]:
    if c in k13:
        r = k13[c]
        facts[c] = f"{iso_to_name.get(c, c)}: knowledge rank {int(r['knowledge_rank'])}, trade rank {int(r['trade_rank'])}, export share {r['export_share']*100:.1f}%, cite ratio {r['cite_ratio']:.2f}" if not pd.isna(r["cite_ratio"]) else ""
for c in ["US", "KR", "TW", "CN", "JP", "DE"]:
    if c in facts:
        print(f"  {facts[c]}")
rho13 = spearman[-1]["rho"] if spearman else None
rho13_top = spearman[-1]["rho_top"] if spearman else None
print(f"  Spearman 2013: full {rho13}, top-15 {rho13_top}")

data = {
    "years": YEARS,
    "countries": records,
    "changes": changes,
    "china_correlogram": china_correlogram,
    "china2010": {
        "years": yrs10,
        "app": [155, 371, 729, 1161, 2148, 1215925, 1757994, 2264775, 2623369, 2687733],
        "fam": [254080, 188646, 95440, 83302, 105760, 866011, 1291250, 1784663, 2184803, 2345886],
        "tls212_cn": tls212_cn,
        "sea": sea,
        "other": other,
        "cn_apps": apps10,
    },
    "sector_lags": sector_lags,
    "wto": {
        "into": {"cn": 1.225, "control": 0.429, "did": 0.796, "t": 13.4},
        "out": {"cn": 1.097, "control": 0.336, "did": 0.760, "t": 10.3},
    },
    "robustness": [
        {"name": "USA", "pagerank": 1, "betweenness": 3, "eigenvector": 5, "robust": True},
        {"name": "Japan", "pagerank": 5, "betweenness": 2, "eigenvector": 5, "robust": True},
        {"name": "UK", "pagerank": -5, "betweenness": -2, "eigenvector": -1, "robust": True},
        {"name": "South Korea", "pagerank": -4, "betweenness": -4, "eigenvector": -3, "robust": True},
        {"name": "China", "pagerank": -5, "betweenness": 3, "eigenvector": -2, "robust": False},
        {"name": "Germany", "pagerank": 0, "betweenness": 1, "eigenvector": -5, "robust": False},
        {"name": "France", "pagerank": -5, "betweenness": 3, "eigenvector": 0, "robust": False},
    ],
    "bootstrap": [
        {"name": "USA", "lag": 1, "r": 0.849, "share": 0.02, "lo": 0.110, "hi": 0.813},
        {"name": "China", "lag": -5, "r": 0.997, "share": 0.00, "lo": -0.594, "hi": 0.408},
        {"name": "UK", "lag": -5, "r": 0.836, "share": 0.00, "lo": -0.596, "hi": 0.399},
        {"name": "South Korea", "lag": -4, "r": 0.906, "share": 0.03, "lo": -0.581, "hi": 0.448},
        {"name": "Japan", "lag": 5, "r": -0.723, "share": 0.08, "lo": -0.509, "hi": 0.608},
        {"name": "Germany", "lag": 0, "r": 0.554, "share": 0.32, "lo": -0.382, "hi": 0.735},
        {"name": "France", "lag": -5, "r": 0.632, "share": 0.07, "lo": -0.525, "hi": 0.506},
    ],
    "semi": {
        "years": YEARS,
        "colored": semi_colored,
        "gray": semi_gray,
        "spearman": spearman,
        "snapshot": snapshot,
        "watch": sorted(WATCH),
        "rho_2013": rho13,
        "rho_top_2013": rho13_top,
    },
}
json_str = json.dumps(data)
print("[4/4] Generating paper HTML...")

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Goods Before Ideas? Trade Centrality, Knowledge Centrality, and the China Puzzle</title>
<style>
:root{
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
  --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --neutral:#898781;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);color:var(--text-primary);line-height:1.65}
.paper{max-width:880px;margin:0 auto;padding:44px 24px 90px}
h1{font-size:1.55rem;font-weight:700;letter-spacing:-0.01em;line-height:1.3;margin-bottom:10px}
.byline{color:var(--text-secondary);font-size:0.9rem;margin-bottom:26px}
h2{font-size:1.18rem;font-weight:700;margin:38px 0 10px;padding-top:8px;border-top:1px solid var(--grid)}
h3{font-size:1.0rem;font-weight:700;margin:18px 0 6px}
p{margin:8px 0}
.abstract{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:14px 0;font-size:0.93rem}
.abstract h2{border:none;margin:0 0 8px;padding:0;font-size:0.85rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text-secondary)}
.viz{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px 18px 12px;margin:16px 0}
.viz .cap{font-size:0.82rem;color:var(--text-secondary);margin:6px 2px 0}
.viz .cap b{color:var(--text-primary)}
.legend{display:flex;gap:16px;align-items:center;margin:4px 2px 10px;font-size:0.82rem;color:var(--text-secondary);flex-wrap:wrap}
.swatch{width:16px;height:4px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:9px;padding:8px 10px 6px}
.panel h3{font-size:0.8rem;margin:0 0 1px}
.panel .tag{font-size:0.63rem;font-weight:700;padding:1px 6px;border-radius:20px;display:inline-block;margin-bottom:3px}
.tag.k{background:rgba(235,104,52,0.14);color:#c14e1e}
.tag.t{background:rgba(42,120,214,0.14);color:#1c5cab}
.tag.b{background:rgba(137,135,129,0.16);color:#52514e}
.panel svg{width:100%;display:block}
svg.viz{width:100%;height:auto;display:block}
.tooltip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--page);padding:7px 10px;border-radius:7px;font-size:0.78rem;opacity:0;transition:opacity .12s;z-index:60;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:0.86rem;margin:10px 0}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600;font-size:0.78rem;border-bottom:2px solid var(--baseline)}
td.num{font-variant-numeric:tabular-nums}
.neg{color:#d03b3b}.pos{color:#0ca30c}
.callout{border-left:4px solid var(--series-2);background:rgba(235,104,52,0.06);border-radius:0 10px 10px 0;padding:12px 16px;margin:14px 0;font-size:0.92rem}
.muted{color:var(--text-secondary)}.small{font-size:0.84rem}
.refs{font-size:0.84rem}.refs li{margin:4px 0}
.minigrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
</style>
</head>
<body>
<div class="paper">
<h1>Goods Before Ideas? Trade Centrality, Knowledge Centrality, and the China Puzzle</h1>
<div class="byline">Working draft · 2026-08-30 · Family-level citations (PATSTAT tls228) · cross-border only · PageRank α = 0.85 · citation window 1995–2013</div>

<div class="abstract">
<h2>Abstract</h2>
<p>Do countries first exchange goods, then ideas — or ideas, then goods? We construct two directed, weighted country networks for 1995–2013: international trade (BACI) and knowledge flows (patent citations from PATSTAT, aggregated to the DOCDB family level so each invention is counted once). Two measurement corrections shape every result. First, China's application-level citation counts jump 566× in 2010 — the moment the Chinese patent office began sharing its examiners' citations with the EPO, which a raw tabulation of tls212 shows directly (CN search-report citations go from ~0 in 2009 to 2.8 million by 2013, while CN applications in tls201 grow smoothly throughout). It is an administrative data exchange, not an innovation surge. Second, family-level counting raises the cross-border share of citations from 11.3% to 40.5% by removing multi-filing inflation. Raw PageRank centrality — comparable across networks because each sums to one — separates three pathways: knowledge-first economies (US: 16.8% of trade centrality vs 27.9% of knowledge centrality in 2013; Japan), trade-first economies (China, Netherlands, Mexico, and most of continental Europe), and balanced economies (Korea, Germany, the UK). Timing, however, resists identification: level correlations are trend-dominated (China's two series correlate at r ≥ 0.94 at <em>every</em> lag), and on year-over-year changes no country shows a lead in either direction — co-movement is contemporaneous. Two causal probes (China's 2001 WTO accession; regional trade agreement entries) find no evidence that trade policy shocks accelerate knowledge flows. A semiconductor case study (HS 8541/42 vs CPC H01L) shows why the question matters: the trade and knowledge hierarchies agree broadly (ρ ≈ 0.7) but diverge at the top tier (ρ ≈ 0.5) — the US and Japan own cited knowledge while China, Korea, and Taiwan dominate exports — and Taiwan's knowledge balance crosses, and Korea's arrives at, the net-source frontier. Goods and ideas move together in time; the data do not support a trade-to-knowledge causal channel at the policy margin.</p>
</div>

<h2>1. Introduction</h2>
<p>The relationship between international trade and knowledge flows is one of the oldest questions in development economics, and one of the least measured directly. Theories cut both ways: countries may learn by exporting — entering world markets forces firms to absorb foreign technology and standards — or they may first build knowledge and only then trade sophisticated goods. Cross-country correlations cannot separate these channels because both are driven by common growth.</p>
<p>This paper approaches the question from network structure. We build two directed networks on the same 190+ countries — a trade network whose edges are export flows, and a knowledge network whose edges are patent-citation flows — and ask a simple ordering question for each country: <strong>which network moves first?</strong> We measure centrality with PageRank — robust on directed networks with dominant hubs and one-way structures, where eigenvector centrality degenerates — and cross-correlate the two yearly series at lags of −5 to +5 years, on levels and on year-over-year changes.</p>

<h2>2. Data and Network Construction</h2>
<p>Trade flows come from BACI (CEPII): 269.9 million product-level rows for 1995–2024, aggregated to country-pair-year level. The trade network grew from $3.2T (1995) to $12.4T (2024) in top-200 flows, but its structure barely changed — density stayed at 0.08–0.10 for three decades; the network thickened rather than widened.</p>
<p>Citations come from PATSTAT (EPO, spring 2018). The canonical construction uses <strong>tls228</strong>, the pre-built DOCDB family-to-family edge list: 155.9 million edges between 67.8 million families. Each family is one invention, assigned the country and year of its earliest filing. Family-level counting removes multi-filing inflation: the cross-border share of citations rises from 11.3% (application level) to <strong>40.5%</strong> (family level), and total citations fall from ~257M to 98M. Country assignment uses the filing authority (<code>appln_auth</code>) as a proxy — a known approximation flagged throughout. PATSTAT coverage ends in 2017, but recent years are truncated; all time series stop at <strong>2013</strong>, the last fully reliable year.</p>

<h2>3. Two Measurement Corrections</h2>
<h3>3.1 The China 2010 discontinuity</h3>
<p>Application-level citation counts for China jump from 2,148 (2009) to 1,215,925 (2010) — a 566-fold increase that reads like an innovation explosion. It is not one.</p>
<div class="viz" style="position:relative">
  <div class="legend">
    <span><span class="swatch" style="background:var(--series-1)"></span>Application-level (tls212)</span>
    <span><span class="swatch" style="background:var(--series-2)"></span>Family-level (tls228)</span>
  </div>
  <svg id="china2010" viewBox="0 0 720 340" role="img" aria-label="China citing totals by year, application versus family level"></svg>
  <div class="cap"><b>Figure 2.</b> China's citing totals, 2005–2014. The 566× jump exists only at the application level; the family level shows 100–250K/year all along.</div>
</div>
<p><strong>Where the discontinuity comes from.</strong> Chinese documents were indexed in PATSTAT all along — CN applications in tls201 grow smoothly from 60,000 (1995) to 579,000 (2009) with no break at 2010 (right panel below). What enters the dataset in 2010 is Chinese <em>citations</em>. Tabulating tls212 by the publication year of the citing CN publication (left panel): essentially zero before 2010 (7 rows in 2009), then 324,421 in 2010 and 2.8 million by 2013. All of them are search-report citations (<code>citn_origin = SEA</code>), and the first wave is tagged <code>citn_gener_auth = CN</code> — the signature of the Chinese office (CNIPA) beginning to share its examiners' citation records with the EPO in 2010. An administrative data exchange, not an innovation surge. The US control shows no such break (7.7M in 2005, 12.4M in 2010).</p>
<div class="viz">
  <div class="minigrid">
    <div>
      <svg id="chinaMechanismCites" viewBox="0 0 340 240" role="img" aria-label="CN citations in tls212 by publication year and origin"></svg>
      <div class="cap">tls212 citations by CN publication year, stacked by citn_origin (SEA vs other). Zero until 2010.</div>
    </div>
    <div>
      <svg id="chinaMechanismApps" viewBox="0 0 340 240" role="img" aria-label="CN applications in tls201 by filing year"></svg>
      <div class="cap">tls201 CN applications by filing year — smooth growth, no 2010 break. The discontinuity lives in tls212, not tls201.</div>
    </div>
  </div>
  <div class="cap" style="margin-top:8px"><b>Figure 2b.</b> The 2010 mechanism, split by table. Citations: absent, then all search-report (SEA). Applications: smooth throughout.</div>
</div>
<p>The family-level view (Figure 2) is smooth through 2010 because it has always seen Chinese inventions through their foreign family members — the ~25% of Chinese inventions also filed in a US, EP, or JP office, whose examiner citations were in the dataset all along. That is why family-level counting is canonical for every time series in this paper.</p>
<div class="callout"><strong>Consequence:</strong> any application-level time series that treats 2010 as real growth is wrong — including lead-lag analyses concluding China's trade led its knowledge.</div>

<h3>3.2 Multi-filing inflation</h3>
<p>At the application level, 88.7% of citations stay domestic; at the family level, 59.5% do. The difference is mechanical — one invention filed ten times domestically generates up to 100 internal citation pairs that collapse to one family-level pair. Japan is the most affected.</p>

<h2>4. Centrality: Why PageRank, and What It Shows</h2>
<h3>4.1 Measure choice</h3>
<p>Eigenvector-style centrality degenerates on directed networks with dead ends or one-way structure — both present here (the US is a near-sink in trade; the citation graph is a near-DAG). In a five-node network where four countries point to a hub that points nowhere, eigenvector centrality converges to hub = 0.999999 and everyone else = 0.000558; PageRank (α = 0.85) gives hub = 0.524 and the others = 0.119 each. The damping factor keeps every node measurable — so every centrality result here uses PageRank.</p>
<p><strong>Edge direction and interpretation.</strong> Trade edges run exporter → importer, so trade PageRank measures importance as an <em>import destination</em> (corr with imports +0.989 vs exports +0.916); citation edges run citing → cited, so citation PageRank measures importance as a <em>cited knowledge source</em>. Both are attractiveness measures — who the world ships goods to, and whose knowledge it builds on. China is the top exporter yet ranks below the US on trade PageRank; Hong Kong's rank reflects its entrepôt role. The pairing is deliberate: comparable network-position measures, each weighting its receiving side.</p>
<p><strong>Raw values, not per-series maxima.</strong> Because PageRank sums to one within each network, the raw values are directly comparable across the trade and citation graphs — the gap between a country's two shares is itself the finding (the US holds 16.8% of trade centrality against 27.9% of knowledge centrality in 2013; Japan 4.5% vs 15.1%). All panels below therefore plot raw PageRank on a shared per-country scale; nothing is normalized to its own maximum.</p>
<h3>4.2 Three development pathways</h3>
<div class="viz">
  <div class="legend">
    <span><span class="swatch" style="background:var(--series-1)"></span>Trade PageRank</span>
    <span><span class="swatch" style="background:var(--series-2)"></span>Citation PageRank</span>
    <span class="small muted">Raw values, one shared scale across all panels (each network's PR sums to ~1). Hover for values.</span>
  </div>
  <div class="grid" id="grid"></div>
  <div class="cap"><b>Figure 1.</b> Per-country PageRank panels, 1995–2013 (cross-border, raw). All panels share one scale, so vertical gaps between the lines are true PageRank-share differences — small countries with small gaps stay visually close. The divergences and crossings are the findings.</div>
</div>
<table id="pathway-table">
  <tr><th>Pathway</th><th>Countries (avg raw difference)</th><th>Pattern</th></tr>
</table>
<p><strong>Korea</strong> is the convergence to watch: its knowledge centrality (4.0% in 2013) already exceeds its trade centrality (2.7%), and the gap is still widening (trend −0.022). The earlier reading of Korea as the widest trade lead was a normalization artifact — relative to its own maximum its trade line looked high; in raw shares it is already on the knowledge side. <strong>Japan</strong> holds the largest knowledge margin among the majors (15.1% vs 4.5% in 2013) and it is still widening (trend −0.027), with the two series moving in opposite directions (best-lag r = −0.72).</p>

<h2>5. Lead-Lag: Which Network Moves First</h2>
<div class="viz">
  <svg id="lagbars" viewBox="0 0 760 360" role="img" aria-label="Best lag by country, levels"></svg>
  <div class="cap"><b>Figure 3.</b> Best cross-correlation lag per country on levels (positive = trade leads). Bar length = |lag|; label shows r. Orange = knowledge-first, blue = trade-first. Levels are trend-dominated — see Figure 3b.</div>
</div>
<p><strong>Level correlations are trend-dominated.</strong> For countries on shared growth paths, both series rise smoothly and correlate highly at every lag — so the "best lag" is not evidence of timing. China is the extreme case: its two centralities climb almost monotonically, correlating at r ≥ 0.94 at <em>every</em> lag from −5 to +5, and the best fit (−5, r = 0.997) simply sits at the window edge.</p>
<div class="viz">
  <div class="legend">
    <span><span class="swatch" style="background:var(--series-1)"></span>Levels</span>
    <span><span class="swatch" style="background:var(--series-2)"></span>Year-over-year changes</span>
  </div>
  <svg id="correlogram" viewBox="0 0 760 300" role="img" aria-label="China correlogram: correlation by lag, levels vs changes"></svg>
  <div class="cap"><b>Figure 3b.</b> China's cross-correlogram. Levels: r ≈ 0.95–1.00 flat across all lags (pure trend). Changes: r ≈ 0.1–0.6 and flat — no lag improves on contemporaneous co-movement. The dip below zero at +5 is r = −0.10 from just 13 year-pairs — noise, not signal: the changes bootstrap CIs include zero at every lag. No lead either way.</div>
</div>
<p><strong>Correlate changes instead.</strong> Differencing removes the trend and asks whether a <em>change</em> in one network anticipates a change in the other. China's changes correlogram is essentially flat (r = 0.57 contemporaneous vs 0.59 at −5; the −0.10 at +5 is a 13-pair estimate, indistinguishable from zero): there is no clear lead in either direction — consistent with the dual-rise reading, in which both centralities climb together. The changes-based best lags across all countries (Figure 3c) tell the same story: point estimates scatter across the window with moderate r and unstable signs.</p>
<div class="viz">
  <svg id="changelagbars" viewBox="0 0 760 380" role="img" aria-label="Best lag by country, year-over-year changes"></svg>
  <div class="cap"><b>Figure 3c.</b> Best cross-correlation lag per country on year-over-year changes. Pale bars: |r| &lt; 0.4. The ordering is noise: in a block bootstrap on the changes (1,000 draws), the sample lag is reproduced at most 44% of draws (Singapore), and every r CI includes zero.</div>
</div>
<p><strong>The changes-based ordering is unidentifiable.</strong> A block bootstrap (1,000 draws, moving blocks of 3 years) on the differenced series: China's −5 is reproduced in 4% of draws while lag 0 wins 62%, with r's CI spanning −0.61 to +0.52. No country does better — the best reproduction rate anywhere is Singapore's 44% (lag +1, r = −0.61, CI [−0.77, +0.13]). On levels the same machinery shows the ordering is trend-driven, not timing (below): the co-movement is real, but the lag is not.</p>
<table id="boot-table">
  <tr><th>Country</th><th>Sample lag</th><th>r</th><th>Lag reproduced</th><th>r 95% CI at sample lag</th></tr>
</table>
<p class="small muted">In every bootstrap draw, China's best lag is 0 — the −5 alignment is an artifact of particular year sequences. Only the US trade-side correlation survives resampling, and barely (CI lower bound +0.11). The pathway typology (Figure 1) is the robust finding; lag signs are suggestive at best.</p>
<h3>Robustness across centrality measures</h3>
<table id="rob-table">
  <tr><th>Country</th><th>PageRank</th><th>Betweenness</th><th>Eigenvector</th><th>Direction robust?</th></tr>
</table>
<p class="small muted">The knowledge-first result for the UK and Korea, and the trade-side result for the US and Japan, survive alternative measures. China's correlation is near-perfect under every measure (r = 0.997/0.927/0.984) but its lag sign flips with unweighted betweenness; Germany and France are not robust.</p>

<h2>6. Level B: The Sector-Level Picture</h2>
<p>Running the same machinery on industry × country networks (HS2 trade chapters ↔ CPC citation sections, hand-picked concordance, eight sectors) produces a different picture for China:</p>
<table>
  <tr><th>China sector</th><th>Best lag</th><th>r</th><th>Direction</th></tr>
  <tr><td>Electronics (HS85 ↔ CPC H)</td><td class="num">+3</td><td class="num">0.960</td><td>trade leads</td></tr>
  <tr><td>Machinery (HS84 ↔ CPC F)</td><td class="num">+2</td><td class="num">0.869</td><td>trade leads</td></tr>
  <tr><td>Plastics (HS39 ↔ CPC C)</td><td class="num">+1</td><td class="num">0.962</td><td>trade leads</td></tr>
  <tr><td>Oil &amp; Fuels (HS27 ↔ CPC B)</td><td class="num">−5</td><td class="num">0.900</td><td>knowledge leads</td></tr>
  <tr><td>Apparel (HS62 ↔ CPC D)</td><td class="num">−5</td><td class="num">0.776</td><td>knowledge leads</td></tr>
  <tr><td>Iron &amp; Steel (HS72 ↔ CPC C)</td><td class="num">−5</td><td class="num">−0.556</td><td>opposite movement</td></tr>
</table>
<div class="viz">
  <div class="legend">
    <span><span class="swatch" style="background:var(--series-2)"></span>Knowledge leads (negative lag)</span>
    <span><span class="swatch" style="background:var(--series-1)"></span>Trade leads (positive lag)</span>
    <span><span class="swatch" style="background:#898781"></span>Synchronous (0)</span>
    <span class="small muted">Faint dot on the 0 line = no estimate (country outside that sector's top-10 by volume)</span>
  </div>
  <svg id="sectorlags" viewBox="0 0 760 300" role="img" aria-label="Sector-level lags"></svg>
  <div class="cap"><b>Figure 4.</b> Best lag by level and sector for four countries. China is sector-heterogeneous: trade led knowledge in manufacturing (electronics, machinery, plastics), knowledge led in resources and textiles (oil, apparel). Subject to the same bootstrap caveat as Section 5.</div>
</div>
<p>The <strong>US</strong> is trade-first in electronics (+2), vehicles (+1) and machinery (+3), but <strong>knowledge-first in pharma (−5, r = 0.519)</strong>. <strong>Germany</strong> is trade-first in vehicles (+5, r = 0.879) but knowledge-first in electronics (−5) and machinery (−3, r = 0.961). The <strong>UK</strong> is consistently knowledge-first: vehicles (−5, r = 0.927), pharma (−4, r = 0.812), machinery (−5, r = 0.977). <strong>Taiwan</strong> shows opposite movement in electronics (r = −0.963). On year-over-year changes the sector lags collapse exactly as at country level — no sector's changes-lag for China is distinguishable from zero (see <code>processed/level_b_lead_lag_changes_summary.csv</code>).</p>
<div class="callout"><strong>Sector bootstrap verdict:</strong> of 56 country-sector lag findings, only 3 survive resampling — and all three are <em>synchronous</em> (lag 0: machinery–NL, plastics–DE, plastics–NL). Every non-zero sector lag, including China's electronics +3 and machinery +2, collapses (0% reproduction; CIs include zero). Trade and knowledge centrality co-move contemporaneously; no lead-lag ordering survives at any level. The raw sector lags above are descriptive patterns, not findings.</div>
<p class="small muted">Sector chord diagrams for all eight concordances: <code>html/chord_level_b_*_standalone.html</code>.</p>

<h2>7. Causal Probe: China's WTO Accession</h2>
<p>China joined the WTO on December 11, 2001. An independent audit of our first design found it wanting, and this section reflects the corrected design. Three caveats frame everything. <strong>Timing:</strong> the US applied MFN rates to China continuously since 1980 under Jackson-Vanik waivers — PNTR (Oct 2000, effective Jan 2002) removed renewal <em>uncertainty</em>, not tariff levels; China's own cuts were pre-announced in the 1999–2001 agreements. <strong>Outcome:</strong> family-level Chinese citation counts rise ~4× from 1997–2001 to 2003–2007 for coverage/stock reasons unrelated to tariffs. <strong>Treatment:</strong> the BACI tariff columns have zero post-2003 coverage for China as importer and no coverage for the US/EU/India/Brazil as partners — the tariff dose-response is unmeasured with this data and is reported as a negative result.</p>
<p>Corrected DiD: within each partner, we compare the growth of its knowledge flows with China against its flows with three mature hubs (US, Japan, Germany), pre (1997–2001) vs post (2003–2007), self-citations and the CN self-pair excluded:</p>
<div class="viz">
  <svg id="wtobars" viewBox="0 0 720 300" role="img" aria-label="WTO diff-in-diff results"></svg>
  <div class="cap"><b>Figure 5.</b> Log-point growth in knowledge flows around China's WTO accession, with hub-based controls. DiD = +0.796 (into China, t=13.4) and +0.760 (out of China, t=10.3). Descriptive, not causal — the outcome window sits inside the coverage ramp.</div>
</div>
<p><strong>Verdict: descriptive support, no identified causal effect.</strong> Knowledge flows on China pairs outgrew their hub benchmarks after accession — consistent with, but not evidence for, export-led learning. Clean identification requires external tariff data (WITS/TRAINS) or an alternative design (RTA entry, instrumental variables).</p>

<h3>7.2 Regional trade agreements as shocks (null result)</h3>
<p>As an alternative shock, we use RTA entries: 2,183 directed pairs enter a regional trade agreement between 1998 and 2008 (5,109 of 43,482 pairs are ever in an RTA). Comparing knowledge-flow growth on event pairs against 38,373 never-RTA pairs matched on pre-period citation volume:</p>
<table>
  <tr><th>Group</th><th>n</th><th>Growth (log-points)</th></tr>
  <tr><td>RTA event pairs</td><td class="num">2,183</td><td class="num">+0.092 (median 0.000)</td></tr>
  <tr><td>Matched controls</td><td class="num">663</td><td class="num">+0.590 (median +0.701)</td></tr>
  <tr><td><strong>DiD</strong></td><td class="num"></td><td class="num"><strong>−0.498 (t = −19.3)</strong></td></tr>
</table>
<p><strong>RTA entry predicts no additional knowledge-flow growth — if anything, less.</strong> Countries sign RTAs with partners they are already deeply integrated with; the agreement formalizes existing relationships rather than creating new knowledge links. China's 28 post-WTO RTA events show mean growth +0.535, but post-2005 outcomes sit inside the coverage ramp (upper bound).</p>
<div class="callout"><strong>The two designs tell a consistent story:</strong> trade levels and knowledge flows co-move strongly, but trade <em>policy shocks</em> (tariff cuts, RTA entries) do not accelerate knowledge flows. The co-movement in Sections 4–6 likely reflects common drivers — growth, size, development — rather than a causal trade-to-knowledge channel at the policy margin.</div>

<h2>8. Case Study: Semiconductors — Who Is in the Lead?</h2>
<p>One last exercise: pick a contentious industry and ask who is in the lead. Semiconductors are the natural choice — the canonical assembly-versus-design industry — measured as <strong>HS 8541</strong> (diodes, transistors, semiconductor devices) + <strong>HS 8542</strong> (integrated circuits) on the trade side and CPC class <strong>H01L</strong> on the knowledge side, at the family level. For each country-year we compute three measures: the <em>knowledge rank</em> (PageRank on the H01L citation network — cited centrality: whose semiconductor prior art the world builds on), the <em>trade rank</em> (PageRank on the 8541/42 export network with reversed edges — export centrality: who the world buys chips from, weighted by buyers' importance), and the <em>knowledge balance</em> (H01L citations received ÷ made).</p>
<div class="viz">
  <div class="legend">
    <span><span class="swatch" style="background:#898781"></span>Thin gray: next tier</span>
    <span class="small muted">Bold lines = Korea and Taiwan (the watch list). Hover for values.</span>
  </div>
  <svg id="semiBump" viewBox="0 0 760 500" role="img" aria-label="Semiconductor rank bump chart: knowledge vs trade"></svg>
  <div class="cap"><b>Figure 6.</b> Semiconductor rank bump chart, 1995–2013. Top left: knowledge rank (H01L citation PageRank). Top right: trade rank (8541/42 export PageRank). Rank 1 = top; lines leaving the panel have ranks beyond 15 — their final ranks are listed in the "off-panel" note under each panel. Bottom: Spearman ρ between the two rank orders — broad agreement across all countries, much weaker agreement within the top-15 tier, where the assembly/design split lives.</div>
</div>
<p><strong>The two hierarchies agree broadly — and diverge exactly where the assembly/design split predicts.</strong> Across all countries the Spearman ρ between the trade and knowledge rank orders sits at 0.6–0.76 in every year: big economies rank high in both networks. Within the top-15 tier the agreement is much weaker (ρ ≈ 0.3–0.55): Malaysia and the Philippines are trade-only, Australia and Austria knowledge-only, and the top-10 sets overlap by just 7 of 10. The trajectories tell the story. The <strong>US</strong> holds the #1 knowledge rank in every year while its trade rank slides from #1 to #3 (8.6% of exports by 2013); <strong>Japan</strong> mirrors it, #2 in knowledge all 19 years while slipping to #5 in trade. <strong>China</strong> rockets from #19 to #1 in trade (13.9% by 2013) while its knowledge rank reaches only #4. <strong>Korea and Taiwan are the convergence</strong>: Korea's knowledge rank climbs 5 → 3 while holding top-tier exports (11.3%); Taiwan's rises 15 → 6 while its export rank reaches #2 — the largest share anywhere (16.6%).</p>
<div class="viz">
  <div class="legend" id="semi-ratio-legend"></div>
  <svg id="semiRatio" viewBox="0 0 760 300" role="img" aria-label="Semiconductor knowledge balance: citations received over made"></svg>
  <div class="cap"><b>Figure 7.</b> H01L knowledge balance (citations received ÷ made, cross-border), log₂ scale. Above 1 = net knowledge source; below 1 = net absorber. Bold = Korea and Taiwan. Ratios on small citation bases (e.g., Singapore, whose H01L volume is an order of magnitude below the majors) are noisy — read the six majors for levels.</div>
</div>
<p>The balance completes the picture with a twist. <strong>Japan</strong> is the clear net source, and rising (citations received ÷ made: 2.1 → 3.6). The <strong>US</strong> sits below 1 (0.79 → 0.51) — the largest source <em>and</em> the largest absorber of semiconductor prior art, citing slightly more than it is cited within H01L. <strong>Korea and Taiwan</strong> started as net absorbers and converged to the frontier: Taiwan crossed it (0.19 → 1.82), Korea arrived at it (0.31 → 0.97). China remains a net absorber (0.52). The assembly economies are beginning to own the ideas their chips embody — but the knowledge pole of the system still sits in Tokyo and Silicon Valley.</p>
<table id="semi-table">
  <tr><th>Country (2013)</th><th>Knowledge rank</th><th>Trade rank</th><th>Export share</th><th>Citations received / made</th></tr>
</table>
<p class="small muted">Small-base ratios (Singapore) are noisy; read the majors for balance levels.</p>

<h2>9. Discussion</h2>
<p><strong>Three pathways, one mechanism?</strong> The typology maps onto development stages: mature knowledge economies (US, Japan) sit above their trade lines; trade-hub economies (Netherlands, Belgium, Mexico, Hong Kong) sit below; late industrializers show the fastest transitions. Korea's rapid knowledge catch-up (trend −0.022 on raw shares) suggests the transition can happen within a decade — in raw levels Korea has already crossed, its knowledge centrality above its trade centrality since the mid-2000s.</p>
<p><strong>China as the pivotal case.</strong> On levels, China's two centralities rise in lockstep — the signature dual rise — but in raw shares its trade centrality (13.8% by 2013) runs ahead of its knowledge centrality (6.2%), placing it narrowly on the trade side of the typology (+0.035). On timing, the data refuse to pick a direction: level correlations are trend-dominated, and on year-over-year changes no lead is distinguishable from contemporaneous co-movement. The apparent "knowledge leads by five years" result is a window-edge artifact of the common trend, not evidence of sequencing.</p>
<p class="small"><strong>Limitations.</strong> (1) The country proxy is the filing office, not the inventor's residence. (2) The HS↔CPC concordance is hand-picked; the ALP crosswalk may change Level B magnitudes. (3) The citation window stops at 2013. (4) PageRank is one measure; Section 5 shows the direction is partly measure-dependent. (5) The WTO probe is descriptive.</p>

<h2>10. Conclusion</h2>
<p>Goods and ideas move together in time, but the balance of power differs by country and by industry. Counting inventions rather than filings — and separating what the data contains from when each office's citations entered it — changes the headline: China's 2010 citation explosion is a CNIPA data-exchange artifact, not an innovation surge. On year-over-year changes, no country shows a reliable lead of trade over knowledge or of knowledge over trade; the co-movement is contemporaneous. In raw levels the world splits into knowledge-first economies (the US, Japan), trade-first economies (China and most of continental Europe), and balanced ones (Korea, Germany, the UK) — with Korea already on the knowledge side of the line. Semiconductors sharpen the picture: the trade and knowledge hierarchies agree broadly but diverge at the top — the US owns the knowledge network while Korea and Taiwan convert prior art into the world's chips, their knowledge balances converging on the frontier. The United States remains the singular knowledge hub; Korea is the convergence to watch. The next step is causal: the WTO probe points in the right direction but does not yet identify the channel.</p>

<h2>References</h2>
<ul class="refs">
<li>Gaulier, G. and Zignago, S. (2010). BACI: International Trade Database at the Product-Level. CEPII.</li>
<li>EPO. PATSTAT Worldwide Patent Statistical Database, spring 2018 edition.</li>
<li>Grossman, G. and Helpman, E. (1991). <em>Innovation and Growth in the Global Economy.</em> MIT Press.</li>
<li>Keller, W. (2004). International Technology Diffusion. <em>Journal of Economic Literature</em> 42(3).</li>
<li>Lybbert, T. and Zolas, N. (2014). Getting patents and economic data to speak to each other. <em>World Patent Information</em> 37.</li>
<li>Autor, D., Dorn, D., and Hanson, G. (2016). The China Shock. <em>Annual Review of Economics</em> 8.</li>
</ul>
</div>

<div class="tooltip" id="tip"></div>

<script>
const DATA = __JSON_DATA__;
const years = DATA.years;
const ns = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");
function el(t, a){ const e = document.createElementNS(ns, t); for(const k in a) e.setAttribute(k, a[k]); return e; }
const B = "#2a78d6", O = "#eb6834", GR = "#e1e0d9", INK = "#898781";
const PALETTE = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"];

/* ---- F1: 16-panel grid (raw values, ONE shared scale across all panels) ---- */
(function(){
  const W = 220, H = 135, mL = 26, mR = 6, mT = 6, mB = 14;
  const GM = Math.max(...DATA.countries.flatMap(c => [...c.trade_series, ...c.cite_series])) || 1;
  const grid = document.getElementById("grid");
  DATA.countries.forEach(c => {
    const panel = document.createElement("div");
    panel.className = "panel";
    const tc = c.pathway === "Knowledge-leading" ? "k" : (c.pathway === "Trade-leading" ? "t" : "b");
    const tt = c.pathway === "Knowledge-leading" ? "knowledge-first" : (c.pathway === "Trade-leading" ? "trade-first" : "balanced");
    panel.innerHTML = `<h3>${c.name}</h3><span class="tag ${tc}">${tt}</span>`;
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": c.name});
    const x = i => mL + i/(years.length-1)*(W-mL-mR);
    const y = v => mT + (H-mT-mB)*(1-v/GM);   // v raw, GM = global max share
    [0,0.5,1].forEach(v => svg.appendChild(el("line",{x1:mL,y1:y(v*GM),x2:W-mR,y2:y(v*GM),stroke:v===0?"#c3c2b7":GR,"stroke-width":v===0?1.2:0.8})));
    [0, Math.floor((years.length-1)/2), years.length-1].forEach(i => {
      const t = el("text",{x:x(i),y:H-3,"text-anchor":"middle","font-size":7,fill:INK});
      t.textContent = years[i]; svg.appendChild(t);
    });
    let d1="",d2="";
    c.trade_series.forEach((v,i)=>{d1+=(i?"L":"M")+x(i)+","+y(v);});
    c.cite_series.forEach((v,i)=>{d2+=(i?"L":"M")+x(i)+","+y(v);});
    svg.appendChild(el("path",{d:d1,fill:"none",stroke:B,"stroke-width":1.6,"stroke-linejoin":"round"}));
    svg.appendChild(el("path",{d:d2,fill:"none",stroke:O,"stroke-width":1.6,"stroke-linejoin":"round"}));
    const cross = el("line",{y1:mT,y2:H-mB,stroke:INK,"stroke-width":1,"stroke-dasharray":"2,2",opacity:0});
    const dA = el("circle",{r:3.2,fill:B,stroke:"#fcfcfb","stroke-width":1.5,opacity:0});
    const dB = el("circle",{r:3.2,fill:O,stroke:"#fcfcfb","stroke-width":1.5,opacity:0});
    svg.appendChild(cross); svg.appendChild(dA); svg.appendChild(dB);
    svg.appendChild(el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"transparent"}));
    svg.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left)*(W/r.width);
      const idx = Math.max(0, Math.min(years.length-1, Math.round((px-mL)/(W-mL-mR)*(years.length-1))));
      cross.setAttribute("x1",x(idx));cross.setAttribute("x2",x(idx));cross.setAttribute("opacity",1);
      dA.setAttribute("cx",x(idx));dA.setAttribute("cy",y(c.trade_series[idx]));dA.setAttribute("opacity",1);
      dB.setAttribute("cx",x(idx));dB.setAttribute("cy",y(c.cite_series[idx]));dB.setAttribute("opacity",1);
      tip.style.opacity=1; tip.style.left=(ev.clientX+14)+"px"; tip.style.top=(ev.clientY-8)+"px";
      tip.innerHTML = `<b>${c.name} ${years[idx]}</b><br>Trade PR: <b>${(c.trade_series[idx]*100).toFixed(1)}%</b><br>Cite PR: <b>${(c.cite_series[idx]*100).toFixed(1)}%</b>`;
    });
    svg.addEventListener("mouseleave", ()=>{cross.setAttribute("opacity",0);dA.setAttribute("opacity",0);dB.setAttribute("opacity",0);tip.style.opacity=0;});
    panel.appendChild(svg); grid.appendChild(panel);
  });
})();

/* ---- pathway table ---- */
(function(){
  const t = document.getElementById("pathway-table");
  const groups = [
    ["<span class='tag k'>Knowledge-first</span>", "Knowledge-leading",
     "citation centrality above trade; US and Japan gaps still widening"],
    ["<span class='tag t'>Trade-first</span>", "Trade-leading",
     "trade centrality above knowledge"],
    ["<span class='tag b'>Balanced</span>", "Balanced",
     "both lines close, in raw shares"],
  ];
  groups.forEach(([label, key, pattern]) => {
    const cs = DATA.countries.filter(c => c.pathway === key);
    if (!cs.length) return;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${cs.map(c => `${c.name} (${c.avg_diff>=0?"+":""}${c.avg_diff})`).join(", ")}</td><td>${pattern}</td>`;
    t.appendChild(tr);
  });
})();

/* ---- F3: China 2010 ---- */
(function(){
  const svg=document.getElementById("china2010");
  const d=DATA.china2010, W=720,H=340,mL=58,mR=120,mT=24,mB=40;
  const MAX=2800000;
  const x=i=>mL+i/(d.years.length-1)*(W-mL-mR);
  const y=v=>mT+(H-mT-mB)*(1-v/MAX);
  for(let v=0;v<=2800000;v+=500000){
    svg.appendChild(el("line",{x1:mL,y1:y(v),x2:W-mR,y2:y(v),stroke:v===0?"#c3c2b7":GR,"stroke-width":v===0?1.2:1}));
    const t=el("text",{x:mL-8,y:y(v)+4,"text-anchor":"end","font-size":10,fill:INK}); t.textContent=(v/1e6)+"M"; svg.appendChild(t);
  }
  d.years.forEach((yr,i)=>{ const t=el("text",{x:x(i),y:H-14,"text-anchor":"middle","font-size":10,fill:INK}); t.textContent=yr; svg.appendChild(t); });
  function line(arr,col){ let s=""; arr.forEach((v,i)=>{s+=(i?"L":"M")+x(i)+","+y(v);}); svg.appendChild(el("path",{d:s,fill:"none",stroke:col,"stroke-width":2.2,"stroke-linejoin":"round"})); }
  line(d.app,B); line(d.fam,O);
  const la=el("text",{x:x(9)+10,y:y(d.app[9])+4,"font-size":10.5,fill:B,"font-weight":600}); la.textContent="App-level"; svg.appendChild(la);
  const lf=el("text",{x:x(9)+10,y:y(d.fam[9])+4,"font-size":10.5,fill:O,"font-weight":600}); lf.textContent="Family-level"; svg.appendChild(lf);
  const annY=y(600000);
  svg.appendChild(el("path",{d:`M${x(4)},${y(2148)} L${x(4)},${annY-8} L${x(5)},${annY-8} L${x(5)},${y(1215925)}`,fill:"none",stroke:INK,"stroke-width":1.1,"stroke-dasharray":"3,3"}));
  const at=el("text",{x:(x(4)+x(5))/2,y:annY-26,"text-anchor":"middle","font-size":12,fill:B,"font-weight":700}); at.textContent="566× (app-level)"; svg.appendChild(at);
  const at2=el("text",{x:(x(4)+x(5))/2,y:y(1150000)-4,"text-anchor":"middle","font-size":11,fill:O,"font-weight":700}); at2.textContent="8× (family-level)"; svg.appendChild(at2);
  const at3=el("text",{x:x(2),y:y(1900000),"font-size":9.5,fill:INK,"font-style":"italic"}); at3.textContent="pre-2010 app-level ≈ invisible"; svg.appendChild(at3);
})();

/* ---- F3b: China mechanism (citations by origin | applications) ---- */
(function(){
  const d = DATA.china2010;
  function mini(id, fn){
    const svg = document.getElementById(id);
    const W=340,H=240,mL=44,mR=10,mT=18,mB=34;
    fn(svg, W, H, mL, mR, mT, mB);
  }
  // left: tls212 citations by CN publication year, stacked SEA vs other
  mini("chinaMechanismCites", (svg,W,H,mL,mR,mT,mB)=>{
    const MAXC = 3600000;
    const x=i=>mL+i/(d.years.length-1)*(W-mL-mR);
    const y=v=>mT+(H-mT-mB)*(1-v/MAXC);
    for(let v=0;v<=MAXC;v+=600000){
      svg.appendChild(el("line",{x1:mL,y1:y(v),x2:W-mR,y2:y(v),stroke:v===0?"#c3c2b7":GR,"stroke-width":v===0?1.2:0.8}));
      const t=el("text",{x:mL-6,y:y(v)+3,"text-anchor":"end","font-size":8,fill:INK}); t.textContent=(v/1e6).toFixed(v%1000000===0?0:1)+"M"; svg.appendChild(t);
    }
    d.years.forEach((yr,i)=>{ if(i%2===0){ const t=el("text",{x:x(i),y:H-14,"text-anchor":"middle","font-size":8,fill:INK}); t.textContent=yr; svg.appendChild(t);} });
    d.tls212_cn.forEach((v,i)=>{
      const seaH = y(d.other[i]) - y(d.sea[i]+d.other[i]);
      svg.appendChild(el("rect",{x:x(i)-7,y:y(d.sea[i]+d.other[i]),width:14,height:seaH,fill:O}));
      if (d.other[i] > 0) svg.appendChild(el("rect",{x:x(i)-7,y:y(d.other[i]),width:14,height:y(0)-y(d.other[i]),fill:INK,opacity:0.35}));
    });
    const t=el("text",{x:x(4),y:y(3600000)-4,"text-anchor":"middle","font-size":9,fill:"#52514e","font-style":"italic"}); t.textContent="zero before 2010"; svg.appendChild(t);
    const t2=el("text",{x:x(7)+8,y:y(d.tls212_cn[7]),"font-size":9,fill:O,"font-weight":600}); t2.textContent="SEA (search report)"; svg.appendChild(t2);
  });
  // right: tls201 CN applications by filing year
  mini("chinaMechanismApps", (svg,W,H,mL,mR,mT,mB)=>{
    const MAXA = 1600000;
    const x=i=>mL+i/(d.years.length-1)*(W-mL-mR);
    const y=v=>mT+(H-mT-mB)*(1-v/MAXA);
    for(let v=0;v<=MAXA;v+=400000){
      svg.appendChild(el("line",{x1:mL,y1:y(v),x2:W-mR,y2:y(v),stroke:v===0?"#c3c2b7":GR,"stroke-width":v===0?1.2:0.8}));
      const t=el("text",{x:mL-6,y:y(v)+3,"text-anchor":"end","font-size":8,fill:INK}); t.textContent=(v/1e6).toFixed(1)+"M"; svg.appendChild(t);
    }
    d.years.forEach((yr,i)=>{ if(i%2===0){ const t=el("text",{x:x(i),y:H-14,"text-anchor":"middle","font-size":8,fill:INK}); t.textContent=yr; svg.appendChild(t);} });
    let s="";
    d.cn_apps.forEach((v,i)=>{ s+=(i?"L":"M")+x(i)+","+y(v); });
    svg.appendChild(el("path",{d:s,fill:"none",stroke:B,"stroke-width":2.2,"stroke-linejoin":"round"}));
    const t=el("text",{x:x(9)-4,y:y(d.cn_apps[9])-8,"text-anchor":"end","font-size":9,fill:B,"font-weight":600}); t.textContent="smooth — no 2010 break"; svg.appendChild(t);
  });
})();

/* ---- F4: lead/lag bars (levels) ---- */
(function(){
  const svg=document.getElementById("lagbars");
  const W=760,H=360,mL=150,mR=120,mT=26,mB=30;
  const xs=v=>mL+(v+6)/12*(W-mL-mR);
  const rows=DATA.countries.filter(c=>Math.abs(c.best_r)>=0.5).sort((a,b)=>a.best_lag-b.best_lag);
  svg.appendChild(el("line",{x1:xs(0),y1:mT,x2:xs(0),y2:H-mB,stroke:"#c3c2b7","stroke-width":1.2}));
  const t0=el("text",{x:xs(0),y:H-8,"text-anchor":"middle","font-size":9,fill:INK}); t0.textContent="0"; svg.appendChild(t0);
  const tL=el("text",{x:xs(-5),y:mT-8,"text-anchor":"middle","font-size":10,fill:O,"font-weight":700}); tL.textContent="← knowledge leads"; svg.appendChild(tL);
  const tR=el("text",{x:xs(5),y:mT-8,"text-anchor":"middle","font-size":10,fill:B,"font-weight":700}); tR.textContent="trade leads →"; svg.appendChild(tR);
  const rowH=(H-mT-mB)/rows.length;
  rows.forEach((c,i)=>{
    const cy=mT+i*rowH+rowH/2;
    const col=c.best_lag<0?O:B;
    const bar=el("rect",{x:Math.min(xs(0),xs(c.best_lag)),y:cy-rowH*0.28,width:Math.abs(xs(c.best_lag)-xs(0)),height:rowH*0.56,rx:3,fill:col,opacity:0.85});
    bar.addEventListener("mousemove",ev=>{tip.style.opacity=1;tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY-8)+"px";tip.innerHTML=`<b>${c.name}</b><br>best lag (levels): <b>${c.best_lag>0?"+":""}${c.best_lag}</b> · r = <b>${c.best_r}</b>`;});
    bar.addEventListener("mouseleave",()=>{tip.style.opacity=0;});
    svg.appendChild(bar);
    const lab=el("text",{x:mL-8,y:cy+3,"text-anchor":"end","font-size":10,fill:"#52514e","font-weight":600}); lab.textContent=c.name; svg.appendChild(lab);
    const rl=el("text",{x:xs(c.best_lag)+(c.best_lag>0?8:-8),y:cy+3,"text-anchor":c.best_lag>0?"start":"end","font-size":9,fill:INK}); rl.textContent="r="+c.best_r; svg.appendChild(rl);
  });
})();

/* ---- F4b: China correlogram ---- */
(function(){
  const svg=document.getElementById("correlogram");
  const d=DATA.china_correlogram;
  const W=760,H=300,mL=58,mR=130,mT=26,mB=42;
  const xs=k=>mL+(k+5)/10*(W-mL-mR);
  const ys=v=>mT+(H-mT-mB)*(1-v/1.05);
  svg.appendChild(el("line",{x1:mL,y1:ys(0),x2:W-mR,y2:ys(0),stroke:"#c3c2b7","stroke-width":1.2}));
  for(let v=0;v<=1;v+=0.25){
    svg.appendChild(el("line",{x1:mL,y1:ys(v),x2:W-mR,y2:ys(v),stroke:GR,"stroke-width":0.8}));
    const t=el("text",{x:mL-8,y:ys(v)+3,"text-anchor":"end","font-size":9,fill:INK}); t.textContent=v.toFixed(2); svg.appendChild(t);
  }
  [-5,-4,-3,-2,-1,0,1,2,3,4,5].forEach(k=>{
    const t=el("text",{x:xs(k),y:H-14,"text-anchor":"middle","font-size":10,fill:INK}); t.textContent=(k>0?"+":"")+k; svg.appendChild(t);
  });
  function line(obj,col){
    let s="";
    [-5,-4,-3,-2,-1,0,1,2,3,4,5].forEach(k=>{
      if (!(k in obj)) return;
      s+=(s?"L":"M")+xs(k)+","+ys(obj[k]);
      svg.appendChild(el("circle",{cx:xs(k),cy:ys(obj[k]),r:3.2,fill:col,stroke:"#fcfcfb","stroke-width":1.2}));
    });
    svg.appendChild(el("path",{d:s,fill:"none",stroke:col,"stroke-width":2,"stroke-linejoin":"round"}));
  }
  line(d.levels, B); line(d.changes, O);
  const l1=el("text",{x:W-mR+8,y:ys(0.99),"font-size":10,fill:B,"font-weight":600}); l1.textContent="Levels"; svg.appendChild(l1);
  const l2=el("text",{x:W-mR+8,y:ys(0.55),"font-size":10,fill:O,"font-weight":600}); l2.textContent="Changes"; svg.appendChild(l2);
  const ann=el("text",{x:xs(0),y:ys(0.92),"text-anchor":"middle","font-size":9,fill:INK,"font-style":"italic"}); ann.textContent="r ≈ 0.99 at EVERY lag — pure trend"; svg.appendChild(ann);
  const ann2=el("text",{x:xs(2.5),y:ys(0.18),"text-anchor":"middle","font-size":9,fill:INK,"font-style":"italic"}); ann2.textContent="flat — no lead either way"; svg.appendChild(ann2);
})();

/* ---- F4c: changes-based lag bars ---- */
(function(){
  const svg=document.getElementById("changelagbars");
  const W=760,H=380,mL=150,mR=120,mT=26,mB=30;
  const xs=v=>mL+(v+6)/12*(W-mL-mR);
  const rows=DATA.changes.slice().sort((a,b)=>a.lag-b.lag);
  svg.appendChild(el("line",{x1:xs(0),y1:mT,x2:xs(0),y2:H-mB,stroke:"#c3c2b7","stroke-width":1.2}));
  const t0=el("text",{x:xs(0),y:H-8,"text-anchor":"middle","font-size":9,fill:INK}); t0.textContent="0"; svg.appendChild(t0);
  const tL=el("text",{x:xs(-5),y:mT-8,"text-anchor":"middle","font-size":10,fill:O,"font-weight":700}); tL.textContent="← knowledge leads"; svg.appendChild(tL);
  const tR=el("text",{x:xs(5),y:mT-8,"text-anchor":"middle","font-size":10,fill:B,"font-weight":700}); tR.textContent="trade leads →"; svg.appendChild(tR);
  const rowH=(H-mT-mB)/rows.length;
  rows.forEach((c,i)=>{
    const cy=mT+i*rowH+rowH/2;
    const col=c.lag<0?O:(c.lag>0?B:INK);
    const bar=el("rect",{x:Math.min(xs(0),xs(c.lag)),y:cy-rowH*0.28,width:Math.abs(xs(c.lag)-xs(0)),height:rowH*0.56,rx:3,fill:col,opacity:c.weak?0.28:0.85});
    bar.addEventListener("mousemove",ev=>{tip.style.opacity=1;tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY-8)+"px";tip.innerHTML=`<b>${c.name}</b><br>best lag (changes): <b>${c.lag>0?"+":""}${c.lag}</b> · r = <b>${c.r>=0?"+":""}${c.r}</b>${c.weak?" · weak":""}`;});
    bar.addEventListener("mouseleave",()=>{tip.style.opacity=0;});
    svg.appendChild(bar);
    const lab=el("text",{x:mL-8,y:cy+3,"text-anchor":"end","font-size":10,fill:"#52514e","font-weight":600}); lab.textContent=c.name; svg.appendChild(lab);
    const rl=el("text",{x:xs(c.lag)+(c.lag>0?8:-8),y:cy+3,"text-anchor":c.lag>0?"start":"end","font-size":9,fill:INK}); rl.textContent="r="+(c.r>=0?"+":"")+c.r; svg.appendChild(rl);
  });
})();

/* ---- F4: sector lags (full 8-sector grid; empty cells marked) ---- */
(function(){
  const svg=document.getElementById("sectorlags");
  const W=760,H=300,mL=120,mR=40,mT=20,mB=40;
  const rows=DATA.sector_lags;
  const cols=[["country","country"],["electronics","elec"],["machinery","mach"],
    ["plastics","plas"],["oil_fuels","oil"],["apparel","apparel"],["iron_steel","steel"],
    ["vehicles","veh"],["pharma","pharma"]];
  const xs=i=>mL+i/(cols.length-1)*(W-mL-mR);
  const ys=v=>mT+(6-v)/12*(H-mT-mB);
  svg.appendChild(el("line",{x1:mL,y1:ys(0),x2:W-mR,y2:ys(0),stroke:"#c3c2b7","stroke-width":1.2}));
  cols.forEach(([label,key],i)=>{
    const t=el("text",{x:xs(i),y:H-12,"text-anchor":"middle","font-size":9.5,fill:INK,"font-weight":600}); t.textContent=label; svg.appendChild(t);
  });
  rows.forEach((r,ri)=>{
    cols.forEach(([label,key],i)=>{
      const v=r[key];
      if(v===undefined || v===null){
        // no estimate: faint dot on the 0 line marks the cell as empty
        svg.appendChild(el("circle",{cx:xs(i),cy:ys(0),r:1.6,fill:"#c3c2b7"}));
        return;
      }
      const col=v<0?O:(v>0?B:"#898781");
      svg.appendChild(el("circle",{cx:xs(i),cy:ys(v),r:5,fill:col,opacity:0.9}));
      const t=el("text",{x:xs(i),y:ys(v)+(v>0?-8:14),"text-anchor":"middle","font-size":8.5,fill:INK}); t.textContent=v>0?"+"+v:v; svg.appendChild(t);
    });
    const t=el("text",{x:mL-10,y:ys(0)+ri*16,"text-anchor":"end","font-size":10,fill:"#52514e","font-weight":600}); t.textContent=r.name; svg.appendChild(t);
  });
})();

/* ---- F6: WTO bars ---- */
(function(){
  const svg=document.getElementById("wtobars");
  const W=720,H=300,mL=60,mR=30,mT=24,mB=44;
  const groups=[["Knowledge into China",DATA.wto.into],["Knowledge out of China",DATA.wto.out]];
  const xs=i=>mL+i/(groups.length-1)*(W-mL-mR);
  const ys=v=>mT+(H-mT-mB)*(1-v/1.4);
  svg.appendChild(el("line",{x1:mL,y1:ys(0),x2:W-mR,y2:ys(0),stroke:"#c3c2b7","stroke-width":1.2}));
  groups.forEach(([label,d],gi)=>{
    const bw=56, off=-bw-10;
    [["CN pairs",d.cn,B,off],["Hub controls",d.control,"#898781",10]].forEach(([lab,v,col,dx])=>{
      svg.appendChild(el("rect",{x:xs(gi)+dx,y:ys(v),width:bw,height:ys(0)-ys(v),rx:3,fill:col,opacity:0.9}));
      const t=el("text",{x:xs(gi)+dx+bw/2,y:ys(v)-5,"text-anchor":"middle","font-size":9.5,fill:"#52514e","font-weight":600});
      t.textContent="+"+v.toFixed(2); svg.appendChild(t);
      const t2=el("text",{x:xs(gi)+dx+bw/2,y:ys(0)+14,"text-anchor":"middle","font-size":9,fill:INK}); t2.textContent=lab; svg.appendChild(t2);
    });
    const t=el("text",{x:xs(gi),y:H-14,"text-anchor":"middle","font-size":10,fill:INK,"font-weight":600}); t.textContent=label+"  ·  DiD +"+d.did.toFixed(3); svg.appendChild(t);
  });
})();

/* ---- F6: semiconductor bump chart (2 bump panels + rank-agreement panel) ---- */
(function(){
  const svg=document.getElementById("semiBump");
  const d=DATA.semi;
  const W=760,H=500,mL=34,mR=92,mT=24,mB=18,panelGap=14;
  const HB=384;   // bump-panel region height
  const pW=(W-mL-mR-panelGap)/2;
  const ys=15;    // max rank shown inside a panel
  const xs=i=>mL+pW*(i/(years.length-1));
  const y=rank=>mT+(rank-1)/(ys-1)*(HB-mT-mB);
  const pal = c=>PALETTE[d.colored.findIndex(x=>x.code===c)];
  function drawPanel(xOff, title, key){
    const X=i=>xOff+xs(i);
    const clipId="clip_"+key+Math.round(xOff);
    const clip=el("clipPath",{id:clipId});
    clip.appendChild(el("rect",{x:xOff-2,y:mT-4,width:pW+4,height:HB-mT+4}));
    svg.appendChild(clip);
    for(let r=1;r<=ys;r+=2){
      svg.appendChild(el("line",{x1:xOff,y1:y(r),x2:xOff+pW,y2:y(r),stroke:GR,"stroke-width":0.8}));
      const t=el("text",{x:xOff-7,y:y(r)+3,"text-anchor":"end","font-size":9,fill:INK}); t.textContent=r; svg.appendChild(t);
    }
    [1995,1998,2001,2004,2007,2010,2013].forEach(yr=>{
      const i=years.indexOf(yr);
      const t=el("text",{x:X(i),y:HB-4,"text-anchor":"middle","font-size":9,fill:INK}); t.textContent=yr; svg.appendChild(t);
    });
    const titleEl=el("text",{x:xOff+4,y:mT-8,"font-size":11,fill:"#0b0b0b","font-weight":700}); titleEl.textContent=title; svg.appendChild(titleEl);
    const offPanel=[];
    const endLabel=(c,col,emph)=>{
      const last=c[key][years.length-1];
      if(last===null) return;
      if(last>ys){ offPanel.push(c.code+" "+last); return; }   // off-panel: listed below the panel
      const yL=y(last);
      svg.appendChild(el("circle",{cx:X(years.length-1),cy:yL,r:emph?4:2.8,fill:col,stroke:"#fcfcfb","stroke-width":1.2}));
      const t=el("text",{x:X(years.length-1)+6,y:yL+3,"font-size":emph?9.5:8.5,"font-weight":emph?"700":"400",fill:col});
      t.textContent=c.code+" "+last; svg.appendChild(t);
    };
    d.gray.forEach(c=>{
      let s="";
      c[key].forEach((v,i)=>{ if(v!==null){ s+=(s?"L":"M")+X(i)+","+y(v); } });
      const p=el("path",{d:s,fill:"none",stroke:"#898781","stroke-width":1,"stroke-opacity":0.55,"stroke-linejoin":"round"});
      p.setAttribute("clip-path","url(#"+clipId+")");
      svg.appendChild(p);
      endLabel(c,"#898781",false);
    });
    d.colored.forEach(c=>{
      const emph=d.watch.includes(c.code);
      let s="";
      c[key].forEach((v,i)=>{ if(v!==null){ s+=(s?"L":"M")+X(i)+","+y(v); } });
      const p=el("path",{d:s,fill:"none",stroke:pal(c.code),"stroke-width":emph?2.8:1.7,"stroke-linejoin":"round","stroke-linecap":"round"});
      p.setAttribute("clip-path","url(#"+clipId+")");
      p.addEventListener("mousemove",ev=>{
        const r=svg.getBoundingClientRect();
        const px=(ev.clientX-r.left)*(W/r.width)-xOff;
        const idx=Math.max(0,Math.min(years.length-1,Math.round(px/pW*(years.length-1))));
        tip.style.opacity=1;tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY-8)+"px";
        tip.innerHTML=`<b>${c.name} ${years[idx]}</b><br>${title.split(" ")[0]} rank: <b>${c[key][idx]??"–"}</b>`;
      });
      p.addEventListener("mouseleave",()=>{tip.style.opacity=0;});
      svg.appendChild(p);
      endLabel(c,pal(c.code),emph);
    });
    if(offPanel.length){
      const t=el("text",{x:xOff+4,y:HB+13,"font-size":8.5,fill:INK});
      t.textContent="off-panel: "+offPanel.join(" · ");
      svg.appendChild(t);
    }
  }
  drawPanel(mL, "Knowledge rank — H01L citation PageRank", "knowledge_ranks");
  drawPanel(mL+pW+panelGap, "Trade rank — HS 8541/42 export PageRank", "trade_ranks");

  // bottom: rank agreement
  const yb0=HB+42, yb1=H-12;
  const ysR=v=>yb1-v*(yb1-yb0);
  const sX=i=>mL+i/(years.length-1)*(W-mL-mR);
  [0,0.5,1].forEach(v=>{
    svg.appendChild(el("line",{x1:mL,y1:ysR(v),x2:W-mR,y2:ysR(v),stroke:v===0?"#c3c2b7":GR,"stroke-width":v===0?1.2:0.8}));
    const t=el("text",{x:mL-7,y:ysR(v)+3,"text-anchor":"end","font-size":9,fill:INK}); t.textContent=v; svg.appendChild(t);
  });
  [1995,1998,2001,2004,2007,2010,2013].forEach(yr=>{
    const i=years.indexOf(yr);
    const t=el("text",{x:sX(i),y:H-2,"text-anchor":"middle","font-size":9,fill:INK}); t.textContent=yr; svg.appendChild(t);
  });
  const tT=el("text",{x:mL+4,y:yb0-8,"font-size":10,fill:"#0b0b0b","font-weight":700});
  tT.textContent="Rank agreement — Spearman ρ between the trade and knowledge rank orders"; svg.appendChild(tT);
  function spLine(pairs,col,lw,dots){
    let s="";
    pairs.forEach(([i,v],j)=>{
      s+=(j?"L":"M")+sX(i)+","+ysR(v);
      if(dots) svg.appendChild(el("circle",{cx:sX(i),cy:ysR(v),r:2.6,fill:col,stroke:"#fcfcfb","stroke-width":1}));
    });
    svg.appendChild(el("path",{d:s,fill:"none",stroke:col,"stroke-width":lw,"stroke-linejoin":"round"}));
  }
  const spFull=[], spTop=[];
  d.spearman.forEach(r=>{
    const i=years.indexOf(r.year);
    if(i>=0){ spFull.push([i,r.rho]); if(r.rho_top!==null) spTop.push([i,r.rho_top]); }
  });
  spLine(spFull,"#898781",1.8,false);
  spLine(spTop,PALETTE[0],2.4,true);
  const labels=[];
  if(d.rho_2013!=null) labels.push(["all countries ρ="+d.rho_2013.toFixed(2), d.rho_2013, "#898781"]);
  if(d.rho_top_2013!=null) labels.push(["top-15 tier ρ="+d.rho_top_2013.toFixed(2), d.rho_top_2013, PALETTE[0]]);
  // stagger end labels if their y positions collide
  const used=[];
  labels.sort((a,b)=>a[1]-b[1]).forEach(([txt,v,col])=>{
    let yv=ysR(v);
    for(let k=0;k<8;k++){
      const hit=used.find(u=>Math.abs(u-yv)<12);
      if(hit===undefined) break;
      yv=hit+(yv<=hit?-12:12);
    }
    used.push(yv);
    const t=el("text",{x:W-mR-8,y:yv+3,"text-anchor":"end","font-size":9,fill:col,"font-weight":col===PALETTE[0]?600:400});
    t.textContent=txt; svg.appendChild(t);
  });
})();

/* ---- F8: semiconductor ratio ---- */
(function(){
  const svg=document.getElementById("semiRatio");
  const d=DATA.semi;
  const leg=document.getElementById("semi-ratio-legend");
  d.colored.forEach(c=>{
    const sp=document.createElement("span");
    const col=PALETTE[d.colored.findIndex(x=>x.code===c)];
    sp.innerHTML=`<span class="swatch" style="background:${col}"></span>${c.name}${d.watch.includes(c.code)?" (bold)":""}`;
    leg.appendChild(sp);
  });
  const W=760,H=300,mL=52,mR=64,mT=30,mB=36;
  const LO=0.0625, HI=16;
  const xs=i=>mL+i/(years.length-1)*(W-mL-mR);
  const y=v=>mT+(H-mT-mB)*(1-(Math.log2(v)-Math.log2(LO))/(Math.log2(HI)-Math.log2(LO)));
  [0.125,0.25,0.5,1,2,4,8].forEach(v=>{
    svg.appendChild(el("line",{x1:mL,y1:y(v),x2:W-mR,y2:y(v),stroke:v===1?"#c3c2b7":GR,"stroke-width":v===1?1.2:0.8}));
    const t=el("text",{x:mL-7,y:y(v)+3,"text-anchor":"end","font-size":9,fill:INK}); t.textContent=v; svg.appendChild(t);
  });
  [1995,1998,2001,2004,2007,2010,2013].forEach(yr=>{
    const i=years.indexOf(yr);
    const t=el("text",{x:xs(i),y:H-8,"text-anchor":"middle","font-size":9,fill:INK}); t.textContent=yr; svg.appendChild(t);
  });
  const s1=el("text",{x:xs(0)+4,y:y(1.35),"font-size":8.5,fill:"#52514e","font-style":"italic"}); s1.textContent="net knowledge source"; svg.appendChild(s1);
  const s2=el("text",{x:xs(0)+4,y:y(0.75),"font-size":8.5,fill:"#52514e","font-style":"italic"}); s2.textContent="net absorber"; svg.appendChild(s2);
  const pal = c=>PALETTE[d.colored.findIndex(x=>x.code===c)];
  const endLabels=[];
  d.colored.forEach(c=>{
    const emph=d.watch.includes(c.code);
    let s="";
    c.ratio.forEach((v,i)=>{ if(v!==null){ s+=(s?"L":"M")+xs(i)+","+y(v); } });
    const p=el("path",{d:s,fill:"none",stroke:pal(c.code),"stroke-width":emph?2.8:1.7,"stroke-linejoin":"round","stroke-linecap":"round"});
    p.addEventListener("mousemove",ev=>{
      const r=svg.getBoundingClientRect();
      const px=(ev.clientX-r.left)*(W/r.width);
      const idx=Math.max(0,Math.min(years.length-1,Math.round((px-mL)/(W-mL-mR)*(years.length-1))));
      tip.style.opacity=1;tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY-8)+"px";
      tip.innerHTML=`<b>${c.name} ${years[idx]}</b><br>received/made: <b>${c.ratio[idx]??"–"}</b>`;
    });
    p.addEventListener("mouseleave",()=>{tip.style.opacity=0;});
    svg.appendChild(p);
    const last=c.ratio[years.length-1];
    if(last!==null){
      svg.appendChild(el("circle",{cx:xs(years.length-1),cy:y(last),r:emph?4:2.8,fill:pal(c.code),stroke:"#fcfcfb","stroke-width":1.2}));
      endLabels.push({txt:c.code, v:last, col:pal(c.code), emph});
    }
  });
  // place end labels with anti-collision staggering (JP 3.60 and SG 3.66
  // would otherwise sit on top of each other)
  const usedY=[];
  endLabels.sort((a,b)=>a.v-b.v).forEach(l=>{
    let yv=y(l.v);
    for(let k=0;k<8;k++){
      const hit=usedY.find(u=>Math.abs(u-yv)<11);
      if(hit===undefined) break;
      yv=hit+(yv<=hit?-11:11);
    }
    usedY.push(yv);
    const t=el("text",{x:xs(years.length-1)+6,y:yv+3,"font-size":l.emph?9.5:8.5,"font-weight":l.emph?"700":"400",fill:l.col});
    t.textContent=l.txt; svg.appendChild(t);
  });
})();

/* ---- robustness table ---- */
(function(){
  const t=document.getElementById("rob-table");
  DATA.robustness.forEach(r=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${r.name}</td>`+
      `<td class="num">${r.pagerank>0?"+"+r.pagerank:r.pagerank}</td>`+
      `<td class="num">${r.betweenness>0?"+"+r.betweenness:r.betweenness}</td>`+
      `<td class="num">${r.eigenvector>0?"+"+r.eigenvector:r.eigenvector}</td>`+
      `<td>${r.robust?"yes":"<span class=neg>partial / no</span>"}</td>`;
    t.appendChild(tr);
  });
})();

/* ---- bootstrap table ---- */
(function(){
  const t=document.getElementById("boot-table");
  DATA.bootstrap.forEach(r=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${r.name}</td>`+
      `<td class="num">${r.lag>0?"+"+r.lag:r.lag}</td>`+
      `<td class="num">${r.r>=0?"+":""}${r.r}</td>`+
      `<td class="num">${(r.share*100).toFixed(0)}% (lag 0 dominates)</td>`+
      `<td class="num">[${r.lo>=0?"+":""}${r.lo}, ${r.hi>=0?"+":""}${r.hi}]</td>`;
    t.appendChild(tr);
  });
})();

/* ---- semiconductor 2013 snapshot table ---- */
(function(){
  const d=DATA.semi;
  const t=document.getElementById("semi-table");
  d.snapshot.forEach(r=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${r.name}</td>`+
      `<td class="num">${r.knowledge_rank}</td>`+
      `<td class="num">${r.trade_rank}</td>`+
      `<td class="num">${r.export_share}%</td>`+
      `<td class="num">${r.cite_ratio!=null?r.cite_ratio:"–"}</td>`;
    t.appendChild(tr);
  });
})();
</script>
</body>
</html>
'''

html = HTML.replace("__JSON_DATA__", json_str)
out_path = "../report/paper.html"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved → {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
