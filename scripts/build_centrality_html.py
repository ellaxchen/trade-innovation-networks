"""
Build a self-contained HTML report of the centrality analysis:
per-country PageRank panels (trade vs knowledge overlaid), the three
development pathways, lead/lag correlations, and crossing analysis.

Usage: python build_centrality_html.py
Output: ../report/centrality_report.html (kept local — report/ is gitignored)
"""
import pandas as pd
import numpy as np
import networkx as nx
import json
import os

YEARS = list(range(1995, 2014))  # citations stop at 2013
ALPHA = 0.85
LAGS = list(range(-5, 6))
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# ---- country codes ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
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

print("[1/3] Computing PageRank series 1995-2013...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[(cite.year >= 1995) & (cite.year <= 2013) &
            (cite.citing_country != cite.cited_country)]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) & ~cite.cited_country.isin(NON_COUNTRIES)]

trade_pr, cite_pr = {}, {}
for yr in YEARS:
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"])
    t = t[t.s != t.t]
    G = nx.DiGraph()
    G.add_weighted_edges_from(t[["s", "t", "trade_value"]].itertuples(index=False, name=None))
    trade_pr[yr] = nx.pagerank(G, alpha=ALPHA, weight="weight")

    c = cite[cite.year == yr]
    G2 = nx.DiGraph()
    G2.add_weighted_edges_from(
        c[["citing_country", "cited_country", "citation_count"]].itertuples(index=False, name=None))
    cite_pr[yr] = nx.pagerank(G2, alpha=ALPHA, weight="weight") if G2.number_of_nodes() else {}

# ---- select top 16 by combined raw volume (matches centrality_analysis.py) ----
print("[2/3] Selecting countries and computing diagnostics...")
t_src = pd.Series({c: sum(trade_pr[y].get(c, 0) for y in YEARS) for c in
                   set().union(*(d.keys() for d in trade_pr.values()))})
# raw trade volume: sum of trade_value over all years (source + target)
trade_vol = pd.Series(dtype=float)
for yr in YEARS:
    t = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    t["s"] = t["exporter_name"].map(name_to_iso)
    t["t"] = t["importer_name"].map(name_to_iso)
    t = t.dropna(subset=["s", "t"])
    trade_vol = trade_vol.add(t.groupby("s")["trade_value"].sum(), fill_value=0)
    trade_vol = trade_vol.add(t.groupby("t")["trade_value"].sum(), fill_value=0)
# raw citation volume: sum of citation_count over all years (citing + cited)
cite_vol = cite.groupby("citing_country")["citation_count"].sum().add(
    cite.groupby("cited_country")["citation_count"].sum(), fill_value=0)
combined = trade_vol.add(cite_vol, fill_value=0)
top = combined.sort_values(ascending=False).index[:16].tolist()
# order: pathway then diff magnitude for nicer grid
print(f"  Top 16: {top}")

records = []
for c in top:
    ts = [trade_pr[y].get(c, 0) for y in YEARS]
    cs = [cite_pr[y].get(c, 0) for y in YEARS]
    t_max, c_max = max(ts) or 1, max(cs) or 1
    tn = [v / t_max for v in ts]
    cn = [v / c_max for v in cs]
    diffs = [tn[i] - cn[i] for i in range(len(YEARS))]
    avg_diff = float(np.mean(diffs))
    trend = float(np.mean(diffs[-5:]) - np.mean(diffs[:5]))

    if avg_diff > 0.05:
        pathway = "Trade-leading"
    elif avg_diff < -0.05:
        pathway = "Knowledge-leading"
    else:
        pathway = "Balanced"

    # first crossing
    first_cross = None
    for i in range(1, len(YEARS)):
        if diffs[i - 1] * diffs[i] < 0:
            first_cross = YEARS[i]
            break
    side = lambda d: "trade" if d > 0.02 else ("cite" if d < -0.02 else "tied")
    start_side, end_side = side(diffs[0]), side(diffs[-1])

    # who rises first (year of fastest rise)
    t_rise = YEARS[int(np.argmax(np.diff(tn)))]
    c_rise = YEARS[int(np.argmax(np.diff(cn)))]
    rises = "trade" if t_rise < c_rise else ("cite" if c_rise < t_rise else "together")

    # lead/lag best correlation (canonical convention from lead_lag_analysis.py:
    # positive lag = corr(trade[t], cite[t+lag]) = trade leads;
    # negative lag = corr(cite[t], trade[t+|lag|]) = knowledge leads)
    corrs = {}
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10:
            corrs[k] = float(np.corrcoef(x, y)[0, 1])
    best_lag = max(corrs, key=lambda k: abs(corrs[k]))
    best_r = corrs[best_lag]

    records.append({
        "code": c,
        "name": iso_to_name.get(c, c),
        "trade_series": [round(v, 4) for v in ts],
        "cite_series": [round(v, 4) for v in cs],
        "avg_diff": round(avg_diff, 3),
        "trend": round(trend, 3),
        "pathway": pathway,
        "first_cross": first_cross,
        "start_side": start_side,
        "end_side": end_side,
        "rises_first": rises,
        "best_lag": best_lag,
        "best_r": round(best_r, 3),
    })

# sort grid: knowledge-leading first, then balanced, then trade-leading
order = {"Knowledge-leading": 0, "Balanced": 1, "Trade-leading": 2}
records.sort(key=lambda r: (order[r["pathway"]], -abs(r["avg_diff"])))

data = {"years": YEARS, "countries": records}
print("[3/3] Generating HTML...")

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Centrality Report — Trade vs Knowledge</title>
<style>
:root{
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
  --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);color:var(--text-primary);line-height:1.6}
.page{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:1.7rem;font-weight:700;margin-bottom:4px}
.subtitle{color:var(--text-secondary);margin-bottom:24px}
h2{font-size:1.2rem;font-weight:700;margin:34px 0 10px}
p{margin:8px 0}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:14px 0}
.muted{color:var(--text-secondary)} .small{font-size:0.86rem}

.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.tile{flex:1;min-width:160px;background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .num{font-size:1.65rem;font-weight:800;letter-spacing:-0.02em}
.tile .cap{font-size:0.8rem;color:var(--text-secondary);margin-top:2px}
.b{color:var(--series-1)} .o{color:var(--series-2)}

/* small multiples grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:10px 12px 8px;position:relative}
.panel h3{font-size:0.86rem;font-weight:700;margin-bottom:2px}
.panel .tag{font-size:0.68rem;font-weight:700;padding:1px 7px;border-radius:20px;display:inline-block;margin-bottom:4px}
.tag.k{background:rgba(235,104,52,0.14);color:#c14e1e}
.tag.t{background:rgba(42,120,214,0.14);color:#1c5cab}
.tag.b{background:rgba(137,135,129,0.16);color:#52514e}
.panel svg{width:100%;display:block}
.legend{display:flex;gap:18px;align-items:center;margin:4px 2px 12px;font-size:0.85rem;color:var(--text-secondary)}
.swatch{width:18px;height:4px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:middle}
.tooltip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--page);padding:7px 10px;border-radius:7px;font-size:0.78rem;opacity:0;transition:opacity .12s;z-index:50;white-space:nowrap}

table{border-collapse:collapse;width:100%;font-size:0.88rem;margin:8px 0}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600;font-size:0.8rem;border-bottom:2px solid var(--baseline)}
td.num{font-variant-numeric:tabular-nums}
.pos{color:#0ca30c}.neg{color:#d03b3b}

.callout{border-left:4px solid var(--series-2);background:rgba(235,104,52,0.06);border-radius:0 10px 10px 0;padding:12px 16px;margin:14px 0;font-size:0.94rem}
</style>
</head>
<body>
<div class="page">
<h1>Centrality Report — Trade vs. Knowledge</h1>
<p class="subtitle">PageRank centrality for each country and year in both networks, 1995–2013.
Family-level citations, cross-border only, α = 0.85, each series normalized to its own maximum.</p>

<div class="tiles">
  <div class="tile"><div class="num o">r = 0.990</div><div class="cap">China: knowledge leads trade by 5 years — the strongest cross-correlation in the data</div></div>
  <div class="tile"><div class="num b">+0.236</div><div class="cap">South Korea: widest trade-centrality lead in the sample</div></div>
  <div class="tile"><div class="num o">−0.180</div><div class="cap">Japan: widest knowledge-centrality lead, still widening</div></div>
  <div class="tile"><div class="num">3</div><div class="cap">Pathways: knowledge-first, trade-first, dual rise</div></div>
</div>

<div class="legend">
  <span><span class="swatch" style="background:var(--series-1)"></span>Trade PageRank (blue)</span>
  <span><span class="swatch" style="background:var(--series-2)"></span>Citation PageRank (orange)</span>
  <span class="muted small">Hover any panel for values. Divergences and crossings are the findings.</span>
</div>

<div class="grid" id="grid"></div>

<h2>The three pathways</h2>
<div class="card">
<table>
  <tr><th>Pathway</th><th>Countries</th><th>Pattern</th></tr>
  <tr><td><span class="tag k">Knowledge-first</span></td><td>US, Japan, Spain, Hong Kong, Singapore, Mexico</td><td>Citation PR consistently above trade PR; the knowledge gap widens for US and Japan</td></tr>
  <tr><td><span class="tag b">Balanced / dual rise</span></td><td>China, Canada, Italy</td><td>Both lines rise in parallel — China is the only major country where neither leads</td></tr>
  <tr><td><span class="tag t">Trade-first</span></td><td>Germany, Netherlands, Belgium, Korea, Switzerland, France, UK</td><td>Trade PR above knowledge PR; Korea's knowledge line is catching up fastest</td></tr>
</table>
</div>

<h2>Lead / lag — which line moves first</h2>
<div class="card">
<p class="small muted">Positive lag = trade centrality changes first, then knowledge follows. Negative lag = knowledge leads.</p>
<table id="lag-table"></table>
</div>

<h2>Crossings — where the lines change places</h2>
<div class="card">
<p class="small muted">First year the normalized trade and citation lines cross; start/end side of the comparison window (1995–2013).</p>
<table id="cross-table"></table>
</div>

<h2>What this tells us</h2>
<div class="callout"><strong>China is the unique case.</strong> Trade and knowledge centrality rise in near lockstep from the early 2000s — the only major country where the two lines overlap throughout. The lag analysis says knowledge actually leads by 5 years (r = 0.990): Chinese innovation centrality rises first, and trade follows half a decade later. This is the opposite of the earlier app-level finding, which was an artifact of EPO's 2010 coverage discontinuity.</div>
<div class="callout"><strong>Japan and the US are knowledge economies.</strong> Both sit clearly above their trade lines, and the gaps are widening. Japan's trade and knowledge centrality move in <em>opposite</em> directions (r = −0.927) — its trade position holds while its share of global knowledge grows.</div>
<div class="callout"><strong>Korea is the convergence to watch.</strong> The widest trade lead in the sample (+0.236) but the fastest knowledge catch-up anywhere (trend −0.491). Canada crossed from knowledge to trade around 2009 and is still moving.</div>
<div class="callout"><strong>Sector-level reversal.</strong> Within sectors, China's trade leads knowledge in electronics (lag +3, r = 0.960) and machinery (lag +2, r = 0.869) — the export boom led the knowledge build-up. The country-level knowledge-first result is a composition across sectors. The US flips in pharma (knowledge leads, −5).</div>

<p class="small muted" style="margin-top:32px">Method: PageRank (α = 0.85) on yearly directed graphs, weighted by trade value (BACI) and family-level citation counts (PATSTAT tls228). Chosen over eigenvector centrality because PageRank converges on directed graphs with dead-ends and one-way structure — both present here. Citations stop at 2013 (last reliable year in the spring-2018 PATSTAT edition).</p>
</div>

<script>
const DATA = __JSON_DATA__;
const years = DATA.years;
const W = 240, H = 120, mL = 30, mR = 8, mT = 8, mB = 16;
const ns = "http://www.w3.org/2000/svg";
const tip = document.createElement("div");
tip.className = "tooltip";
document.body.appendChild(tip);

function el(tag, attrs){ const e = document.createElementNS(ns, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }

const grid = document.getElementById("grid");
DATA.countries.forEach((c, ci) => {
  const panel = document.createElement("div");
  panel.className = "panel";
  const tagCls = c.pathway === "Knowledge-leading" ? "k" : (c.pathway === "Trade-leading" ? "t" : "b");
  const tagTxt = c.pathway === "Knowledge-leading" ? "knowledge-first" : (c.pathway === "Trade-leading" ? "trade-first" : "balanced");
  panel.innerHTML = `<h3>${c.name}</h3><span class="tag ${tagCls}">${tagTxt}</span>`;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${c.name}: trade vs citation PageRank`});

  const x = i => mL + i / (years.length - 1) * (W - mL - mR);
  const y = v => mT + (H - mT - mB) * (1 - v);
  const tMax = Math.max(...c.trade_series), cMax = Math.max(...c.cite_series);

  // gridlines
  [0, 0.5, 1].forEach(v => {
    svg.appendChild(el("line", {x1: mL, y1: y(v), x2: W - mR, y2: y(v), stroke: v === 0 ? "#c3c2b7" : "#e1e0d9", "stroke-width": v === 0 ? 1 : 0.8}));
  });
  // x ticks: first, mid, last year
  [0, Math.floor((years.length-1)/2), years.length-1].forEach(i => {
    const t = el("text", {x: x(i), y: H - 4, "text-anchor": "middle", "font-size": 7.5, fill: "#898781"});
    t.textContent = years[i];
    svg.appendChild(t);
  });

  // trade line
  let d = "";
  c.trade_series.forEach((v, i) => { d += (i ? "L" : "M") + x(i) + "," + y(v / tMax); });
  svg.appendChild(el("path", {d, fill: "none", stroke: "#2a78d6", "stroke-width": 1.8, "stroke-linejoin": "round"}));
  // cite line
  let d2 = "";
  c.cite_series.forEach((v, i) => { d2 += (i ? "L" : "M") + x(i) + "," + y(v / cMax); });
  svg.appendChild(el("path", {d: d2, fill: "none", stroke: "#eb6834", "stroke-width": 1.8, "stroke-linejoin": "round"}));

  // hover
  const cross = el("line", {y1: mT, y2: H - mB, stroke: "#898781", "stroke-width": 1, "stroke-dasharray": "2,2", opacity: 0});
  const dA = el("circle", {r: 3.5, fill: "#2a78d6", stroke: "#fcfcfb", "stroke-width": 1.5, opacity: 0});
  const dB = el("circle", {r: 3.5, fill: "#eb6834", stroke: "#fcfcfb", "stroke-width": 1.5, opacity: 0});
  svg.appendChild(cross); svg.appendChild(dA); svg.appendChild(dB);
  svg.appendChild(el("rect", {x: mL, y: mT, width: W - mL - mR, height: H - mT - mB, fill: "transparent"}));

  svg.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (W / r.width);
    const idx = Math.max(0, Math.min(years.length - 1, Math.round((px - mL) / (W - mL - mR) * (years.length - 1))));
    cross.setAttribute("x1", x(idx)); cross.setAttribute("x2", x(idx)); cross.setAttribute("opacity", 1);
    dA.setAttribute("cx", x(idx)); dA.setAttribute("cy", y(c.trade_series[idx] / tMax)); dA.setAttribute("opacity", 1);
    dB.setAttribute("cx", x(idx)); dB.setAttribute("cy", y(c.cite_series[idx] / cMax)); dB.setAttribute("opacity", 1);
    tip.style.opacity = 1;
    tip.style.left = (ev.clientX + 14) + "px";
    tip.style.top = (ev.clientY - 8) + "px";
    tip.innerHTML = `<b>${c.name} ${years[idx]}</b><br>Trade PR: <b>${(c.trade_series[idx]*100).toFixed(1)}</b>% of max<br>Cite PR: <b>${(c.cite_series[idx]*100).toFixed(1)}</b>% of max`;
  });
  svg.addEventListener("mouseleave", () => {
    cross.setAttribute("opacity", 0); dA.setAttribute("opacity", 0); dB.setAttribute("opacity", 0); tip.style.opacity = 0;
  });

  panel.appendChild(svg);
  grid.appendChild(panel);
});

// lag table
(function(){
  const t = document.getElementById("lag-table");
  let html = "<tr><th>Country</th><th>Best lag</th><th>r</th><th>Interpretation</th></tr>";
  [...DATA.countries].sort((a,b) => Math.abs(b.best_r) - Math.abs(a.best_r)).forEach(c => {
    let interp;
    if (Math.abs(c.best_r) < 0.4) interp = "weak signal";
    else if (c.best_r < 0) interp = "opposite movement";
    else if (c.best_lag > 0) interp = "trade leads";
    else if (c.best_lag < 0) interp = "knowledge leads";
    else interp = "synchronous";
    const rCls = c.best_r >= 0 ? "pos" : "neg";
    html += `<tr><td>${c.name}</td><td class="num">${c.best_lag > 0 ? "+" + c.best_lag : c.best_lag}</td><td class="num ${rCls}">${c.best_r >= 0 ? "+" : ""}${c.best_r.toFixed(3)}</td><td>${interp}</td></tr>`;
  });
  t.innerHTML = html;
})();

// crossing table
(function(){
  const t = document.getElementById("cross-table");
  let html = "<tr><th>Country</th><th>First crossing</th><th>Start side</th><th>End side</th><th>Rises first</th></tr>";
  [...DATA.countries].sort((a,b) => (a.first_cross || 9999) - (b.first_cross || 9999)).forEach(c => {
    html += `<tr><td>${c.name}</td><td class="num">${c.first_cross || "—"}</td><td>${c.start_side}</td><td>${c.end_side}</td><td>${c.rises_first}</td></tr>`;
  });
  t.innerHTML = html;
})();
</script>
</body>
</html>
'''

json_str = json.dumps(data)
html = HTML.replace("__JSON_DATA__", json_str)
out_path = "../report/centrality_report.html"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved → {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
