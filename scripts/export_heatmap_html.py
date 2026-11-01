"""
Interactive matrix heatmaps: trade (exporter × importer) vs knowledge
(citing × cited), log-scaled, shared year slider, family-level citations.

Features:
  - two side-by-side log heatmaps (Flows mode)
  - country highlight dropdown (dims all but one country's row+column)
  - Divergence mode: one wide heatmap of (trade share − citation share)
    per pair — red = trade-dominant pair, blue = knowledge-dominant pair
  - color scale legends

Usage: python export_heatmap_html.py [--top-n 20]
Output: ../html/heatmap_standalone.html (self-contained)
"""
import pandas as pd
import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--top-n", type=int, default=20)
args = parser.parse_args()

TOP_N = args.top_n
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
iso_to_name["TW"] = "Taiwan"
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia",
         "Rep. of Korea": "South Korea",
         "China, Hong Kong SAR": "Hong Kong"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

print("[1/3] Loading trade data...")
trade = pd.read_parquet("../rawdata/baci_panel_hs6.parquet",
                        columns=["year", "exporter", "importer", "trade_value"])
trade["source"] = trade["exporter"].map(baci_to_iso)
trade["target"] = trade["importer"].map(baci_to_iso)
trade = trade.dropna(subset=["source", "target"])
trade = trade[trade.source != trade.target]
trade = trade[~trade.source.isin(NON_COUNTRIES) & ~trade.target.isin(NON_COUNTRIES)]
trade = trade.groupby(["year", "source", "target"], as_index=False)["trade_value"].sum()
trade_years = sorted(int(y) for y in trade.year.unique())

print("[2/3] Loading family-level citations...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[cite.year.between(1995, 2017)]
cite = cite.rename(columns={"citing_country": "source",
                            "cited_country": "target",
                            "citation_count": "weight"})
cite = cite[cite.source != cite.target]
cite = cite[~cite.source.isin(NON_COUNTRIES) & ~cite.target.isin(NON_COUNTRIES)]
cite_years = sorted(int(y) for y in cite.year.unique())

tv = trade.groupby("source")["trade_value"].sum().add(
    trade.groupby("target")["trade_value"].sum(), fill_value=0)
cv = cite.groupby("source")["weight"].sum().add(
    cite.groupby("target")["weight"].sum(), fill_value=0)
combined = tv.add(cv, fill_value=0)
top_individuals = combined.nlargest(TOP_N).index.tolist()
countries = top_individuals + ["ROW"]
print(f"  Countries: {countries}")

idx = {c: i for i, c in enumerate(countries)}
ind_set = set(top_individuals)
N = len(countries)

def build_matrices(df, wcol, years):
    out = {}
    for yr in years:
        m = [[0.0] * N for _ in range(N)]
        for _, r in df[df.year == yr].iterrows():
            s = r["source"] if r["source"] in ind_set else "ROW"
            t = r["target"] if r["target"] in ind_set else "ROW"
            if s == t:
                continue
            m[idx[s]][idx[t]] += float(r[wcol])
        out[yr] = m
    return out

print("[3/3] Building matrices + divergence + HTML...")
trade_mat = build_matrices(trade, "trade_value", trade_years)
cite_mat = build_matrices(cite, "weight", cite_years)
all_years = sorted(set(trade_years) | set(cite_years))

# Divergence: (trade share of max) − (citation share of max), per pair-year
divergence = {}
for yr in all_years:
    tm = trade_mat.get(yr)
    cm = cite_mat.get(yr)
    if not tm or not cm:
        continue
    tmax = max(max(row) for row in tm) or 1.0
    cmax = max(max(row) for row in cm) or 1.0
    divergence[yr] = [[(tm[i][j] / tmax) - (cm[i][j] / cmax) for j in range(N)]
                      for i in range(N)]

data = {
    "countries": countries,
    "country_names": {c: (iso_to_name.get(c, c) if c != "ROW" else "Rest of World") for c in countries},
    "trade_years": trade_years,
    "citation_years": cite_years,
    "all_years": all_years,
    "trade": {str(y): trade_mat.get(y, []) for y in all_years},
    "citation": {str(y): cite_mat.get(y, []) for y in all_years},
    "divergence": {str(y): divergence[y] for y in sorted(divergence)},
}
json_str = json.dumps(data)

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade vs Knowledge Matrix Heatmaps</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#1a1a2e;color:#eee;overflow:hidden;height:100vh}
#app{display:flex;height:100vh}
.panel{flex:1;position:relative;padding-bottom:64px;display:flex;flex-direction:column}
#left h2{position:absolute;top:12px;left:18px;font-size:14px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none;color:#4ECDC4;text-align:left}
#right h2{position:absolute;top:12px;right:18px;font-size:14px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none;color:#FF6B6B;text-align:right}
#divpanel h2{position:absolute;top:12px;left:18px;font-size:14px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none;color:#E9C46A;text-align:left}
.wrap{flex:1;display:flex;align-items:center;justify-content:center;padding-top:44px}
svg{width:96%;height:96%}
.legend{position:absolute;bottom:26px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:8px;font-size:9.5px;color:#aaa;pointer-events:none}
.legend .bar{width:120px;height:8px;border-radius:4px}
.legend.dual{left:20px;transform:none}
#divpanel .legend{left:50%;transform:translateX(-50%)}
#controls{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(30,30,50,.95);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:14px 24px;display:flex;gap:16px;align-items:center;z-index:100;backdrop-filter:blur(10px)}
#controls label{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:#aaa}
#year-slider{width:200px;accent-color:#4ECDC4}
#year-display{font-size:18px;font-weight:700;min-width:48px;text-align:center}
#play-btn{background:#4ECDC4;color:#1a1a2e;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-weight:700;font-size:13px}
#play-btn.paused{background:#FF6B6B}
select{background:#2a2a40;color:#eee;border:1px solid rgba(255,255,255,.2);border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
.tooltip{position:fixed;background:rgba(0,0,0,.9);color:#fff;padding:8px 12px;border-radius:6px;font-size:11px;pointer-events:none;z-index:200;opacity:0;transition:opacity .12s}
.note{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);font-size:9px;color:#888;pointer-events:none;white-space:nowrap}
#cite-overlay{position:absolute;top:0;left:0;right:0;bottom:64px;display:none;align-items:center;justify-content:center;z-index:20;pointer-events:none}
#cite-overlay .box{background:rgba(0,0,0,0.72);border:1px solid rgba(255,107,107,0.5);border-radius:10px;padding:16px 28px;text-align:center}
#cite-overlay .a{font-size:15px;font-weight:700;color:#FF6B6B}
#cite-overlay .b{font-size:10px;color:#aaa;margin-top:4px}
</style>
</head>
<body>
<div id="app">
  <div class="panel" id="left"><h2>TRADE FLOWS<br><span style="font-weight:400;font-size:11px;color:#aaa">row = exporter → column = importer</span></h2>
    <div class="wrap"><svg id="tsvg"></svg></div>
    <div class="legend" id="tlegend"><span>$1K</span><div class="bar" id="tbar"></div><span>$1M</span><div class="bar" id="tbar2"></div><span>$1B+</span></div>
    <div class="note">log scale · cross-border only</div>
  </div>
  <div class="panel" id="right"><h2>PATENT CITATIONS<br><span style="font-weight:400;font-size:11px;color:#aaa">row = citing → column = cited</span></h2>
    <div class="wrap"><svg id="csvgsvg"></svg></div>
    <div class="legend" id="clegend"><span>1</span><div class="bar" id="cbar"></div><span>1K</span><div class="bar" id="cbar2"></div><span>1M+</span></div>
    <div class="note">log scale · family-level · cross-border only</div>
    <div id="cite-overlay"><div class="box"><div class="a">Citation data ends 2017</div><div class="b">~2013 last fully reliable year</div></div></div>
  </div>
  <div class="panel" id="divpanel" style="display:none"><h2>DIVERGENCE<br><span style="font-weight:400;font-size:11px;color:#aaa">trade share − citation share per pair (row → column)</span></h2>
    <div class="wrap"><svg id="divsvg"></svg></div>
    <div class="legend" id="dlegend"><span style="color:#3987e5">knowledge-dominant</span><div class="bar" id="dbar"></div><span style="color:#e66767">trade-dominant</span></div>
    <div class="note">unit-free shares of each network's max · cross-border only</div>
  </div>
</div>
<div id="controls">
  <label>Year</label><span id="year-display">2015</span>
  <input type="range" id="year-slider" min="1995" max="2024" value="2015" step="1">
  <button id="play-btn" onclick="togglePlay()">Play</button>
  <select id="mode-select" onchange="setMode(this.value)">
    <option value="flows">Flows</option>
    <option value="divergence">Divergence</option>
  </select>
  <select id="highlight-select" onchange="setHighlight(this.value)"><option value="">All Countries</option></select>
</div>
<div class="tooltip" id="tip"></div>

<script>
const DATA = __JSON_DATA__;
const countries = DATA.countries;
const N = countries.length;
const names = DATA.country_names;
const T_RAMP = ["#B8D8F0","#9AC8E6","#7DB9DC","#5BA3D9","#3D8BC0","#1E6BA8","#0B2D5C"];
const C_RAMP = ["#F5D5B8","#EDC09E","#E0A87A","#D4915A","#C87A4A","#A85A2E","#7A3512"];
const EMPTY = "#23233c";

// legend gradient bars
document.getElementById("tbar").style.background = "linear-gradient(90deg,"+T_RAMP.slice(0,3).join(",")+")";
document.getElementById("tbar2").style.background = "linear-gradient(90deg,"+T_RAMP.slice(3,7).join(",")+")";
document.getElementById("cbar").style.background = "linear-gradient(90deg,"+C_RAMP.slice(0,3).join(",")+")";
document.getElementById("cbar2").style.background = "linear-gradient(90deg,"+C_RAMP.slice(3,7).join(",")+")";
document.getElementById("dbar").style.background = "linear-gradient(90deg,#3987e5,#23233c,#e66767)";

const slider = document.getElementById("year-slider");
slider.min = DATA.all_years[0]; slider.max = DATA.all_years[DATA.all_years.length-1]; slider.value = 2015;
document.getElementById("year-display").textContent = 2015;

const sel = document.getElementById("highlight-select");
countries.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = names[c]||c; sel.appendChild(o); });

let highlight = "";
let mode = "flows";

function colorOf(v, ramp){ if (v <= 0) return EMPTY; const lv = Math.log10(v); return ramp[Math.max(0, Math.min(ramp.length-1, Math.floor(lv)))]; }
function divColor(d){
  // d in [-1, 1]: negative = knowledge-dominant (blue), positive = trade-dominant (red)
  const t = Math.max(-1, Math.min(1, d * 3));  // amplify weak signals
  if (t >= 0) return lerp("#23233c", "#e66767", t);
  return lerp("#23233c", "#3987e5", -t);
}
function lerp(a, b, t){
  const pa = hex(a), pb = hex(b);
  const r = Math.round(pa[0]+(pb[0]-pa[0])*t), g = Math.round(pa[1]+(pb[1]-pa[1])*t), bl = Math.round(pa[2]+(pb[2]-pa[2])*t);
  return `rgb(${r},${g},${bl})`;
}
function hex(h){ return [parseInt(h[1]+h[2],16), parseInt(h[3]+h[4],16), parseInt(h[5]+h[6],16)]; }
function fmtTrade(v){ if(v>=1e6) return "$"+(v/1e6).toFixed(1)+"B"; if(v>=1e3) return "$"+(v/1e3).toFixed(1)+"M"; return "$"+v.toFixed(0)+"K"; }

const NS = "http://www.w3.org/2000/svg";
function el(t, a){ const e = document.createElementNS(NS, t); for(const k in a) e.setAttribute(k, a[k]); return e; }

function cellOpacity(i, j){
  if (!highlight) return 1;
  return (countries[i] === highlight || countries[j] === highlight) ? 1 : 0.12;
}

function drawGrid(svg, mat, colorFn, tooltipFn){
  svg.innerHTML = "";
  const wrap = svg.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const label = 86, pad = 12;
  const cellW = (W - label - pad) / N;
  const cellH = (H - label - pad) / N;
  const off = label + 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  for (let i = 0; i < N; i++){
    const t = el("text", {x: off + i*cellW + cellW/2, y: label - 8,
      transform: `rotate(-45 ${off + i*cellW + cellW/2} ${label - 8})`,
      "text-anchor":"start","font-size":9, fill:"#aaa"});
    t.textContent = countries[i]; svg.appendChild(t);
  }
  for (let i = 0; i < N; i++){
    const t = el("text", {x: label - 4, y: off + i*cellH + cellH/2 + 3,
      "text-anchor":"end","font-size":9, fill:"#aaa"});
    t.textContent = countries[i]; svg.appendChild(t);
  }
  for (let i = 0; i < N; i++){
    for (let j = 0; j < N; j++){
      const v = mat[i] ? mat[i][j] : 0;
      const r = el("rect", {x: off + j*cellW + 1, y: off + i*cellH + 1,
        width: cellW - 2, height: cellH - 2, rx: 2, fill: colorFn(v), opacity: cellOpacity(i, j)});
      r.addEventListener("mousemove", ev => {
        const tip = document.getElementById("tip");
        tip.style.opacity = 1; tip.style.left = (ev.clientX + 14) + "px"; tip.style.top = (ev.clientY - 10) + "px";
        tip.innerHTML = tooltipFn(i, j, v);
      });
      r.addEventListener("mouseleave", () => { document.getElementById("tip").style.opacity = 0; });
      svg.appendChild(r);
    }
  }
}

function update(){
  const yr = +slider.value;
  document.getElementById("year-display").textContent = yr;

  if (mode === "flows"){
    document.getElementById("left").style.display = "flex";
    document.getElementById("right").style.display = "flex";
    document.getElementById("divpanel").style.display = "none";
    const tm = DATA.trade[String(yr)] || [];
    const cm = DATA.citation[String(yr)] || [];
    drawGrid(document.getElementById("tsvg"), tm,
      v => colorOf(v, T_RAMP),
      (i,j,v) => `<b>${names[countries[i]]}</b> → <b>${names[countries[j]]}</b><br>${fmtTrade(v)}`);
    drawGrid(document.getElementById("csvgsvg"), cm,
      v => colorOf(v, C_RAMP),
      (i,j,v) => `<b>${names[countries[i]]}</b> → <b>${names[countries[j]]}</b><br>${v.toLocaleString("en-US")} citations`);
    const maxCite = DATA.citation_years[DATA.citation_years.length-1];
    document.getElementById("cite-overlay").style.display = (yr > maxCite) ? "flex" : "none";
  } else {
    document.getElementById("left").style.display = "none";
    document.getElementById("right").style.display = "none";
    document.getElementById("divpanel").style.display = "flex";
    const dm = DATA.divergence[String(yr)] || [];
    drawGrid(document.getElementById("divsvg"), dm,
      v => v === 0 ? EMPTY : divColor(v),
      (i,j,v) => {
        const t = DATA.trade[String(yr)][i][j];
        const c = DATA.citation[String(yr)][i][j];
        const side = Math.abs(v) < 0.02 ? "balanced" : (v > 0 ? "trade-dominant" : "knowledge-dominant");
        return `<b>${names[countries[i]]}</b> → <b>${names[countries[j]]}</b><br>` +
          `trade: ${fmtTrade(t)} · cites: ${(c||0).toLocaleString("en-US")}<br>${side} (d=${v.toFixed(3)})`;
      });
  }
}

window.setMode = function(m){ mode = m; update(); };
window.setHighlight = function(c){ highlight = c; update(); };
slider.addEventListener("input", update);
window._pi = null;
window.togglePlay = function(){
  const btn = document.getElementById("play-btn");
  if (window._pi){ clearInterval(window._pi); window._pi = null; btn.textContent = "Play"; btn.classList.remove("paused"); return; }
  btn.textContent = "Pause"; btn.classList.add("paused");
  window._pi = setInterval(() => {
    let v = +slider.value + 1;
    if (v > +slider.max) v = +slider.min;
    slider.value = v; update();
  }, 450);
};
update();
</script>
</body>
</html>
'''

html = HTML.replace("__JSON_DATA__", json_str)
out_path = "../html/heatmap_standalone.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved → {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
