"""
Interactive circular network: highlight country-pair arcs across ALL years.
Year slider + auto-animation on pair selection.

Output: interactive_comparison.html
"""
import pandas as pd
import numpy as np
import json
from math import pi, sin, cos, atan2

TOP_N = 16
MAX_EDGES = 100
YEARS = list(range(1995, 2025))  # trade to 2024, citations end ~2017
NON_COUNTRIES = {"EP","WO","DD","OA","AP","GC","IB","EM","BX","YU","CS","SU"}

# ---- Load mappings ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
name_to_iso["Other Asia, nes"] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
iso_to_name["TW"] = "Taiwan"

# ---- Helper ----
def top_nodes(df, wcol, n):
    a = df[["source", wcol]].rename(columns={"source": "c", wcol: "w"})
    b = df[["target", wcol]].rename(columns={"target": "c", wcol: "w"})
    return pd.concat([a, b]).groupby("c")["w"].sum().nlargest(n).index.tolist()

# ---- Determine fixed node set (union of top countries across all years) ----
print("Building fixed node set across all years...")
all_trade_nodes = set()
all_cite_nodes = set()
trade_edges_by_year = {}
cite_edges_by_year = {}

for year in YEARS:
    # Trade
    tdf = pd.read_csv(f"../processed/trade_edges_{year}.csv")
    tdf["source"] = tdf["exporter"].map(baci_to_iso)
    tdf["target"] = tdf["importer"].map(baci_to_iso)
    tdf["source"] = tdf["source"].fillna(tdf["exporter_name"].map(name_to_iso))
    tdf["target"] = tdf["target"].fillna(tdf["importer_name"].map(name_to_iso))
    tdf = tdf.dropna(subset=["source", "target"])
    tdf = tdf[tdf["source"] != tdf["target"]]
    tdf = tdf.groupby(["source", "target"], as_index=False)["trade_value"].sum()
    all_trade_nodes.update(top_nodes(tdf, "trade_value", TOP_N))

    # Citation
    cdf = pd.read_csv("../processed/country_citation_edges_by_year.csv")
    cdf = cdf.rename(columns={"citing_country": "source", "cited_country": "target",
                               "citation_count": "weight"})
    cdf = cdf[(cdf["year"] == year) & (~cdf["source"].isin(NON_COUNTRIES))
              & (~cdf["target"].isin(NON_COUNTRIES))]
    cdf = cdf[cdf["source"] != cdf["target"]]
    all_cite_nodes.update(top_nodes(cdf, "weight", TOP_N))

all_nodes = list(dict.fromkeys(all_trade_nodes | all_cite_nodes))[:TOP_N + 8]
n_nodes = len(all_nodes)
print(f"  {n_nodes} fixed nodes")

# ---- Fixed layout ----
# Use 2015 volumes for ordering
tdf15 = pd.read_csv("../processed/trade_edges_2015.csv")
tdf15["source"] = tdf15["exporter"].map(baci_to_iso)
tdf15["target"] = tdf15["importer"].map(baci_to_iso)
tdf15["source"] = tdf15["source"].fillna(tdf15["exporter_name"].map(name_to_iso))
tdf15["target"] = tdf15["target"].fillna(tdf15["importer_name"].map(name_to_iso))
tdf15 = tdf15.dropna(subset=["source", "target"])
tdf15 = tdf15[tdf15["source"] != tdf15["target"]]

cdf15 = pd.read_csv("../processed/country_citation_edges_by_year.csv")
cdf15 = cdf15.rename(columns={"citing_country": "source", "cited_country": "target",
                               "citation_count": "weight"})
cdf15 = cdf15[(cdf15["year"] == 2015) & (~cdf15["source"].isin(NON_COUNTRIES))
              & (~cdf15["target"].isin(NON_COUNTRIES))]
cdf15 = cdf15[cdf15["source"] != cdf15["target"]]

def vol(node):
    tv = sum(r["trade_value"] for _, r in tdf15[tdf15["source"] == node].iterrows())
    tv += sum(r["trade_value"] for _, r in tdf15[tdf15["target"] == node].iterrows())
    cv = sum(r["weight"] for _, r in cdf15[cdf15["source"] == node].iterrows())
    cv += sum(r["weight"] for _, r in cdf15[cdf15["target"] == node].iterrows())
    return tv + cv

anchors = ["CN", "US", "DE", "JP"]
anchor_pos = [0, n_nodes // 4, 2 * n_nodes // 4, 3 * n_nodes // 4]
amap = {a: p for a, p in zip(anchors, anchor_pos) if a in all_nodes}
rem = sorted([n for n in all_nodes if n not in amap], key=vol, reverse=True)
avail = [i for i in range(n_nodes) if i not in amap.values()]
ordered = [None] * n_nodes
for a, p in amap.items():
    ordered[p] = a
for i, n in enumerate(rem):
    ordered[avail[i]] = n

pos = {}
for i, n in enumerate(ordered):
    angle = 2 * pi * i / n_nodes - pi / 2
    pos[n] = (cos(angle), sin(angle))

# ---- Bezier helper ----
def bezier_path(p1, p2):
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = np.hypot(dx, dy)
    if L < 0.01:
        return [[p1[0], p1[1]], [p2[0], p2[1]]]
    mid_dist = np.hypot(mx, my)
    bulge = max(0.2, 0.6 / (L + 0.3))
    bulge = min(bulge, mid_dist * 0.85)
    if mid_dist > 0.001:
        ndx, ndy = -mx / mid_dist, -my / mid_dist
    else:
        ndx, ndy = -dy / L, dx / L
    cx, cy = mx + ndx * bulge, my + ndy * bulge
    pts = []
    for t in np.linspace(0, 1, 20):
        mt = 1 - t
        pts.append([float(mt * mt * p1[0] + 2 * mt * t * cx + t * t * p2[0]),
                     float(mt * mt * p1[1] + 2 * mt * t * cy + t * t * p2[1])])
    return pts

# ---- Build edges by year ----
print("Building edges for all years...")
for year in YEARS:
    # Trade
    tdf = pd.read_csv(f"../processed/trade_edges_{year}.csv")
    tdf["source"] = tdf["exporter"].map(baci_to_iso)
    tdf["target"] = tdf["importer"].map(baci_to_iso)
    tdf["source"] = tdf["source"].fillna(tdf["exporter_name"].map(name_to_iso))
    tdf["target"] = tdf["target"].fillna(tdf["importer_name"].map(name_to_iso))
    tdf = tdf.dropna(subset=["source", "target"])
    tdf = tdf[tdf["source"] != tdf["target"]]
    tdf = tdf.groupby(["source", "target"], as_index=False)["trade_value"].sum()
    sub = tdf[tdf["source"].isin(all_nodes) & tdf["target"].isin(all_nodes)]
    sub = sub.nlargest(MAX_EDGES, "trade_value")
    edges = []
    for _, r in sub.iterrows():
        p1, p2 = pos.get(r["source"]), pos.get(r["target"])
        if p1 and p2:
            edges.append({"source": r["source"], "target": r["target"],
                          "value": float(r["trade_value"]),
                          "path": bezier_path(p1, p2)})
    trade_edges_by_year[str(year)] = edges

    # Citation
    cdf = pd.read_csv("../processed/country_citation_edges_by_year.csv")
    cdf = cdf.rename(columns={"citing_country": "source", "cited_country": "target",
                               "citation_count": "weight"})
    cdf = cdf[(cdf["year"] == year) & (~cdf["source"].isin(NON_COUNTRIES))
              & (~cdf["target"].isin(NON_COUNTRIES))]
    cdf = cdf[cdf["source"] != cdf["target"]]
    sub = cdf[cdf["source"].isin(all_nodes) & cdf["target"].isin(all_nodes)]
    sub = sub.nlargest(MAX_EDGES, "weight")
    edges = []
    for _, r in sub.iterrows():
        p1, p2 = pos.get(r["source"]), pos.get(r["target"])
        if p1 and p2:
            edges.append({"source": r["source"], "target": r["target"],
                          "value": float(r["weight"]),
                          "path": bezier_path(p1, p2)})
    cite_edges_by_year[str(year)] = edges

print(f"  {sum(len(v) for v in trade_edges_by_year.values())} trade edge-snapshots")
print(f"  {sum(len(v) for v in cite_edges_by_year.values())} citation edge-snapshots")

# ---- Country dropdown ----
country_opts = [{"label": f"{c} ({iso_to_name.get(c, c)})", "value": c}
                for c in sorted(all_nodes) if c in iso_to_name]

# ---- Build HTML ----
data_js = {
    "nodes": list(all_nodes),
    "positions": {k: [float(v[0]), float(v[1])] for k, v in pos.items()},
    "tradeEdgesByYear": trade_edges_by_year,
    "citeEdgesByYear": cite_edges_by_year,
    "years": YEARS,
    "defaultYear": 2015,
    "countryOpts": country_opts,
    "isoNames": iso_to_name,
}

print(f"  Data JSON: {len(json.dumps(data_js)):,} bytes")

# ---- HTML template (same as before, with year slider) ----
with open("../html/interactive_comparison.html", "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Trade & Citation — Interactive Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0}
#wrap{display:flex;height:100vh}
#sidebar{width:250px;min-width:250px;background:rgba(25,25,45,0.98);padding:16px;display:flex;flex-direction:column;gap:10px;
  border-right:1px solid #333;overflow-y:auto;z-index:10}
#sidebar h2{font-size:15px;color:#FFD166;margin-bottom:4px}
#sidebar label{font-size:10px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.5px}
#sidebar select,#sidebar input,#sidebar button{padding:8px 10px;border-radius:6px;border:1px solid #444;background:#2a2a40;color:#e0e0e0;font-size:12px;cursor:pointer;width:100%}
#sidebar button{font-weight:600;border:none}
#sidebar input[type=range]{padding:0;height:6px;-webkit-appearance:none;appearance:none;background:#444;outline:none;border-radius:3px}
#sidebar input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#FFD166;cursor:pointer}
.year-display{text-align:center;font-size:22px;font-weight:700;color:#FFD166}
.btn-stop{background:#555!important;color:white!important}
.btn-reset{background:#444!important;color:#ccc!important}
#info-card{background:#2a2a40;border-radius:8px;padding:12px;font-size:11px;line-height:1.6}
#info-card .val{font-size:16px;font-weight:700;color:#FFD166}
#info-card .trade-val{color:#3A7CA5}
#info-card .cite-val{color:#C87A4A}
#chart{flex:1;min-width:0}
</style>
</head>
<body>
<div id="wrap">
<div id="sidebar">
  <h2>🔍 Country Pair Explorer</h2>
  <label>Year</label>
  <div class="year-display" id="year-label">2015</div>
  <input type="range" id="year-slider" min="1995" max="2024" value="2015" step="1">
  <div style="display:flex;gap:6px">
    <button id="btn-prev" style="flex:1;background:#444;color:white">◀</button>
    <button id="btn-next" style="flex:1;background:#444;color:white">▶</button>
  </div>
  <label>Preset pairs</label>
  <select id="preset"><option value="">— Show all edges —</option></select>
  <label>Source (FROM)</label>
  <select id="src-sel"><option value="__all__">All Countries</option><option value="">— Pick —</option></select>
  <label>Target (TO)</label>
  <select id="tgt-sel"><option value="__all__">All Countries</option><option value="">— Pick —</option></select>
  <button class="btn-stop" id="btn-stop">⏹ Stop Animation</button>
  <button class="btn-reset" id="btn-reset">↺ Reset</button>
  <div id="info-card"><small style="color:#888">Select a pair to see flow details</small></div>
  <div style="font-size:10px;color:#666;margin-top:auto">
    Trade arrow → importer (buyer)<br>
    Citation arrow → cited country<br>
    <span style="color:#3A7CA5">●</span> Trade | <span style="color:#C87A4A">●</span> Citation
  </div>
</div>
<div id="chart"></div>
</div>
""")

    f.write("\n<script>\n")
    f.write(f"const DATA = {json.dumps(data_js)};\n")
    f.write("""
// ---- State ----
let highlightSrc = null, highlightTgt = null, currentYear = DATA.defaultYear;
let pulseTimer = null, pulsePhase = 0;

// ---- Dropdowns ----
const preset = document.getElementById('preset');
const srcSel = document.getElementById('src-sel');
const tgtSel = document.getElementById('tgt-sel');
const yearSlider = document.getElementById('year-slider');
const yearLabel = document.getElementById('year-label');
const infoCard = document.getElementById('info-card');

const topPairs = [
  ['CN','US'],['US','CN'],['DE','US'],['US','JP'],
  ['CN','JP'],['JP','CN'],['DE','GB'],['US','KR'],
  ['CN','DE'],['JP','US'],['GB','US'],['KR','US'],
  ['FR','DE'],['DE','CN'],['CN','KR'],['US','DE'],
  ['TW','CN'],['TW','US'],['CN','HK'],['DE','FR'],
];
topPairs.forEach(([s,t]) => {
  const o = document.createElement('option');
  o.value = s + '->' + t;
  o.textContent = DATA.isoNames[s] + ' → ' + DATA.isoNames[t];
  preset.appendChild(o);
});

DATA.countryOpts.forEach(c => {
  srcSel.innerHTML += `<option value="${c.value}">${c.label}</option>`;
  tgtSel.innerHTML += `<option value="${c.value}">${c.label}</option>`;
});

// ---- Helpers ----
function formatVal(v, type) {
  if(type==='trade'){
    if(v>=1e12) return '$'+(v/1e12).toFixed(2)+'T';
    if(v>=1e9) return '$'+(v/1e9).toFixed(1)+'B';
    return '$'+(v/1e6).toFixed(1)+'M';
  } else {
    if(v>=1e6) return (v/1e6).toFixed(1)+'M';
    if(v>=1e3) return (v/1e3).toFixed(1)+'K';
    return v.toLocaleString();
  }
}

function getSelected(){
  const p = preset.value;
  if(p){ const [s,t]=p.split('->'); return {src:s,tgt:t}; }
  const s = srcSel.value, t = tgtSel.value;
  if(!s && !t) return null;
  if(s==='__all__' && t==='__all__') return null; // all->all = no filter
  return {src: s || null, tgt: t || null};
}

function edgeMatches(e, src, tgt){
  // src/tgt can be: specific country, '__all__', null (not set), or '' (not set)
  const sMatch = !src || src==='__all__' || src==='' || e.source===src;
  const tMatch = !tgt || tgt==='__all__' || tgt==='' || e.target===tgt;
  return sMatch && tMatch;
}

function findEdge(edges, src, tgt){
  return edges.find(e=>e.source===src&&e.target===tgt) || null;
}

function getEdges(type){
  const y = String(currentYear);
  const all = type==='trade' ? DATA.tradeEdgesByYear : DATA.citeEdgesByYear;
  return all[y] || [];
}

// ---- Build Plotly traces ----
function addPulseDot(traces, src, tgt, type, phase, offsetX, xaxis, yaxis){
  const edges = getEdges(type);
  const e = findEdge(edges, src, tgt);
  if(!e) return;
  const idx = Math.floor(phase * e.path.length);
  const px = e.path[idx][0] + offsetX, py = e.path[idx][1];
  const pc = offsetX===0 ? '#3A7CA5' : '#C87A4A';
  traces.push({
    x:[px], y:[py], mode:'markers', type:'scatter',
    marker:{size:14,color:pc,symbol:'circle',opacity:0.95,
            line:{color:'white',width:1.5}},
    showlegend:false, hoverinfo:'skip', xaxis:xaxis, yaxis:yaxis
  });
}

function buildTraces(highlightSrc, highlightTgt, pulseData){
  const traces = [];
  const tradeC = '#3A7CA5', citeC = '#C87A4A', hlC = '#FFD166';
  const hasHL = !!(highlightSrc || highlightTgt);
  const cOff = 3.0;
  const tradeEdges = getEdges('trade'), citeEdges = getEdges('cite');

  // Compute edge width scaling (log scale)
  function scaleWidths(edges, minW, maxW){
    if(!edges.length) return {};
    const vals = edges.map(e=>Math.log1p(e.value));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    if(hi===lo) return {};
    const map = {};
    edges.forEach(e=>{ map[e.source+'->'+e.target] = minW + (maxW-minW)*(Math.log1p(e.value)-lo)/(hi-lo); });
    return map;
  }
  const tradeW = scaleWidths(tradeEdges, 0.3, 5.5);
  const citeW = scaleWidths(citeEdges, 0.3, 5.5);

  // ---- TRADE PANEL ----
  const tx=[], ty=[], tl=[];
  DATA.nodes.forEach(n=>{const p=DATA.positions[n]; tx.push(p[0]); ty.push(p[1]); tl.push(n);});
  traces.push({
    x:tx, y:ty, mode:'markers+text', type:'scatter',
    text:tl, textposition:'top center', textfont:{size:9,color:'#ccc'},
    marker:{size:16,color:tradeC,opacity:0.7,line:{color:'#555',width:0.8}},
    showlegend:false, hoverinfo:'skip', xaxis:'x', yaxis:'y'
  });

  tradeEdges.forEach(e=>{
    const xs=e.path.map(p=>p[0]), ys=e.path.map(p=>p[1]);
    const isHL = hasHL && edgeMatches(e, highlightSrc, highlightTgt);
    const w = tradeW[e.source+'->'+e.target] || 0.8;
    traces.push({
      x:xs, y:ys, mode:'lines', type:'scatter',
      line:{color:isHL?hlC:tradeC, width:isHL?Math.max(w*1.3,2.5):w},
      opacity: isHL ? 1.0 : (hasHL ? 0.06 : 0.28),
      showlegend:false, hoverinfo:'skip', xaxis:'x', yaxis:'y'
    });
  });

  // ---- CITATION PANEL ----
  const cx=[], cy=[], cl=[];
  DATA.nodes.forEach(n=>{const p=DATA.positions[n]; cx.push(p[0]+cOff); cy.push(p[1]); cl.push(n);});
  traces.push({
    x:cx, y:cy, mode:'markers+text', type:'scatter',
    text:cl, textposition:'top center', textfont:{size:9,color:'#ccc'},
    marker:{size:16,color:citeC,opacity:0.7,line:{color:'#555',width:0.8}},
    showlegend:false, hoverinfo:'skip', xaxis:'x2', yaxis:'y2'
  });

  citeEdges.forEach(e=>{
    const xs=e.path.map(p=>p[0]+cOff), ys=e.path.map(p=>p[1]);
    const isHL = hasHL && edgeMatches(e, highlightSrc, highlightTgt);
    const w = citeW[e.source+'->'+e.target] || 0.8;
    traces.push({
      x:xs, y:ys, mode:'lines', type:'scatter',
      line:{color:isHL?hlC:citeC, width:isHL?Math.max(w*1.3,2.5):w},
      opacity: isHL ? 1.0 : (hasHL ? 0.06 : 0.28),
      showlegend:false, hoverinfo:'skip', xaxis:'x2', yaxis:'y2'
    });
  });

  // ---- Pulse dots on all highlighted edges ----
  if(pulseData){
    const tradeHL = tradeEdges.filter(e=>edgeMatches(e,pulseData.src,pulseData.tgt));
    const citeHL = citeEdges.filter(e=>edgeMatches(e,pulseData.src,pulseData.tgt));
    tradeHL.forEach(e => {
      const idx = Math.floor(pulseData.phase * e.path.length);
      traces.push({
        x:[e.path[idx][0]], y:[e.path[idx][1]], mode:'markers', type:'scatter',
        marker:{size:10,color:'#3A7CA5',symbol:'circle',opacity:0.9,line:{color:'white',width:1}},
        showlegend:false, hoverinfo:'skip', xaxis:'x', yaxis:'y'
      });
    });
    citeHL.forEach(e => {
      const idx = Math.floor(pulseData.phase * e.path.length);
      traces.push({
        x:[e.path[idx][0]+cOff], y:[e.path[idx][1]], mode:'markers', type:'scatter',
        marker:{size:10,color:'#C87A4A',symbol:'circle',opacity:0.9,line:{color:'white',width:1}},
        showlegend:false, hoverinfo:'skip', xaxis:'x2', yaxis:'y2'
      });
    });
  }

  return traces;
}

function makeLayout(highlightSrc, highlightTgt){
  const hasHL = !!(highlightSrc && highlightTgt);
  let title = '<b style="color:#3A7CA5">📦 TRADE</b> &nbsp;vs&nbsp; <b style="color:#C87A4A">📚 CITATION</b>';
  if(hasHL){
    const sn = DATA.isoNames[highlightSrc] || highlightSrc;
    const tn = DATA.isoNames[highlightTgt] || highlightTgt;
    title += ' — <b style="color:#FFD166">' + sn + ' → ' + tn + '</b>';
  }
  title += ' &nbsp;<span style="font-size:14px;color:#aaa">Year: ' + currentYear + '</span>';
  return {
    title:{text:title,font:{size:15,color:'#e0e0e0'},x:0.5},
    showlegend:false,
    paper_bgcolor:'#1a1a2e', plot_bgcolor:'#1a1a2e',
    margin:{l:20,r:20,t:50,b:20},
    xaxis:{domain:[0,0.48],range:[-1.4,1.4],showgrid:false,zeroline:false,showticklabels:false,fixedrange:true},
    yaxis:{domain:[0,1],range:[-1.4,1.4],showgrid:false,zeroline:false,showticklabels:false,fixedrange:true,scaleanchor:'x'},
    xaxis2:{domain:[0.52,1],range:[3.0-1.4,3.0+1.4],showgrid:false,zeroline:false,showticklabels:false,fixedrange:true},
    yaxis2:{domain:[0,1],range:[-1.4,1.4],showgrid:false,zeroline:false,showticklabels:false,fixedrange:true},
    shapes:[
      {type:'circle',xref:'x',yref:'y',x0:-1.2,y0:-1.2,x1:1.2,y1:1.2,
       line:{color:'#3A7CA5',width:1,dash:'dot'},opacity:0.25},
      {type:'circle',xref:'x2',yref:'y2',x0:3.0-1.2,y0:-1.2,x1:3.0+1.2,y1:1.2,
       line:{color:'#C87A4A',width:1,dash:'dot'},opacity:0.25},
    ]
  };
}

// ---- Update ----
function update(pulseData){
  const sel = getSelected();
  highlightSrc = sel ? sel.src : null;
  highlightTgt = sel ? sel.tgt : null;
  yearLabel.textContent = currentYear;
  yearSlider.value = currentYear;

  const traces = buildTraces(highlightSrc, highlightTgt, pulseData || null);
  const layout = makeLayout(highlightSrc, highlightTgt);
  Plotly.react('chart', traces, layout, {responsive:true, displayModeBar:false});

  if(highlightSrc || highlightTgt){
    const tradeE = getEdges('trade'), citeE = getEdges('cite');
    const hlTrade = tradeE.filter(e=>edgeMatches(e,highlightSrc,highlightTgt));
    const hlCite = citeE.filter(e=>edgeMatches(e,highlightSrc,highlightTgt));
    const tv = hlTrade.reduce((s,e)=>s+e.value,0);
    const cv = hlCite.reduce((s,e)=>s+e.value,0);
    let label = '';
    if(highlightSrc && highlightSrc!=='__all__') label += (DATA.isoNames[highlightSrc]||highlightSrc);
    else label += 'All';
    label += ' → ';
    if(highlightTgt && highlightTgt!=='__all__') label += (DATA.isoNames[highlightTgt]||highlightTgt);
    else label += 'All';
    infoCard.innerHTML = `
      <strong style="color:#FFD166">${label}</strong>
      <span style="color:#aaa"> (${currentYear})</span><br>
      <span class="trade-val">📦 Trade: ${formatVal(tv,'trade')} (${hlTrade.length} flows)</span><br>
      <span class="cite-val">📚 Citations: ${formatVal(cv,'cite')} (${hlCite.length} flows)</span>
    `;
  } else {
    infoCard.innerHTML = '<small style="color:#888">Select source/target to highlight flows.<br>Use "All Countries" to see all flows from/to a country.<br>Yellow arcs = highlighted.</small>';
  }
}

// ---- Animation ----
function startAutoPulse(){
  if(!getSelected()) return;
  stopPulse();
  pulsePhase = 0;
  pulseTimer = setInterval(() => {
    pulsePhase = (pulsePhase + 0.025) % 1;
    const sel = getSelected();
    update(sel ? {src:sel.src, tgt:sel.tgt, phase:pulsePhase} : null);
  }, 35);
}

function stopPulse(){
  if(pulseTimer){ clearInterval(pulseTimer); pulseTimer = null; }
  update();
}

// ---- Event bindings ----
preset.addEventListener('change', () => {
  const sel = getSelected();
  if(sel){ srcSel.value = sel.src; tgtSel.value = sel.tgt; }
  stopPulse();
  if(sel) startAutoPulse();
});
srcSel.addEventListener('change', () => {
  preset.value = ''; stopPulse();
  if(srcSel.value && tgtSel.value) startAutoPulse();
  else update();
});
tgtSel.addEventListener('change', () => {
  preset.value = ''; stopPulse();
  if(srcSel.value && tgtSel.value) startAutoPulse();
  else update();
});
yearSlider.addEventListener('input', () => {
  currentYear = parseInt(yearSlider.value);
  if(pulseTimer){ const s=getSelected(); update(s?{src:s.src,tgt:s.tgt,phase:pulsePhase}:null); }
  else update();
});
document.getElementById('btn-prev').addEventListener('click', ()=>{
  currentYear = Math.max(1995, currentYear-1);
  if(pulseTimer){ const s=getSelected(); update(s?{src:s.src,tgt:s.tgt,phase:pulsePhase}:null); }
  else update();
});
document.getElementById('btn-next').addEventListener('click', ()=>{
  currentYear = Math.min(2024, currentYear+1);
  if(pulseTimer){ const s=getSelected(); update(s?{src:s.src,tgt:s.tgt,phase:pulsePhase}:null); }
  else update();
});
document.getElementById('btn-stop').addEventListener('click', stopPulse);
document.getElementById('btn-reset').addEventListener('click', ()=>{
  stopPulse();
  preset.value=''; srcSel.value=''; tgtSel.value=''; update();
});

// ---- Init ----
update();
</script>
</body>
</html>""")

print(f"Saved -> interactive_comparison.html")
