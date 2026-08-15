"""
Export Level B chord data as JSON and generate standalone interactive HTML.
Level B = country×industry nodes, filtered to one HS2/CPC sector pair.

Usage:
  python export_chord_level_b.py                              # Electronics, top-15
  python export_chord_level_b.py --sector vehicles --top-n 12
  python export_chord_level_b.py --sector pharma --json-only   # skip HTML
  python export_chord_level_b.py --list-sectors

Output:
  ../html/chord_data_level_b_{sector}.json                    # chord matrix data
  ../html/chord_level_b_{sector}_standalone.html              # self-contained interactive
"""
import pandas as pd
import numpy as np
import json
import argparse
import os
import sys

# ---- Sector concordance ----
SECTORS = {
    "electronics": {"hs2": "85", "hs_label": "Electronics (HS85)",
                    "cpc": "H",  "cpc_label": "Electricity (CPC H)", "name": "Electronics"},
    "vehicles":    {"hs2": "87", "hs_label": "Vehicles (HS87)",
                    "cpc": "B",  "cpc_label": "Ops & Transport (CPC B)", "name": "Vehicles"},
    "pharma":      {"hs2": "30", "hs_label": "Pharma (HS30)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "name": "Pharma"},
    "machinery":   {"hs2": "84", "hs_label": "Machinery (HS84)",
                    "cpc": "F",  "cpc_label": "Engineering (CPC F)", "name": "Machinery"},
    "oil_fuels":   {"hs2": "27", "hs_label": "Oil & Fuels (HS27)",
                    "cpc": "B",  "cpc_label": "Ops & Transport (CPC B)", "name": "Oil & Fuels"},
    "iron_steel":  {"hs2": "72", "hs_label": "Iron & Steel (HS72)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "name": "Iron & Steel"},
    "plastics":    {"hs2": "39", "hs_label": "Plastics (HS39)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "name": "Plastics"},
    "apparel":     {"hs2": "62", "hs_label": "Apparel (HS62)",
                    "cpc": "D",  "cpc_label": "Textiles (CPC D)", "name": "Apparel"},
}

COLORS = ["#E63946", "#3A7CA5", "#2A9D8F", "#F4A261", "#9B5DE5", "#E76F51",
          "#264653", "#A8DADC", "#E9C46A", "#6D597A", "#B5838D", "#4895EF",
          "#FF6B6B", "#4ECDC4", "#FFD166", "#F15BB5", "#00BBF9", "#FEE440",
          "#D4A373", "#073B4C"]

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

parser = argparse.ArgumentParser()
parser.add_argument("--sector", type=str, default="electronics")
parser.add_argument("--top-n", type=int, default=15,
                    help="Number of individual countries (rest grouped as ROW)")
parser.add_argument("--output-dir", type=str, default="../html")
parser.add_argument("--json-only", action="store_true",
                    help="Export JSON only, skip HTML generation")
parser.add_argument("--list-sectors", action="store_true")
args = parser.parse_args()

if args.list_sectors:
    print("Level B sector pairs:\n")
    for key, s in SECTORS.items():
        print(f"  {key:<14} Trade: {s['hs_label']:<22} ↔  Citation: {s['cpc_label']}")
    sys.exit(0)

SECTOR = SECTORS.get(args.sector)
if SECTOR is None:
    print(f"Unknown sector '{args.sector}'. Use --list-sectors.")
    sys.exit(1)

TOP_N = args.top_n
INDIV_N = TOP_N - 1  # last slot is ROW
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load country codes ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"  # Taiwan
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
iso_to_name["TW"] = "Taiwan"

# Clean historical/verbose country names for display
CLEAN_NAMES = {
    "Fed. Rep. of Germany (...1990)": "Germany",
    "Belgium-Luxembourg (...1998)": "Belgium",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Mayotte (Overseas France)": "Mayotte",
    "Falkland Isds (Malvinas)": "Falkland Islands",
    "Russian Federation": "Russia",
    "Netherlands Antilles (...2010)": "Netherlands Antilles",
    "Sudan (...2011)": "Sudan",
    "Dem. Rep. of Germany (...1990)": "East Germany",
    "Czechoslovakia (...1992)": "Czechoslovakia",
    "USSR (...1990)": "USSR",
    "Serbia and Montenegro (...2005)": "Serbia and Montenegro",
    "Southern African Customs Union (...1999)": "S. African Customs Union",
}
for old, new in CLEAN_NAMES.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

# ============================================================
# 1. Load and filter trade data by HS2 chapter
# ============================================================
print(f"=== Level B Chord: {SECTOR['name']} ===")
print(f"Trade: {SECTOR['hs_label']}  |  Citation: {SECTOR['cpc_label']}")
print(f"Top {INDIV_N} countries + ROW\n")

print("[1/4] Loading trade data...")
trade_cols = ["year", "exporter", "importer", "hs2", "trade_value"]
trade_raw = pd.read_parquet("../rawdata/baci_panel_hs6.parquet", columns=trade_cols)
# Filter by HS2 chapter (hs2 is Int32, e.g. 85)
hs2_int = int(SECTOR["hs2"])
trade_raw = trade_raw[trade_raw["hs2"] == hs2_int]
print(f"  {len(trade_raw):,} trade rows for HS{hs2_int} (all years)")

# Map to ISO2 and aggregate
trade_raw["source"] = trade_raw["exporter"].map(baci_to_iso)
trade_raw["target"] = trade_raw["importer"].map(baci_to_iso)
trade_raw = trade_raw.dropna(subset=["source", "target"])
trade_raw = trade_raw[trade_raw["source"] != trade_raw["target"]]
trade_raw = trade_raw[~trade_raw["source"].isin(NON_COUNTRIES) &
                      ~trade_raw["target"].isin(NON_COUNTRIES)]
trade_agg = trade_raw.groupby(
    ["year", "source", "target"], as_index=False)["trade_value"].sum()
del trade_raw
trade_years = sorted(trade_agg["year"].unique())
trade_min, trade_max = int(trade_years[0]), int(trade_years[-1])
print(f"  {len(trade_agg):,} aggregated flows, years {trade_min}–{trade_max}")

# ============================================================
# 2. Load and filter citation data by CPC section
# ============================================================
print("\n[2/4] Loading citation data...")
cite_raw = pd.read_csv("../processed/country_citation_family_by_cpc.csv")
cite_raw = cite_raw[cite_raw["cpc_section"] == SECTOR["cpc"]]
cite_raw = cite_raw.rename(columns={
    "citing_country": "source", "cited_country": "target",
    "citation_count": "weight"
})
cite_raw = cite_raw[~cite_raw["source"].isin(NON_COUNTRIES) &
                    ~cite_raw["target"].isin(NON_COUNTRIES)]
cite_raw = cite_raw[cite_raw["source"] != cite_raw["target"]]
cite_years = sorted(cite_raw["year"].unique())
cite_overlap = [y for y in cite_years if 1995 <= y <= 2017]
print(f"  {len(cite_raw):,} citation rows for CPC {SECTOR['cpc']}, years {int(min(cite_overlap))}–{int(max(cite_overlap))}")

# ============================================================
# 3. Select top countries by combined trade + citation volume
# ============================================================
print("\n[3/4] Selecting top countries...")

def total_volume(df, wcol):
    s = df.groupby("source")[wcol].sum()
    t = df.groupby("target")[wcol].sum()
    return s.add(t, fill_value=0)

trade_vol = total_volume(trade_agg, "trade_value")
cite_vol = total_volume(cite_raw[cite_raw["year"].isin(cite_overlap)], "weight")
combined = trade_vol.add(cite_vol, fill_value=0)
top_individuals = combined.nlargest(INDIV_N).index.tolist()
all_countries = top_individuals + ["ROW"]

print(f"  Top {INDIV_N}: {', '.join(top_individuals)}")
print(f"  All (incl ROW): {all_countries}")

# ---- Color assignments ----
node_colors = {}
for i, n in enumerate(top_individuals):
    node_colors[n] = COLORS[i % len(COLORS)]
node_colors["ROW"] = "#999999"

# ============================================================
# 4. Build chord matrices
# ============================================================
print("\n[4/4] Building chord matrices...")

def build_matrices(df, wcol, years, individuals, all_groups):
    """Build non-interleaved export/import matrices for each year."""
    result = {}
    ind_set = set(individuals)
    N = len(all_groups)
    for yr in years:
        yr_df = df[df["year"] == yr]
        idx = {c: i for i, c in enumerate(all_groups)}

        # Matrix: rows 0..N-1 = exports, rows N..2N-1 = imports (zeros)
        mat = [[0.0] * (2 * N) for _ in range(2 * N)]

        for _, r in yr_df.iterrows():
            s, t = r["source"], r["target"]
            v = float(r[wcol])
            if v <= 0 or s == t:
                continue
            s_grp = s if s in ind_set else "ROW"
            t_grp = t if t in ind_set else "ROW"
            if s_grp == t_grp:
                continue
            i_src = idx[s_grp]          # row 0..N-1
            i_tgt = idx[t_grp] + N      # col N..2N-1
            mat[i_src][i_tgt] += v

        # Arc sizes: export row sums, import column sums
        exp_sizes = [float(sum(mat[i])) for i in range(N)]
        imp_sizes = [float(sum(mat[j][i] for j in range(N))) for i in range(N, 2 * N)]
        sizes = exp_sizes + imp_sizes

        # Interleaved for D3 display
        interleaved_sizes = []
        for i in range(N):
            interleaved_sizes.append(exp_sizes[i])
            interleaved_sizes.append(imp_sizes[i])

        result[int(yr)] = {
            "matrix": mat,
            "arc_sizes": sizes,
            "interleaved_sizes": interleaved_sizes,
        }
    return result

# All years for trade and overlapping years for citations
trade_yr_list = sorted(trade_agg["year"].unique())
cite_yr_list = [y for y in cite_years if 1995 <= y <= 2017]
all_yr_list = sorted(set(trade_yr_list) | set(cite_yr_list))

trade_data = build_matrices(trade_agg, "trade_value", trade_yr_list,
                            top_individuals, all_countries)
cite_data = build_matrices(cite_raw, "weight", cite_yr_list,
                           top_individuals, all_countries)

# ---- JSON encoder ----
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "sector": args.sector,
    "sector_name": SECTOR["name"],
    "trade_label": SECTOR["hs_label"],
    "cite_label": SECTOR["cpc_label"],
    "countries": all_countries,
    "country_names": {c: iso_to_name.get(c, c) if c != "ROW" else "Rest of World"
                      for c in all_countries},
    "colors": node_colors,
    "trade_years": [int(y) for y in trade_yr_list],
    "citation_years": [int(y) for y in cite_yr_list],
    "all_years": [int(y) for y in all_yr_list],
    "trade": {},
    "citation": {},
}

for yr in trade_yr_list:
    yr_key = str(int(yr))
    d = trade_data[int(yr)]
    output["trade"][yr_key] = {
        "matrix": d["matrix"],
        "arc_sizes": [float(v) for v in d["arc_sizes"]],
        "interleaved_sizes": [float(v) for v in d["interleaved_sizes"]],
    }

for yr in cite_yr_list:
    yr_key = str(int(yr))
    d = cite_data[int(yr)]
    output["citation"][yr_key] = {
        "matrix": d["matrix"],
        "arc_sizes": [float(v) for v in d["arc_sizes"]],
        "interleaved_sizes": [float(v) for v in d["interleaved_sizes"]],
    }

json_path = os.path.join(OUT_DIR, f"chord_data_level_b_{args.sector}.json")
json_str = json.dumps(output, cls=NpEncoder, indent=None)
with open(json_path, "w") as f:
    f.write(json_str)
print(f"  JSON saved → {json_path} ({len(json_str)/1024:.0f} KB)")

if args.json_only:
    print("Done (--json-only).")
    sys.exit(0)

# ============================================================
# 5. Generate standalone HTML
# ============================================================
print("\n[5/5] Generating standalone HTML...")

html_path = os.path.join(OUT_DIR, f"chord_level_b_{args.sector}_standalone.html")

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Level B Chord — SECTOR_NAME</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a2e;color:#eee;overflow:hidden;height:100vh}
#app{display:flex;height:100vh}
#left,#right{flex:1;position:relative;padding-bottom:60px}
#left h2{position:absolute;top:12px;left:18px;text-align:left;font-size:14px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none}
#right h2{position:absolute;top:12px;right:18px;text-align:right;font-size:14px;font-weight:600;letter-spacing:.5px;z-index:10;pointer-events:none}
#left h2{color:#4ECDC4}#right h2{color:#FF6B6B}
#sector-badge{position:fixed;top:8px;left:50%;transform:translateX(-50%);background:rgba(30,30,50,.9);border:1px solid rgba(255,255,255,.2);border-radius:8px;padding:4px 14px;font-size:11px;color:#aaa;z-index:100;letter-spacing:.5px}
svg{width:100%;height:calc(100% - 60px)}
#controls{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(30,30,50,.95);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:14px 24px;display:flex;gap:20px;align-items:center;z-index:100;backdrop-filter:blur(10px)}
#controls label{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:#aaa}
#year-slider{width:220px;accent-color:#4ECDC4}
#year-display{font-size:18px;font-weight:700;min-width:48px;text-align:center}
#play-btn{background:#4ECDC4;color:#1a1a2e;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-weight:700;font-size:13px}
#play-btn:hover{background:#6ee7dd}
#play-btn.paused{background:#FF6B6B}
select{background:#2a2a40;color:#eee;border:1px solid rgba(255,255,255,.2);border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
.tooltip{position:fixed;background:rgba(0,0,0,.85);color:#fff;padding:8px 12px;border-radius:6px;font-size:11px;pointer-events:none;z-index:200;opacity:0;transition:opacity .15s}
</style>
</head>
<body>
<div id="sector-badge">Level B: SECTOR_NAME — Industry × Country</div>
<div id="app">
  <div id="left"><h2>TRADE_LABEL<br><span style="font-weight:400;font-size:11px;color:#aaa">Trade Flows</span></h2></div>
  <div id="right"><h2>CITE_LABEL<br><span style="font-weight:400;font-size:11px;color:#aaa">Patent Citations</span></h2>
    <div id="cite-overlay" style="position:absolute;top:0;left:0;right:0;bottom:60px;display:none;align-items:center;justify-content:center;z-index:20;pointer-events:none;">
      <div style="background:rgba(0,0,0,0.72);border:1px solid rgba(255,107,107,0.5);border-radius:10px;padding:16px 28px;text-align:center;">
        <div style="font-size:15px;font-weight:700;color:#FF6B6B;">Citation data ends 2017</div>
        <div style="font-size:10px;color:#aaa;margin-top:4px;">~2013 last fully reliable year</div>
      </div>
    </div>
  </div>
</div>
<div id="controls">
  <label>Year</label><span id="year-display">2015</span>
  <input type="range" id="year-slider" min="1995" max="2024" value="2015" step="1">
  <button id="play-btn" onclick="togglePlay()">Play</button>
  <select id="highlight-select" onchange="setHighlight(this.value)"><option value="">All Countries</option></select>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const data = __JSON_DATA__;

(function() {
  const N = data.countries.length;
  const countries = data.countries;
  const colors = data.colors;

  const impColors = {};
  countries.forEach(c => {
    const h = colors[c];
    if (!h) { impColors[c] = "#888"; return; }
    const rr = parseInt(h[1]+h[2],16), gg = parseInt(h[3]+h[4],16), bb = parseInt(h[5]+h[6],16);
    impColors[c] = `rgb(${Math.round(rr*.5+127)},${Math.round(gg*.5+127)},${Math.round(bb*.5+127)})`;
  });

  const slider = document.getElementById("year-slider");
  slider.min = data.all_years[0]; slider.max = data.all_years[data.all_years.length-1]; slider.value = 2015;
  document.getElementById("year-display").textContent = 2015;

  const sel = document.getElementById("highlight-select");
  countries.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = data.country_names[c]||c; sel.appendChild(o); });

  const tradeSvg = d3.select("#left").append("svg");
  const citeSvg = d3.select("#right").append("svg");
  let highlight = "";

  function fmtVal(v, isTrade) {
    if (isTrade) {
      if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'B';
      if (v >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'M';
      return '$' + d3.format(',.0f')(v) + 'K';
    }
    return d3.format(',.0f')(v);
  }

  function render(svg, key, year) {
    const yrData = data[key][String(year)];
    if (!yrData) return;

    const el = svg.node().parentNode;
    const W = el.clientWidth, H = el.clientHeight;
    const cx = W/2, cy = H/2;
    const outerR = Math.min(W,H) * 0.40;
    const innerR = outerR * 0.90;

    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`);

    const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);
    g.append("circle").attr("r", innerR).attr("fill", "#fff");

    const matrix = yrData.matrix;
    const interleavedSizes = yrData.interleaved_sizes;

    const totalSize = d3.sum(interleavedSizes);
    const minSize = totalSize * 0.025;
    const adjSizes = interleavedSizes.map(s => Math.max(s, minSize));
    const adjTotal = d3.sum(adjSizes);

    const adjMatrix = matrix.map((row, i) => {
      if (i < N) {
        const rowSum = d3.sum(row);
        const adjSum = adjSizes[i];
        if (rowSum > 0 && adjSum > rowSum) {
          const scale = adjSum / rowSum;
          return row.map(v => v * scale);
        }
      }
      return [...row];
    });

    const layout2 = d3.chord().padAngle(0.002).sortGroups(null).sortSubgroups(d3.descending)(adjMatrix);
    const rawGroups = layout2.groups;

    const padAng = 0.002;
    const usableAngle = 2*Math.PI - (2*N)*padAng;
    let cursor = 0;
    const groups = [];
    for (let i = 0; i < N; i++) {
      const eSpan = (adjSizes[2*i]/adjTotal)*usableAngle;
      const iSpan = (adjSizes[2*i+1]/adjTotal)*usableAngle;
      groups.push({startAngle:cursor, endAngle:cursor+eSpan, _isExport:true, _country:i,
        value: interleavedSizes[2*i], _actualValue: interleavedSizes[2*i]});
      cursor += eSpan + padAng;
      groups.push({startAngle:cursor, endAngle:cursor+iSpan, _isExport:false, _country:i,
        value: interleavedSizes[2*i+1], _actualValue: interleavedSizes[2*i+1]});
      cursor += iSpan + padAng;
    }

    const chords = layout2
      .filter(d => d.source.index < N && d.target.index >= N)
      .map(d => {
        const si = d.source.index, ti = d.target.index - N;
        const newSi = 2*si, newTi = 2*ti+1;
        const sGrp = groups[newSi], tGrp = groups[newTi];
        if (!sGrp || !tGrp) return null;
        const sRaw = rawGroups[si];
        const sRawSpan = sRaw.endAngle - sRaw.startAngle || 1;
        const sf0 = (d.source.startAngle - sRaw.startAngle) / sRawSpan;
        const sf1 = (d.source.endAngle - sRaw.startAngle) / sRawSpan;
        const ss = sGrp.endAngle - sGrp.startAngle;
        const ts = tGrp.endAngle - tGrp.startAngle;
        const tCol = adjMatrix.map(r => r[ti+N]);
        const tSum = d3.sum(tCol) || 1;
        const tCum = d3.sum(tCol.slice(0, si));
        const val = d.source.value;
        const tf0 = tCum / tSum;
        const tf1 = (tCum + val) / tSum;
        const sa = sGrp.startAngle + sf0*ss, sb = sGrp.startAngle + sf1*ss;
        const ta = tGrp.startAngle + tf0*ts, tb = tGrp.startAngle + tf1*ts;
        if (isNaN(sa+sb+ta+tb)) return null;
        return {
          source: {index:newSi, startAngle:sa, endAngle:sb, value:val},
          target: {index:newTi, startAngle:ta, endAngle:tb, value:val}
        };
      }).filter(Boolean);

    // Country outlines — wrap each export+import pair with a colored border
    const outlineArcGen = d3.arc().innerRadius(innerR - 0.5).outerRadius(outerR + 4);
    const countryOutlines = [];
    for (let i = 0; i < N; i++) {
      const expGrp = groups[2*i];
      const impGrp = groups[2*i + 1];
      if (expGrp && impGrp && (expGrp.endAngle - expGrp.startAngle + impGrp.endAngle - impGrp.startAngle) > 0.001) {
        countryOutlines.push({
          startAngle: expGrp.startAngle,
          endAngle: impGrp.endAngle,
          _country: i
        });
      }
    }

    g.selectAll("path.country-outline").data(countryOutlines).join("path")
      .attr("class", "country-outline")
      .attr("d", outlineArcGen)
      .attr("fill", "none")
      .attr("stroke", d => colors[countries[d._country]])
      .attr("stroke-width", 2.5)
      .style("opacity", d => highlight && countries[d._country] !== highlight ? 0.12 : 0.85);

    const arcGen = d3.arc().innerRadius(innerR).outerRadius(outerR);

    g.selectAll("path.arc").data(groups).join("path")
      .attr("class", "arc")
      .attr("d", arcGen)
      .attr("fill", d => d._isExport ? colors[countries[d._country]] : impColors[countries[d._country]])
      .attr("stroke", "none")
      .style("opacity", d => highlight && countries[d._country]!==highlight ? 0.25 : 1)
      .on("mouseover", (ev,d) => {
        const c = countries[d._country];
        const side = key==="trade" ? (d._isExport?"Export":"Import") : (d._isExport?"Citing":"Cited");
        showTip(ev, `<b>${data.country_names[c]||c}</b> ${side}<br>${fmtVal(d._actualValue||d.value, key==='trade')}`);
      }).on("mousemove", moveTip).on("mouseout", hideTip);

    layout2.groups = groups;
    const ribbon = d3.ribbon().radius(innerR);

    g.selectAll("path.chord").data(chords).join("path")
      .attr("class", "chord")
      .attr("d", ribbon)
      .attr("fill", d => colors[countries[Math.floor(d.source.index/2)]])
      .attr("stroke", "none")
      .style("opacity", d => {
        if (!highlight) return 0.75;
        const s = countries[Math.floor(d.source.index/2)], t = countries[Math.floor(d.target.index/2)];
        return (s===highlight||t===highlight) ? 0.95 : 0.04;
      })
      .on("mouseover", (ev,d) => {
        const s = countries[Math.floor(d.source.index/2)], t = countries[Math.floor(d.target.index/2)];
        showTip(ev, `<b>${data.country_names[s]||s}</b> → <b>${data.country_names[t]||t}</b><br>${fmtVal(d.source.value, key==='trade')}`);
      }).on("mousemove", moveTip).on("mouseout", hideTip);

    // Labels — centered inside each country's export+import pair
    g.selectAll("text.label").data(countryOutlines).join("text")
      .attr("class", "label")
      .attr("transform", d => {
        const midAngle = (d.startAngle + d.endAngle) / 2 - Math.PI / 2;
        const r = (innerR + outerR) / 2;
        return `translate(${r*Math.cos(midAngle)},${r*Math.sin(midAngle)})`;
      })
      .attr("text-anchor", "middle")
      .attr("dy", ".35em")
      .style("font-size", d => {
        const span = d.endAngle - d.startAngle;
        return Math.max(8, Math.min(14, span * 42)) + "px";
      })
      .style("font-weight", "800")
      .style("fill", "#1a1a2e")
      .style("paint-order", "stroke fill")
      .style("stroke", "#fff")
      .style("stroke-width", "2.5px")
      .style("stroke-linejoin", "round")
      .style("opacity", d => highlight && countries[d._country] !== highlight ? 0.25 : 1)
      .text(d => countries[d._country]);

    g.append("text")
      .attr("x", 0).attr("y", outerR + 32)
      .attr("text-anchor", "middle")
      .style("fill", "#888").style("font-size", "9.5px")
      .text("Small arcs enlarged to 2.5% min for readability");

    let caveat;
    if (key === "trade") {
      caveat = "Self-flows excluded";
    } else {
      caveat = "Self-citations excluded · Country = filing office (proxy)";
      if (year >= 2014 && year <= 2017) {
        caveat += " · ⚠ Undercounted (2018 edition, ~2013 last reliable)";
      }
    }
    const caveatColor = (key === "citation" && year >= 2014 && year <= 2017) ? "#E9C46A" : "#777";
    g.append("text")
      .attr("x", 0).attr("y", outerR + 44)
      .attr("text-anchor", "middle")
      .style("fill", caveatColor).style("font-size", "9.5px").style("font-style", "italic")
      .text(caveat);
  }

  function showTip(ev, html) { d3.select("#tooltip").style("opacity",1).html(html); }
  function moveTip(ev) { d3.select("#tooltip").style("left",(ev.pageX+12)+"px").style("top",(ev.pageY-12)+"px"); }
  function hideTip() { d3.select("#tooltip").style("opacity",0); }

  window.setHighlight = function(c) { highlight = c; update(); };
  function update() {
    render(tradeSvg, "trade", +slider.value);
    render(citeSvg, "citation", +slider.value);
    const maxCiteYr = data.citation_years[data.citation_years.length - 1];
    const overlay = document.getElementById("cite-overlay");
    if (overlay) overlay.style.display = (+slider.value > maxCiteYr) ? "flex" : "none";
  }

  slider.addEventListener("input", () => { document.getElementById("year-display").textContent = slider.value; update(); });

  window._pi = null;
  window.togglePlay = function() {
    const btn = document.getElementById("play-btn");
    if (window._pi) { clearInterval(window._pi); window._pi = null; btn.textContent = "Play"; btn.classList.remove("paused"); return; }
    btn.textContent = "Pause"; btn.classList.add("paused");
    window._pi = setInterval(() => {
      let v = +slider.value + 1;
      if (v > +slider.max) v = +slider.min;
      slider.value = v;
      document.getElementById("year-display").textContent = v;
      update();
    }, 400);
  };

  update();
})();
</script>
</body>
</html>'''

# Replace placeholders
html = HTML.replace("SECTOR_NAME", SECTOR["name"])
html = html.replace("TRADE_LABEL", SECTOR["hs_label"])
html = html.replace("CITE_LABEL", SECTOR["cpc_label"])
html = html.replace("__JSON_DATA__", json_str)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

sz_kb = os.path.getsize(html_path) / 1024
print(f"  HTML saved → {html_path} ({sz_kb:.0f} KB)")

print(f"\nDone! Open the standalone HTML to explore Level B chord diagrams:")
print(f"  {html_path}")
print(f"\nKey Level B features:")
print(f"  - Trade flows filtered to {SECTOR['hs_label']}")
print(f"  - Citation flows filtered to {SECTOR['cpc_label']}")
print(f"  - Nodes are countries within the {SECTOR['name']} sector")
print(f"  - Same interaction model: year slider, Play, country highlight")
