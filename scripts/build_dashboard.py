"""
Build an interactive dashboard for exploring country-pair flows over time.
Features: year slider, time-series chart, directional flow cards,
+ country aggregation dropdown showing total flows to/from a single country.

Output: interactive_dashboard.html (standalone)
"""
import pandas as pd
import numpy as np
import json

OVERLAP_YEARS = list(range(1995, 2018))
NON_COUNTRIES = {"EP","WO","DD","OA","AP","GC","IB","EM","BX","YU","CS","SU"}
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))

print("Building combined trade + citation dataset...")
rows = []
for year in OVERLAP_YEARS:
    try:
        tdf = pd.read_csv(f"../processed/trade_edges_{year}.csv")
        tdf["source"] = tdf["exporter"].map(baci_to_iso)
        tdf["target"] = tdf["importer"].map(baci_to_iso)
        tdf["source"] = tdf["source"].fillna(tdf["exporter_name"].map(name_to_iso))
        tdf["target"] = tdf["target"].fillna(tdf["importer_name"].map(name_to_iso))
        tdf = tdf.dropna(subset=["source","target"])
        tdf = tdf[tdf["source"]!=tdf["target"]]
        for _,r in tdf.iterrows():
            rows.append({"year":year,"source":r["source"],"target":r["target"],
                         "trade_value":r["trade_value"],"citation_count":0})
    except FileNotFoundError: pass

    cdf = pd.read_csv("../processed/country_citation_edges_by_year.csv")
    cdf = cdf[(cdf["year"]==year)&(~cdf["citing_country"].isin(NON_COUNTRIES))
              &(~cdf["cited_country"].isin(NON_COUNTRIES))]
    cdf = cdf[cdf["citing_country"]!=cdf["cited_country"]]
    for _,r in cdf.iterrows():
        rows.append({"year":year,"source":r["citing_country"],"target":r["cited_country"],
                     "trade_value":0,"citation_count":r["citation_count"]})

all_data = pd.DataFrame(rows).groupby(["year","source","target"],as_index=False).sum()
all_countries = sorted(set(all_data["source"].unique())|set(all_data["target"].unique()))
country_opts = [{"label":f"{c} — {iso_to_name.get(c,c)}","value":c}
                for c in all_countries if c in iso_to_name]
print(f"  {len(all_data):,} rows, {len(country_opts)} countries")

# ---- Country-pair time series ----
ts_data = {}
for (src,tgt),grp in all_data.groupby(["source","target"]):
    ts_data[f"{src}->{tgt}"] = {
        "years":grp["year"].tolist(),
        "trade":grp["trade_value"].tolist(),
        "citations":grp["citation_count"].tolist(),
    }

# ---- Country-aggregated time series (exports, imports, citing, cited) ----
country_agg = {}
for c in all_countries:
    exports = all_data[all_data["source"]==c].groupby("year").agg(
        trade_out=("trade_value","sum"), cite_out=("citation_count","sum")).reset_index()
    imports = all_data[all_data["target"]==c].groupby("year").agg(
        trade_in=("trade_value","sum"), cite_in=("citation_count","sum")).reset_index()
    agg = exports.merge(imports, on="year", how="outer").fillna(0)
    agg = agg.sort_values("year")
    country_agg[c] = {
        "years": agg["year"].tolist(),
        "trade_out": agg["trade_out"].tolist(),
        "trade_in": agg["trade_in"].tolist(),
        "cite_out": agg["cite_out"].tolist(),
        "cite_in": agg["cite_in"].tolist(),
    }

# ---- HTML ----
html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Trade & Citation Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f7fa;color:#2c3e50}
.container{max-width:1200px;margin:0 auto;padding:20px}
h1{text-align:center;margin-bottom:4px;font-size:22px}
.subtitle{text-align:center;color:#7f8c8d;margin-bottom:20px;font-size:13px}
.controls{display:flex;gap:12px;justify-content:center;align-items:center;flex-wrap:wrap;margin-bottom:20px}
.controls select,.controls input{padding:8px 12px;border:2px solid #dcdde1;border-radius:6px;font-size:14px}
.controls label{font-weight:600;font-size:13px}
#mode-cp,#mode-agg{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.flow-panels{display:flex;gap:20px;margin-bottom:20px}
.flow-card{flex:1;background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center}
.flow-card.trade{border-left:4px solid #3A7CA5}
.flow-card.cite{border-left:4px solid #C87A4A}
.flow-card h3{margin-bottom:8px;font-size:15px}
.flow-arrow{font-size:32px;margin:8px 0}
.flow-value{font-size:26px;font-weight:700}
.flow-unit{font-size:13px;color:#7f8c8d}
#ts-chart{background:white;border-radius:12px;padding:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);height:420px}
.info{background:white;border-radius:12px;padding:15px;margin-top:15px;box-shadow:0 2px 8px rgba(0,0,0,0.08);font-size:12px}
</style>
</head>
<body>
<div class="container">
<h1>Trade &amp; Citation Flow Dashboard</h1>
<p class="subtitle">Country-pair explorer or single-country aggregation &mdash; compare goods trade (blue) with knowledge flows (orange)</p>
<div class="controls">
  <label>Mode:</label>
  <select id="mode-sel"><option value="pair">Country Pair</option><option value="country">Single Country</option></select>
  <span id="mode-cp">
    <label>Source:</label><select id="src-sel"></select>
    <span style="font-size:18px">&rarr;</span>
    <label>Target:</label><select id="tgt-sel"></select>
  </span>
  <span id="mode-agg" style="display:none">
    <label>Country:</label><select id="ctry-sel"></select>
  </span>
  <label>Year:</label>
  <input type="range" id="year-slider" min="1995" max="2017" value="2015" step="1" style="width:200px">
  <span id="year-label" style="font-weight:700;font-size:18px;min-width:50px">2015</span>
  <button onclick="toggleAnim()" style="padding:8px 16px;border:none;border-radius:6px;background:#27ae60;color:white;cursor:pointer;font-weight:600" id="anim-btn">&blacktriangleright; Play</button>
</div>
<div class="flow-panels">
  <div class="flow-card trade"><h3 id="trade-title">Trade Flow</h3><div class="flow-arrow" id="trade-dir">&mdash;</div><div class="flow-value" id="trade-val">&mdash;</div><div class="flow-unit">trade value (USD)</div></div>
  <div class="flow-card cite"><h3 id="cite-title">Citation Flow</h3><div class="flow-arrow" id="cite-dir">&mdash;</div><div class="flow-value" id="cite-val">&mdash;</div><div class="flow-unit">patent citations</div></div>
</div>
<div id="ts-chart"></div>
<div class="info"><strong>Trade arrow &rarr; importer (buyer)</strong> | <strong>Citation arrow &rarr; cited country (knowledge source)</strong> | Blue = trade | Orange = citations | Use this dashboard to explore lead/lag patterns over time</div>
</div>
"""

# Embed data
html += f"\n<script>\nconst countryOpts = {json.dumps(country_opts)};\n"
html += f"const tsData = {json.dumps(ts_data)};\n"
html += f"const countryAgg = {json.dumps(country_agg)};\n"

html += """
const srcEl=document.getElementById('src-sel'),tgtEl=document.getElementById('tgt-sel');
const ctryEl=document.getElementById('ctry-sel');
const modeEl=document.getElementById('mode-sel');
const slider=document.getElementById('year-slider'),yrLbl=document.getElementById('year-label');
let currentYear=2015,animTimer=null;

countryOpts.forEach(c=>{
  srcEl.innerHTML+=`<option value="${c.value}">${c.label}</option>`;
  tgtEl.innerHTML+=`<option value="${c.value}">${c.label}</option>`;
  ctryEl.innerHTML+=`<option value="${c.value}">${c.label}</option>`;
});
srcEl.value='CN';tgtEl.value='US';ctryEl.value='CN';

function fmt(v,t){
  if(t==='trade'){
    if(v>=1e12)return'$'+(v/1e12).toFixed(2)+'T';
    if(v>=1e9)return'$'+(v/1e9).toFixed(1)+'B';
    return'$'+(v/1e6).toFixed(1)+'M';
  }else{
    if(v>=1e6)return(v/1e6).toFixed(1)+'M';
    if(v>=1e3)return(v/1e3).toFixed(1)+'K';
    return v.toLocaleString();
  }
}

function update(){
  const mode=modeEl.value;
  document.getElementById('mode-cp').style.display=mode==='pair'?'flex':'none';
  document.getElementById('mode-agg').style.display=mode==='country'?'flex':'none';

  const y=parseInt(slider.value);
  currentYear=y;yrLbl.textContent=y;

  if(mode==='pair'){
    updatePair(y);
  }else{
    updateCountry(y);
  }
}

function updatePair(y){
  const src=srcEl.value,tgt=tgtEl.value;
  const fk=src+'->'+tgt,rk=tgt+'->'+src;
  const fwd=tsData[fk],rev=tsData[rk];
  let tv=0,tsrc=src,ttgt=tgt,cv=0,csrc=src,ctgt=tgt;

  if(fwd){const i=fwd.years.indexOf(y);if(i>=0&&fwd.trade[i]>0){tv=fwd.trade[i];}}
  if(rev){const i=rev.years.indexOf(y);if(i>=0&&rev.trade[i]>tv){tv=rev.trade[i];tsrc=tgt;ttgt=src;}}
  document.getElementById('trade-title').textContent='Trade Flow';
  document.getElementById('trade-dir').innerHTML=tv>0?tsrc+' <span style="font-size:24px">&rarr;</span> '+ttgt:'&mdash;';
  document.getElementById('trade-val').textContent=tv>0?fmt(tv,'trade'):'&mdash;';
  document.getElementById('trade-val').style.color=tv>0?'#3A7CA5':'#999';

  if(fwd){const i=fwd.years.indexOf(y);if(i>=0&&fwd.citations[i]>0){cv=fwd.citations[i];}}
  if(rev){const i=rev.years.indexOf(y);if(i>=0&&rev.citations[i]>cv){cv=rev.citations[i];csrc=tgt;ctgt=src;}}
  document.getElementById('cite-title').textContent='Citation Flow';
  document.getElementById('cite-dir').innerHTML=cv>0?csrc+' <span style="font-size:24px">&rarr;</span> '+ctgt:'&mdash;';
  document.getElementById('cite-val').textContent=cv>0?fmt(cv,'cite'):'&mdash;';
  document.getElementById('cite-val').style.color=cv>0?'#C87A4A':'#999';

  const ya=[],tva=[],cva=[];
  for(let yr=1995;yr<=2017;yr++){
    ya.push(yr);let tv2=0,cv2=0;
    if(fwd){const i=fwd.years.indexOf(yr);if(i>=0){tv2=fwd.trade[i];cv2=fwd.citations[i];}}
    if(rev){const i=rev.years.indexOf(yr);if(i>=0){tv2=Math.max(tv2,rev.trade[i]);cv2=Math.max(cv2,rev.citations[i]);}}
    tva.push(tv2);cva.push(cv2);
  }
  Plotly.react('ts-chart',[
    {x:ya,y:tva,type:'scatter',mode:'lines+markers',name:'Trade (USD)',line:{color:'#3A7CA5',width:2.5},marker:{size:5,color:'#3A7CA5'},yaxis:'y'},
    {x:ya,y:cva,type:'scatter',mode:'lines+markers',name:'Citations',line:{color:'#C87A4A',width:2.5},marker:{size:5,color:'#C87A4A'},yaxis:'y2'}
  ],{
    title:src+' &harr; '+tgt+': Trade & Citation Flows (1995&ndash;2017)',
    xaxis:{title:'Year',dtick:2,gridcolor:'#eee'},
    yaxis:{title:'Trade Value (USD)',color:'#3A7CA5',gridcolor:'#eee',rangemode:'tozero',type:'log'},
    yaxis2:{title:'Citations',color:'#C87A4A',overlaying:'y',side:'right',rangemode:'tozero',type:'log'},
    legend:{x:0.01,y:0.99},hovermode:'x unified',margin:{t:40,r:60,l:60,b:40},
    shapes:[{type:'line',x0:currentYear,x1:currentYear,y0:0,y1:1,yref:'paper',line:{color:'#333',width:1,dash:'dot'}}]
  },{responsive:true});
}

function updateCountry(y){
  const c=ctryEl.value;
  const agg=countryAgg[c];
  if(!agg) return;
  const i=agg.years.indexOf(y);

  // Trade out / in
  const tradeOut=i>=0?agg.trade_out[i]:0;
  const tradeIn =i>=0?agg.trade_in[i]:0;
  document.getElementById('trade-title').textContent=c+' Trade Aggregated';
  document.getElementById('trade-dir').innerHTML=
    '<span style="color:#3A7CA5">Exports</span> '+fmt(tradeOut,'trade')+' &nbsp;|&nbsp; '+
    fmt(tradeIn,'trade')+' <span style="color:#2980b9">Imports</span>';
  document.getElementById('trade-val').textContent='Balance: '+fmt(tradeOut-tradeIn,'trade');
  document.getElementById('trade-val').style.color=tradeOut>=tradeIn?'#27ae60':'#e74c3c';

  // Cite out / in
  const citeOut=i>=0?agg.cite_out[i]:0;
  const citeIn =i>=0?agg.cite_in[i]:0;
  document.getElementById('cite-title').textContent=c+' Citation Aggregated';
  document.getElementById('cite-dir').innerHTML=
    '<span style="color:#C87A4A">Citing</span> '+fmt(citeOut,'cite')+' &nbsp;|&nbsp; '+
    fmt(citeIn,'cite')+' <span style="color:#e67e22">Cited</span>';
  document.getElementById('cite-val').textContent='Net: '+(citeOut>citeIn?'+':'')+fmt(citeOut-citeIn,'cite');
  document.getElementById('cite-val').style.color=citeOut>=citeIn?'#27ae60':'#e74c3c';

  // Time series: 4 lines
  const traces=[
    {x:agg.years,y:agg.trade_out,type:'scatter',mode:'lines',name:'Exports',line:{color:'#3A7CA5',width:2.5},yaxis:'y'},
    {x:agg.years,y:agg.trade_in,type:'scatter',mode:'lines',name:'Imports',line:{color:'#2980b9',width:2,dash:'dash'},yaxis:'y'},
    {x:agg.years,y:agg.cite_out,type:'scatter',mode:'lines',name:'Citing',line:{color:'#C87A4A',width:2.5},yaxis:'y2'},
    {x:agg.years,y:agg.cite_in,type:'scatter',mode:'lines',name:'Cited',line:{color:'#e67e22',width:2,dash:'dash'},yaxis:'y2'}
  ];
  Plotly.react('ts-chart',traces,{
    title:c+': Aggregated Trade & Citation Flows (1995&ndash;2017)',
    xaxis:{title:'Year',dtick:2,gridcolor:'#eee'},
    yaxis:{title:'Trade Value (USD)',color:'#3A7CA5',gridcolor:'#eee',rangemode:'tozero',type:'log'},
    yaxis2:{title:'Citations',color:'#C87A4A',overlaying:'y',side:'right',rangemode:'tozero',type:'log'},
    legend:{x:0.01,y:0.99},hovermode:'x unified',margin:{t:40,r:60,l:60,b:40},
    shapes:[{type:'line',x0:currentYear,x1:currentYear,y0:0,y1:1,yref:'paper',line:{color:'#333',width:1,dash:'dot'}}]
  },{responsive:true});
}

function toggleAnim(){
  if(animTimer){clearInterval(animTimer);animTimer=null;document.getElementById('anim-btn').innerHTML='&#9654; Play';return;}
  slider.value=1995;update();
  animTimer=setInterval(()=>{
    let y=parseInt(slider.value)+1;
    if(y>2017){clearInterval(animTimer);animTimer=null;document.getElementById('anim-btn').innerHTML='&#9654; Play';return;}
    slider.value=y;update();
  },250);
  document.getElementById('anim-btn').innerHTML='&#9208; Pause';
}

slider.addEventListener('input',update);
srcEl.addEventListener('change',update);tgtEl.addEventListener('change',update);
ctryEl.addEventListener('change',update);modeEl.addEventListener('change',update);
update();
</script></body></html>"""

with open("../html/interactive_dashboard.html","w",encoding="utf-8") as f:
    f.write(html)
print(f"Saved -> ../html/interactive_dashboard.html ({len(html):,} bytes)")
