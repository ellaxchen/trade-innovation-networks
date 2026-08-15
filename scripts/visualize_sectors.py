"""
Side-by-side sector comparison: trade (HS2) vs. citation (CPC) networks.
4 rows: Electronics, Vehicles, Pharma, Machinery.
Usage: python visualize_sectors.py [--year 2015]
"""
import pandas as pd; import numpy as np; import matplotlib.pyplot as plt
import matplotlib.patches as mpatches; from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap; import networkx as nx
import argparse; from math import pi, sin, cos, atan2

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2015)
parser.add_argument("--top-n", type=int, default=10)
parser.add_argument("--max-edges", type=int, default=40)
args = parser.parse_args()
YEAR, TOP_N, MAX_EDGES = args.year, getattr(args, "top_n", 10), getattr(args, "max_edges", 40)
NON_COUNTRIES = {"EP","WO","DD","OA","AP","GC","IB","EM","BX","YU","CS","SU"}

SECTOR_PAIRS = [
    ("85", "Electronics (HS85)", "H", "Electricity (CPC H)"),
    ("87", "Vehicles (HS87)", "B", "Ops & Transport (CPC B)"),
    ("30", "Pharma (HS30)", "C", "Chemistry (CPC C)"),
    ("84", "Machinery (HS84)", "F", "Engineering (CPC F)"),
]

TRADE_CMAP = LinearSegmentedColormap.from_list("t",["#B8D8F0","#9AC8E6","#7DB9DC","#5BA3D9","#3D8BC0","#1E6BA8","#0B2D5C"])
CITE_CMAP  = LinearSegmentedColormap.from_list("c",["#F5D5B8","#EDC09E","#E0A87A","#D4915A","#C87A4A","#A85A2E","#7A3512"])

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))

print(f"Loading CPC citation data for {YEAR}...")
cite_all = pd.read_csv("../processed/country_citation_by_cpc.csv")
cite_all = cite_all[(cite_all["year"]==YEAR) & (~cite_all["citing_country"].isin(NON_COUNTRIES))
                    & (~cite_all["cited_country"].isin(NON_COUNTRIES))]
cite_all = cite_all[cite_all["citing_country"]!=cite_all["cited_country"]]
cite_all = cite_all.rename(columns={"citing_country":"source","cited_country":"target"})

# ---- helpers ----
def build_graph(df, wcol, nodes, me):
    g=nx.DiGraph(); g.add_nodes_from(nodes)
    s=df[df["source"].isin(nodes)&df["target"].isin(nodes)].nlargest(me,wcol)
    for _,r in s.iterrows(): g.add_edge(r["source"],r["target"],weight=r[wcol])
    return g

def top_nodes(df, wcol, n):
    a=df[["source",wcol]].rename(columns={"source":"c",wcol:"w"})
    b=df[["target",wcol]].rename(columns={"target":"c",wcol:"w"})
    return pd.concat([a,b]).groupby("c")["w"].sum().nlargest(n).index.tolist()

def node_sizes(G, nodes):
    t={n:0 for n in nodes}
    for u,v,d in G.edges(data=True): w=d["weight"]; t[u]+=w; t[v]+=w
    sv=np.sqrt([t[n] for n in nodes]); lo,hi=sv.min(),sv.max()
    if hi==lo: return {n:400 for n in nodes}
    return {n:80+1500*(s-lo)/(hi-lo) for n,s in zip(nodes,sv)}

def net_pos(df, wcol, nodes):
    s=df[df["source"].isin(nodes)].groupby("source")[wcol].sum()
    r=df[df["target"].isin(nodes)].groupby("target")[wcol].sum()
    return {n:((s.get(n,0)-r.get(n,0))/(s.get(n,0)+r.get(n,0))) if (s.get(n,0)+r.get(n,0))>0 else 0 for n in nodes}

def edge_widths(G):
    ws=[d["weight"] for _,_,d in G.edges(data=True)]
    if not ws: return {}
    lo,hi=min(ws),max(ws)
    if hi==lo: return {(u,v):1.5 for u,v in G.edges()}
    return {(u,v):0.3+3.5*(d["weight"]-lo)/(hi-lo) for u,v,d in G.edges(data=True)}

def draw_panel(ax, G, pos, sizes, net, edge_w, edge_color, cmap, title):
    mw=max((d["weight"] for _,_,d in G.edges(data=True)), default=1)
    drawn=set()
    for u,v,d in sorted(G.edges(data=True), key=lambda e:e[2].get("weight",0)):
        if u==v or (v,u) in drawn: continue
        x1,y1,x2,y2=pos[u][0],pos[u][1],pos[v][0],pos[v][1]
        lw=edge_w.get((u,v),0.5); w=d.get("weight",1); alpha=0.30+0.65*(w/mw)
        has_rev=G.has_edge(v,u); bulge=0.22 if has_rev else 0.06
        mx,my=(x1+x2)/2,(y1+y2)/2; dx,dy=x2-x1,y2-y1; L=np.hypot(dx,dy)
        if L<0.01: continue
        ndx,ndy=-dy/L,dx/L; cx,cy=mx+ndx*bulge*L, my+ndy*bulge*L
        pd=[(Path.MOVETO,(x1,y1)),(Path.CURVE3,(cx,cy)),(Path.CURVE3,(x2,y2))]
        codes,verts=zip(*pd)
        ax.add_patch(mpatches.PathPatch(Path(verts,codes),fc="none",ec=edge_color,
                     lw=lw,alpha=min(alpha,0.85),zorder=2,capstyle="round"))
        tx,ty=2*(x2-cx),2*(y2-cy); tn=np.hypot(tx,ty)
        if tn>0.001: tx,ty=tx/tn,ty/tn
        nr=np.sqrt(sizes.get(v,400))*0.0012+0.015; arr_s=0.006+0.003*lw
        tix,tiy=x2-tx*nr,y2-ty*nr; px,py=-ty,tx
        bx,by=tix-tx*arr_s*2,tiy-ty*arr_s*2
        ax.add_patch(mpatches.Polygon([(tix,tiy),(bx+px*arr_s,by+py*arr_s),
                     (bx-px*arr_s,by-py*arr_s)],closed=True,fc=edge_color,alpha=alpha,zorder=6))
        if has_rev:
            drawn.add((u,v)); rw=G[v][u].get("weight",1)
            rlw=edge_w.get((v,u),0.5); ra=0.30+0.65*(rw/mw)
            rcx,rcy=mx+ndx*(-0.22)*L, my+ndy*(-0.22)*L
            rpd=[(Path.MOVETO,(x2,y2)),(Path.CURVE3,(rcx,rcy)),(Path.CURVE3,(x1,y1))]
            rc,rv=zip(*rpd)
            ax.add_patch(mpatches.PathPatch(Path(rv,rc),fc="none",ec=edge_color,
                         lw=rlw,alpha=min(ra,0.85),zorder=2,capstyle="round"))
            rtx,rty=2*(x1-rcx),2*(y1-rcy); rtn=np.hypot(rtx,rty)
            if rtn>0.001: rtx,rty=rtx/rtn,rty/rtn
            rnr=np.sqrt(sizes.get(u,400))*0.0012+0.015; rs=0.006+0.003*rlw
            rt2x,rt2y=x1-rtx*rnr,y1-rty*rnr; rpx,rpy=-rty,rtx
            rb2x,rb2y=rt2x-rtx*rs*2,rt2y-rty*rs*2
            ax.add_patch(mpatches.Polygon([(rt2x,rt2y),(rb2x+rpx*rs,rb2y+rpy*rs),
                         (rb2x-rpx*rs,rb2y-rpy*rs)],closed=True,fc=edge_color,alpha=ra,zorder=6))
    for node in G.nodes():
        x,y=pos[node]; s=sizes.get(node,400); nv=net.get(node,0)
        fc=cmap((max(-1,min(1,nv))+1)/2)
        ax.scatter(x,y,s=np.sqrt(s)*6,c=[fc],alpha=0.92,ec="#555555",lw=0.5,zorder=5)
        angle=atan2(y,x); r=1.14; lx,ly=r*cos(angle),r*sin(angle)
        ha="center"
        if lx>0.03: ha="left"
        elif lx<-0.03: ha="right"
        ax.annotate(node,(lx,ly),fontsize=6,fontweight="bold",ha=ha,va="center",annotation_clip=False)
    ax.set_title(title,fontsize=8.5,fontweight="bold")
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); ax.axis("off")

# ---- global positions ----
all_cty=set()
for hs2,_,cpc,_ in SECTOR_PAIRS:
    try:
        td=pd.read_csv(f"../processed/trade_edges_{YEAR}_hs{hs2}.csv")
        td["source"]=td["exporter"].map(baci_to_iso); td["target"]=td["importer"].map(baci_to_iso)
        td["source"]=td["source"].fillna(td["exporter_name"].map(name_to_iso))
        td["target"]=td["target"].fillna(td["importer_name"].map(name_to_iso))
        td=td.dropna(subset=["source","target"])
        all_cty.update(top_nodes(td,"trade_value",TOP_N))
    except: pass
    cd=cite_all[cite_all["cpc_section"]==cpc]
    if len(cd): all_cty.update(top_nodes(cd,"citation_count",TOP_N))
all_cty=sorted(all_cty); n=len(all_cty)
gpos={c:(cos(2*pi*i/n-pi/2),sin(2*pi*i/n-pi/2)) for i,c in enumerate(all_cty)}
print(f"{len(all_cty)} countries")

# ---- plot ----
fig,axes=plt.subplots(4,2,figsize=(14,28))
for row,(hs2,tlabel,cpc,clabel) in enumerate(SECTOR_PAIRS):
    ax_t,ax_c=axes[row][0],axes[row][1]
    try:
        td=pd.read_csv(f"../processed/trade_edges_{YEAR}_hs{hs2}.csv")
        td["source"]=td["exporter"].map(baci_to_iso); td["target"]=td["importer"].map(baci_to_iso)
        td["source"]=td["source"].fillna(td["exporter_name"].map(name_to_iso))
        td["target"]=td["target"].fillna(td["importer_name"].map(name_to_iso))
        td=td.dropna(subset=["source","target"]); td=td[td["source"]!=td["target"]]
        tn=top_nodes(td,"trade_value",TOP_N); Gt=build_graph(td,"trade_value",tn,MAX_EDGES)
        tv=td["trade_value"].sum()/1e6
        draw_panel(ax_t,Gt,gpos,node_sizes(Gt,tn),net_pos(td,"trade_value",tn),
                   edge_widths(Gt),"#3A7CA5",TRADE_CMAP,
                   f"{tlabel}\n{Gt.number_of_edges()} edges, ${tv:,.0f}M total")
    except Exception as e:
        ax_t.set_title(f"{tlabel}\n(error: {e})",fontsize=8); ax_t.axis("off")
    cd=cite_all[cite_all["cpc_section"]==cpc]
    if len(cd):
        cn=top_nodes(cd,"citation_count",TOP_N); Gc=build_graph(cd,"citation_count",cn,MAX_EDGES)
        cv=cd["citation_count"].sum()
        draw_panel(ax_c,Gc,gpos,node_sizes(Gc,cn),net_pos(cd,"citation_count",cn),
                   edge_widths(Gc),"#C87A4A",CITE_CMAP,
                   f"{clabel}\n{Gc.number_of_edges()} edges, {cv:,.0f} citations")
    else: ax_c.set_title(f"{clabel}\n(no data)",fontsize=8); ax_c.axis("off")

fig.suptitle(f"Trade vs. Knowledge Flows by Sector — {YEAR} (cross-border only)\n"
             "Trade: arrow → importer  |  Citation: arrow → cited country",
             fontsize=12,fontweight="bold",y=0.995)
out_path=f"../plots/sector_comparison_{YEAR}.png"
plt.tight_layout(rect=[0,0,1,0.99])
plt.savefig(out_path,dpi=150,bbox_inches="tight",facecolor="white")
print(f"Saved -> {out_path}")
