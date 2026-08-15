"""
Chord diagram matching migration-chord.png reference style.
Each country = one ring segment split into export (solid) / import (faded) halves,
separated by a visible gap. Chords fill full ring height at both ends, simple curves.

Usage: python chord_diagram.py [--year 2015] [--top-n 4]
"""
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
from matplotlib.path import Path; import argparse; from math import pi, sin, cos

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2015)
parser.add_argument("--top-n", type=int, default=4)
args = parser.parse_args()
YEAR, TOP_N = args.year, getattr(args, "top_n", 4)

# ---- Load ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"])); baci_to_iso[490]="TW"
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"])); name_to_iso["Other Asia, nes"]="TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"])); iso_to_name["TW"]="Taiwan"

full_trade = pd.read_parquet("../rawdata/baci_panel_hs6.parquet",
                             columns=["year","exporter","importer","trade_value"])
full_trade = full_trade[full_trade["year"]==YEAR]
full_trade["source"]=full_trade["exporter"].map(baci_to_iso)
full_trade["target"]=full_trade["importer"].map(baci_to_iso)
full_trade = full_trade.dropna(subset=["source","target"])
full_trade = full_trade[full_trade["source"]!=full_trade["target"]]
full_trade = full_trade.groupby(["source","target"],as_index=False)["trade_value"].sum()

cite = pd.read_csv("../processed/country_citation_edges_by_year.csv")
cite = cite.rename(columns={"citing_country":"source","cited_country":"target","citation_count":"weight"})
NON = {"EP","WO","DD","OA","AP","GC","IB","EM","BX","YU","CS","SU"}
cite = cite[(cite["year"]==YEAR)&(~cite["source"].isin(NON))&(~cite["target"].isin(NON))]
cite = cite[cite["source"]!=cite["target"]]

# ---- Top N ----
def tot_flow(df, wcol):
    s=df.groupby("source")[wcol].sum(); r=df.groupby("target")[wcol].sum()
    return s.add(r,fill_value=0)
all_vol = tot_flow(full_trade,"trade_value").add(tot_flow(cite,"weight"),fill_value=0)
top = all_vol.nlargest(TOP_N).index.tolist()
print(f"Top {TOP_N}: {top}")

# Colors
COLORS = ["#E63946","#3A7CA5","#2A9D8F","#F4A261","#9B5DE5"]
node_colors = {n:COLORS[i] for i,n in enumerate(top)}

# Anchor positions: CN at 0, US at pi/2, DE at pi, JP at 3pi/2
anchors = {}
for i, n in enumerate(["CN","US","DE","JP"]):
    if n in top: anchors[n] = 2*pi*i/4
remaining = [n for n in top if n not in anchors]
for i, n in enumerate(remaining):
    anchors[n] = 2*pi*(i+0.5)/len(remaining) + pi/6
segment_order = sorted(top, key=lambda n: anchors.get(n,0))

def draw_chord(ax, df, wcol, nodes, title, single_chord=None):
    """Chord diagram matching migration-chord reference style."""
    exports = df.groupby("source")[wcol].sum()
    imports = df.groupby("target")[wcol].sum()
    totals = {n: exports.get(n,0)+imports.get(n,0) for n in nodes}
    grand = sum(totals.values())
    if grand==0: return

    GAP_DEG = 2.0        # gap between export/import halves (degrees)
    SEG_GAP_DEG = 1.5    # gap between adjacent countries (degrees)
    R_out, R_in = 1.0, 0.88   # ring outer/inner radius

    total_gap = (len(nodes) * 2 * GAP_DEG + len(nodes) * SEG_GAP_DEG) * pi/180
    usable_circle = 2*pi - total_gap

    # ---- Segment positions with gaps ----
    sorted_nodes = sorted(nodes, key=lambda n: anchors.get(n, 0))
    seg = {}
    angle = 0
    for idx, node in enumerate(sorted_nodes):
        frac = totals[node]/grand
        span = frac * usable_circle
        if idx > 0:
            angle += SEG_GAP_DEG * pi/180
        seg[node] = (angle, angle + span)
        angle += span

    # Export/import split
    exp_frac = {n: exports.get(n,0)/totals[n] if totals[n]>0 else 0.5 for n in nodes}

    # Export/import arc boundaries (accounting for gap)
    exp_arc = {}
    imp_arc = {}
    for node in nodes:
        a0, a1 = seg[node]; span = a1-a0
        ef = exp_frac[node]
        exp_span = max(0, (span - GAP_DEG*pi/180) * ef)
        exp_arc[node] = (a0, a0 + exp_span)
        gap_start = a0 + exp_span + GAP_DEG*pi/180
        imp_arc[node] = (gap_start, a1)

    # ---- Draw segments ----
    for node in nodes:
        a0, a1 = seg[node]; span = a1-a0
        ea0, ea1 = exp_arc[node]
        ia0, ia1 = imp_arc[node]

        # Export (solid, darker)
        ax.add_patch(mpatches.Wedge((0,0), R_out, ea0*180/pi, ea1*180/pi,
                     width=R_out-R_in, fc=node_colors[node], alpha=0.85,
                     ec="none", zorder=4))
        # Import (faded, lighter)
        ax.add_patch(mpatches.Wedge((0,0), R_out, ia0*180/pi, ia1*180/pi,
                     width=R_out-R_in, fc=node_colors[node], alpha=0.30,
                     ec="none", zorder=4))
        # Tick marks at export/import boundaries
        tick_r1 = R_in - 0.02; tick_r2 = R_out + 0.02
        for a in [ea1, ia0]:
            ax.plot([tick_r1*cos(a), tick_r2*cos(a)],
                    [tick_r1*sin(a), tick_r2*sin(a)],
                    color="white", lw=0.8, zorder=6)

        # Label at center of full segment
        amid = (a0+a1)/2; lr = R_out + 0.07
        ax.annotate(node, (lr*cos(amid), lr*sin(amid)), fontsize=9, fontweight="bold",
                    ha="center", va="center", annotation_clip=False, zorder=10)

    # ---- Build chord list ----
    if single_chord:
        src_only, tgt_only = single_chord
        chord_list = []
        for _, r in df.iterrows():
            s, t, v = r["source"], r["target"], r[wcol]
            if s==src_only and t==tgt_only and s in nodes and t in nodes and v>0:
                chord_list.append({"source":s, "target":t, "value":v})
    else:
        chord_list = []
        for _, r in df.iterrows():
            s, t, v = r["source"], r["target"], r[wcol]
            if s in nodes and t in nodes and s!=t and v>0:
                chord_list.append({"source":s, "target":t, "value":v})
    chord_list.sort(key=lambda x: x["value"], reverse=True)

    if not chord_list:
        print(f"  No chords to draw for {title}")
        return

    # ---- Position chords within export/import arcs ----
    for src in nodes:
        src_chords = sorted([c for c in chord_list if c["source"]==src],
                           key=lambda x: x["value"], reverse=True)
        if not src_chords: continue
        src_total = sum(c["value"] for c in src_chords)
        all_exports = exports.get(src, 0)
        show_frac = src_total/all_exports if all_exports>0 else 1
        exp_a0, exp_a1 = exp_arc[src]
        exp_len = exp_a1 - exp_a0
        cursor = exp_a0
        for c in src_chords:
            frac = c["value"]/src_total if src_total>0 else 0
            w = frac * exp_len * show_frac
            c["s_start"] = cursor
            c["s_end"] = cursor + w
            c["s_mid"] = (c["s_start"] + c["s_end"]) / 2
            c["s_width"] = w
            cursor += w

    for tgt in nodes:
        tgt_chords = sorted([c for c in chord_list if c["target"]==tgt],
                           key=lambda x: x["value"], reverse=True)
        if not tgt_chords: continue
        tgt_total = sum(c["value"] for c in tgt_chords)
        all_imports = imports.get(tgt, 0)
        show_frac = tgt_total/all_imports if all_imports>0 else 1
        imp_a0, imp_a1 = imp_arc[tgt]
        imp_len = imp_a1 - imp_a0
        cursor = imp_a0
        for c in tgt_chords:
            frac = c["value"]/tgt_total if tgt_total>0 else 0
            w = frac * imp_len * show_frac
            c["t_start"] = cursor
            c["t_end"] = cursor + w
            c["t_mid"] = (c["t_start"] + c["t_end"]) / 2
            c["t_width"] = w
            cursor += w

    # ---- Debug ----
    title_ascii = title.replace("→","->").replace("●","*")
    print(f"\n--- {title_ascii} ---")
    for n in nodes:
        ea0,ea1=exp_arc[n]; ia0,ia1=imp_arc[n]
        print(f"  {n}: exp=[{ea0*180/pi:.1f},{ea1*180/pi:.1f}] ({((ea1-ea0)*180/pi):.1f}deg)  "
              f"imp=[{ia0*180/pi:.1f},{ia1*180/pi:.1f}] ({((ia1-ia0)*180/pi):.1f}deg)")
    for c in chord_list:
        s,t,v = c["source"],c["target"],c["value"]
        print(f"  {s}->{t}: v={v:,.0f}  s=[{c['s_start']*180/pi:.2f},{c['s_end']*180/pi:.2f}]deg  "
              f"t=[{c['t_start']*180/pi:.2f},{c['t_end']*180/pi:.2f}]deg")

    # ---- Draw chords ----
    max_val = max(c["value"] for c in chord_list)
    for c in chord_list:
        alpha = 0.55 + 0.4 * (c["value"]/max_val)
        color = node_colors.get(c["source"], "#888")
        pts = build_ribbon(c["s_start"], c["s_end"], c["t_start"], c["t_end"], R_in)
        ax.add_patch(mpatches.Polygon(pts, closed=True, fc=color, alpha=alpha,
                     ec="none", lw=0, zorder=3))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-1.20,1.20); ax.set_ylim(-1.20,1.20); ax.set_aspect("equal"); ax.axis("off")

def build_ribbon(s_a, s_b, t_a, t_b, R_in):
    """Simple ribbon emerging from inner ring edge R_in at both ends.
    Both edges curve inward symmetrically — clean, no end caps needed."""
    a_src = (s_a + s_b) / 2
    a_tgt = (t_a + t_b) / 2
    w_src = s_b - s_a
    w_tgt = t_b - t_a

    da = a_tgt - a_src
    if da > pi: da -= 2*pi
    elif da < -pi: da += 2*pi

    steps = 60
    pts = []

    # Depth: starts at 0 (touching R_in at ends), dips to 0.45 at center
    def depth_at(t):
        return 0.45 * (1 - (2*t - 1)**2)

    # Forward edge (one side of ribbon, source->target)
    for i in range(steps+1):
        t = i/steps
        ang = a_src + da*t
        w = w_src*(1-t) + w_tgt*t
        half_w = w/2
        r = R_in - depth_at(t)
        pts.append((r*cos(ang - half_w), r*sin(ang - half_w)))

    # Return edge (other side, target->source)
    for i in range(steps, -1, -1):
        t = i/steps
        ang = a_src + da*t
        w = w_src*(1-t) + w_tgt*t
        half_w = w/2
        r = R_in - depth_at(t)
        pts.append((r*cos(ang + half_w), r*sin(ang + half_w)))

    return pts

# ---- Plot: CN->US only for verification ----
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))

draw_chord(ax1, full_trade, "trade_value", segment_order,
           "Trade (2015) - CN->US only\nSolid=exports | Faded=imports | White ticks = arc split",
           single_chord=("CN","US"))

draw_chord(ax2, cite, "weight", segment_order,
           "Citations (2015) - CN->US only\nSolid=citing | Faded=cited | White ticks = arc split",
           single_chord=("CN","US"))

fig.suptitle("Chord Diagram - Single CN->US Chord (Reference Style)",
             fontsize=13, fontweight="bold", y=0.99)
out_path=f"../plots/chord_{YEAR}_cn2us.png"
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out_path}")
