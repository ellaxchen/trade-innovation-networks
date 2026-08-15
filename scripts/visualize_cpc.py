"""
Visualize citation networks split by CPC technology section.
Multi-panel grid showing which countries dominate which technologies.

Usage: python visualize_cpc.py [--year 2015] [--top-n 12]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import argparse
from math import pi, sin, cos, atan2

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2015)
parser.add_argument("--top-n", type=int, default=12)
parser.add_argument("--max-edges", type=int, default=60)
args = parser.parse_args()

YEAR, TOP_N, MAX_EDGES = args.year, getattr(args, "top_n", 12), getattr(args, "max_edges", 60)

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

CPC_NAMES = {
    "A": "Human Necessities", "B": "Operations & Transport",
    "C": "Chemistry", "D": "Textiles", "E": "Construction",
    "F": "Engineering", "G": "Physics", "H": "Electricity",
}

# Focus on the major sections
MAJOR_SECTIONS = ["A", "B", "C", "F", "G", "H"]

# Color scheme: same net position gradient as main viz
NET_CMAP = LinearSegmentedColormap.from_list("netpos", [
    "#B8D8F0", "#9AC8E6", "#7DB9DC", "#5BA3D9", "#3D8BC0", "#1E6BA8", "#0B2D5C",
])

EDGE_COLOR = "#C87A4A"  # rust orange (matches citation panel)

# ---------- load data ----------
print(f"Loading CPC-filtered citation data for {YEAR}...")
df = pd.read_csv("../processed/country_citation_by_cpc.csv")
df = df[(df["year"] == YEAR)
        & (~df["citing_country"].isin(NON_COUNTRIES))
        & (~df["cited_country"].isin(NON_COUNTRIES))]
df = df[df["citing_country"] != df["cited_country"]]  # cross-border only
print(f"  {len(df):,} cross-border edges across {df.cpc_section.nunique()} sections")

# ---------- per-section networks ----------
def build_section_network(data, nodes, max_e):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    sub = data[data["citing_country"].isin(nodes) & data["cited_country"].isin(nodes)]
    sub = sub.nlargest(max_e, "citation_count")
    for _, r in sub.iterrows():
        g.add_edge(r["citing_country"], r["cited_country"], weight=r["citation_count"])
    return g

def node_sizes(G, nodes):
    total = {n: 0 for n in nodes}
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        total[u] = total.get(u, 0) + w
        total[v] = total.get(v, 0) + w
    vals = np.array([total[n] for n in nodes])
    sqrt_vals = np.sqrt(vals)
    lo, hi = sqrt_vals.min(), sqrt_vals.max()
    if hi == lo:
        return {n: 400 for n in nodes}
    return {n: 100 + 1800 * (s - lo) / (hi - lo) for n, s in zip(nodes, sqrt_vals)}

def net_position(data, nodes):
    sent = data.groupby("citing_country")["citation_count"].sum()
    recv = data.groupby("cited_country")["citation_count"].sum()
    result = {}
    for n in nodes:
        s = sent.get(n, 0)
        r = recv.get(n, 0)
        t = s + r
        result[n] = (s - r) / t if t > 0 else 0
    return result

def edge_widths(G):
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if not weights:
        return {}
    lo, hi = min(weights), max(weights)
    if hi == lo:
        return {(u, v): 1.5 for u, v in G.edges()}
    return {(u, v): 0.3 + 4.0 * (d["weight"] - lo) / (hi - lo)
            for u, v, d in G.edges(data=True)}

# ---------- determine shared top nodes across all sections ----------
global_nodes = set()
for sec in MAJOR_SECTIONS:
    sdf = df[df["cpc_section"] == sec]
    if len(sdf) == 0:
        continue
    a = sdf[["citing_country", "citation_count"]].rename(
        columns={"citing_country": "c", "citation_count": "w"})
    b = sdf[["cited_country", "citation_count"]].rename(
        columns={"cited_country": "c", "citation_count": "w"})
    nodes = pd.concat([a, b]).groupby("c")["w"].sum().nlargest(TOP_N).index.tolist()
    global_nodes.update(nodes)

global_nodes = sorted(global_nodes)
print(f"  {len(global_nodes)} unique countries across all sections")

# ---------- draw ----------
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
axes = axes.flatten()

for idx, sec in enumerate(MAJOR_SECTIONS):
    ax = axes[idx]
    sdf = df[df["cpc_section"] == sec]
    if len(sdf) == 0:
        ax.set_title(f"{sec}: {CPC_NAMES.get(sec, sec)}\n(no data)", fontsize=10)
        ax.axis("off")
        continue

    # Build network for this section
    G = build_section_network(sdf, global_nodes, MAX_EDGES)
    active_nodes = [n for n in G.nodes() if G.degree(n) > 0]

    # Circular layout using global positions
    n_all = len(global_nodes)
    pos_all = {node: (cos(2 * pi * i / n_all - pi/2), sin(2 * pi * i / n_all - pi/2))
               for i, node in enumerate(global_nodes)}

    sizes = node_sizes(G, active_nodes)
    net_pos = net_position(sdf, active_nodes)
    lw = edge_widths(G)

    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)

    # Draw edges
    drawn = set()
    for u, v, d in sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0)):
        if (v, u) in drawn:
            continue
        x1, y1 = pos_all[u]
        x2, y2 = pos_all[v]
        w = d.get("weight", 1)
        elw = lw.get((u, v), 0.5)
        alpha = 0.25 + 0.70 * (w / max_w)

        has_rev = G.has_edge(v, u)
        bulge = 0.22 if has_rev else 0.06

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 0.01:
            continue
        nd = -dy / length
        nd2 = dx / length
        cx, cy = mx + nd * bulge * length, my + nd2 * bulge * length

        path_data = [(Path.MOVETO, (x1, y1)), (Path.CURVE3, (cx, cy)), (Path.CURVE3, (x2, y2))]
        codes, verts = zip(*path_data)
        path = mpatches.PathPatch(Path(verts, codes),
            facecolor="none", edgecolor=EDGE_COLOR, lw=elw, alpha=min(alpha, 0.85),
            zorder=2, capstyle="round", joinstyle="round")
        ax.add_patch(path)

        # Arrowhead
        tx = 2 * (x2 - cx)
        ty = 2 * (y2 - cy)
        tn = np.hypot(tx, ty)
        if tn > 0.001:
            tx, ty = tx / tn, ty / tn
        nr = np.sqrt(sizes.get(v, 400)) * 0.0015 + 0.02
        arr_s = 0.008 + 0.003 * elw
        tip_x, tip_y = x2 - tx * nr, y2 - ty * nr
        px, py = -ty, tx
        bx, by = tip_x - tx * arr_s * 2.5, tip_y - ty * arr_s * 2.5
        lx, ly = bx + px * arr_s * 1.2, by + py * arr_s * 1.2
        rx, ry = bx - px * arr_s * 1.2, by - py * arr_s * 1.2
        ax.add_patch(mpatches.Polygon(
            [(tip_x, tip_y), (lx, ly), (rx, ry)], closed=True,
            facecolor=EDGE_COLOR, alpha=alpha, zorder=6))

        if has_rev:
            drawn.add((u, v))
            # Draw reverse with negative bulge
            rw_val = G[v][u].get("weight", 1)
            rlw = lw.get((v, u), 0.5)
            ralpha = 0.25 + 0.70 * (rw_val / max_w)
            rcx, rcy = mx + nd * (-0.22) * length, my + nd2 * (-0.22) * length
            rpath_data = [(Path.MOVETO, (x2, y2)), (Path.CURVE3, (rcx, rcy)), (Path.CURVE3, (x1, y1))]
            rcodes, rverts = zip(*rpath_data)
            rpath = mpatches.PathPatch(Path(rverts, rcodes),
                facecolor="none", edgecolor=EDGE_COLOR, lw=rlw, alpha=min(ralpha, 0.85),
                zorder=2, capstyle="round", joinstyle="round")
            ax.add_patch(rpath)
            # Reverse arrowhead
            rtx, rty = 2*(x1-rcx), 2*(y1-rcy)
            rtn = np.hypot(rtx, rty)
            if rtn > 0.001:
                rtx, rty = rtx/rtn, rty/rtn
            rnr = np.sqrt(sizes.get(u, 400)) * 0.0015 + 0.02
            rarr_s = 0.008 + 0.003 * rlw
            rtip_x, rtip_y = x1 - rtx*rnr, y1 - rty*rnr
            rpx, rpy = -rty, rtx
            rbx, rby = rtip_x - rtx*rarr_s*2.5, rtip_y - rty*rarr_s*2.5
            rlx, rly = rbx + rpx*rarr_s*1.2, rby + rpy*rarr_s*1.2
            rrx, rry = rbx - rpx*rarr_s*1.2, rby - rpy*rarr_s*1.2
            ax.add_patch(mpatches.Polygon(
                [(rtip_x, rtip_y), (rlx, rly), (rrx, rry)], closed=True,
                facecolor=EDGE_COLOR, alpha=ralpha, zorder=6))

    # Draw nodes
    for node in active_nodes if active_nodes else global_nodes:
        x, y = pos_all[node]
        s = sizes.get(node, 400)
        nv = net_pos.get(node, 0)
        fc = NET_CMAP((max(-1, min(1, nv)) + 1) / 2)
        size_in_ax = np.sqrt(s) * 0.08
        ax.scatter(x, y, s=size_in_ax, c=[fc], alpha=0.92,
                   edgecolors="#555555", linewidths=0.5, zorder=5)
        angle = atan2(y, x)
        r = 1.14
        lx, ly = r * cos(angle), r * sin(angle)
        ha = "center"
        if lx > 0.05:
            ha = "left"
        elif lx < -0.05:
            ha = "right"
        ax.annotate(node, (lx, ly), fontsize=6.5, fontweight="bold",
                    ha=ha, va="center", annotation_clip=False)

    total_cit = sdf["citation_count"].sum()
    n_edges = G.number_of_edges()
    name = CPC_NAMES.get(sec, sec)
    ax.set_title(f"{sec}: {name}\n{n_edges} edges, {total_cit:,.0f} cross-border citations",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

# ---------- summary bar chart ----------
ax_summary = axes[5]  # last panel (replaces the 6th section slot)
ax_summary.axis("on")

section_totals = {}
section_cross = {}
for sec in MAJOR_SECTIONS:
    sdf = df[df["cpc_section"] == sec]
    if len(sdf) == 0:
        continue
    section_totals[sec] = sdf["citation_count"].sum()
    section_cross[sec] = sdf[sdf["citing_country"] != sdf["cited_country"]]["citation_count"].sum()

sections = list(section_totals.keys())
totals = [section_totals[s] / 1e6 for s in sections]
cross_vals = [section_cross[s] / 1e6 for s in sections]
labels = [f"{s}: {CPC_NAMES.get(s, s)}" for s in sections]

y_pos = range(len(sections))
bars_total = ax_summary.barh(y_pos, totals, height=0.5, color="#C87A4A", alpha=0.8, label="Total")
bars_cross = ax_summary.barh(y_pos, cross_vals, height=0.5, color="#3A7CA5", alpha=0.8, label="Cross-border")

ax_summary.set_yticks(y_pos)
ax_summary.set_yticklabels(labels, fontsize=7.5)
ax_summary.set_xlabel("Citations (millions)", fontsize=8)
ax_summary.set_title("Citation Volume by Technology Section", fontsize=10, fontweight="bold")
ax_summary.legend(fontsize=7, loc="lower right")
ax_summary.invert_yaxis()

# Add share of cross-border
for i, (t, c) in enumerate(zip(totals, cross_vals)):
    pct = c / t * 100 if t > 0 else 0
    ax_summary.text(t + max(totals) * 0.02, i, f"{pct:.1f}% cross-border",
                    fontsize=6.5, va="center", color="#333333")

fig.suptitle(f"Knowledge Flows by Technology Sector — {YEAR} (cross-border only)",
             fontsize=13, fontweight="bold", y=0.99)

out_path = f"../plots/cpc_sections_{YEAR}.png"
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out_path}")
