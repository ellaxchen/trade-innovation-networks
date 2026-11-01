"""
Side-by-side circular networks for the semiconductor cut.

Trade   = HS 8541+8542 export flows (exporter -> importer)
Knowledge = CPC H01L family citations (citing -> cited)

Node size   = total flow in that network (sqrt-scaled)
Node colour = net position (dark = net exporter / net citing,
              light = net importer / net cited)
Edges       = top cross-border flows, curved toward the centre, arrowheads
              at the exact node boundary.

Usage: python visualize_semiconductor_side_by_side.py [--years 1995,2005,2013]
                                                      [--country auth|inventor]
                                                      [--top-n 14] [--max-edges 80]
Output: ../plots/semiconductor_side_by_side_{year}[_inventor].png
"""
import argparse
from math import pi, sin, cos, atan2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--years", type=str, default="1995,2000,2005,2010,2013")
parser.add_argument("--country", choices=["auth", "inventor"], default="auth")
parser.add_argument("--top-n", type=int, default=14)
parser.add_argument("--max-edges", type=int, default=80)
args = parser.parse_args()

YEARS = [int(y) for y in args.years.split(",")]
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
TRADE_CMAP = LinearSegmentedColormap.from_list(
    "t", ["#B8D8F0", "#9AC8E6", "#7DB9DC", "#5BA3D9", "#3D8BC0", "#1E6BA8", "#0B2D5C"])
CITE_CMAP = LinearSegmentedColormap.from_list(
    "c", ["#F5D5B8", "#EDC09E", "#E0A87A", "#D4915A", "#C87A4A", "#A85A2E", "#7A3512"])
TRADE_EC, CITE_EC = "#3A7CA5", "#C87A4A"
SUFFIX = "" if args.country == "auth" else "_inventor"
CITE_FILE = ("../processed/semiconductor_citations_family_by_year.csv"
             if args.country == "auth"
             else "../processed/semiconductor_citations_family_by_year_inventor.csv")

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
iso_to_name["TW"] = "Taiwan"

print("Loading semiconductor edges...")
trade = pd.read_csv("../processed/semiconductor_trade_edges_by_year.csv")
trade["source"] = trade["exporter"].map(baci_to_iso)
trade["target"] = trade["importer"].map(baci_to_iso)
trade = trade.dropna(subset=["source", "target"])
trade = trade[~trade["source"].isin(NON_COUNTRIES) &
              ~trade["target"].isin(NON_COUNTRIES)]
trade = trade[trade["source"] != trade["target"]]
trade = trade.groupby(["year", "source", "target"], as_index=False)["trade_value"].sum()

cite = pd.read_csv(CITE_FILE)
cite = cite.rename(columns={"citing_country": "source",
                            "cited_country": "target",
                            "citation_count": "weight"})
cite = cite[~cite["source"].isin(NON_COUNTRIES) &
            ~cite["target"].isin(NON_COUNTRIES)]
cite = cite[cite["source"] != cite["target"]]


def total_by_country(df, wcol):
    s = df.groupby("source")[wcol].sum()
    t = df.groupby("target")[wcol].sum()
    return s.add(t, fill_value=0)


def node_sizes(G, nodes):
    tot = {n: 0.0 for n in nodes}
    for u, v, d in G.edges(data=True):
        tot[u] += d["weight"]; tot[v] += d["weight"]
    sv = np.sqrt([tot[n] for n in nodes])
    lo, hi = sv.min(), sv.max()
    if hi == lo:
        return {n: 800 for n in nodes}
    return {n: 200 + 3300 * (s - lo) / (hi - lo) for n, s in zip(nodes, sv)}


def net_position(df, wcol, nodes):
    s = df[df["source"].isin(nodes)].groupby("source")[wcol].sum()
    r = df[df["target"].isin(nodes)].groupby("target")[wcol].sum()
    return {n: ((s.get(n, 0) - r.get(n, 0)) / (s.get(n, 0) + r.get(n, 0)))
            if (s.get(n, 0) + r.get(n, 0)) > 0 else 0 for n in nodes}


def edge_widths(G):
    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    if not ws:
        return {}
    lo, hi = min(ws), max(ws)
    if hi == lo:
        return {(u, v): 2.0 for u, v in G.edges()}
    return {(u, v): 0.3 + 5.5 * (d["weight"] - lo) / (hi - lo)
            for u, v, d in G.edges(data=True)}


def bezier_pt(x1, y1, cx, cy, x2, y2, t):
    mt = 1 - t
    return (mt * mt * x1 + 2 * mt * t * cx + t * t * x2,
            mt * mt * y1 + 2 * mt * t * cy + t * t * y2)


def bezier_dir(x1, y1, cx, cy, x2, y2, t):
    mt = 1 - t
    tx = 2 * mt * (cx - x1) + 2 * t * (x2 - cx)
    ty = 2 * mt * (cy - y1) + 2 * t * (y2 - cy)
    tn = np.hypot(tx, ty)
    return (tx / tn, ty / tn) if tn > 0.001 else (1.0, 0.0)


def draw_curved_edge(ax, x1, y1, x2, y2, to_rad, lw, color, alpha,
                     bulge, use_perp=False):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 0.01:
        return
    if use_perp:
        p1x, p1y = -dy / L, dx / L
        p2x, p2y = dy / L, -dx / L
        d1 = np.hypot(mx + bulge * p1x, my + bulge * p1y)
        d2 = np.hypot(mx + bulge * p2x, my + bulge * p2y)
        ndx, ndy = (p1x, p1y) if d1 < d2 else (p2x, p2y)
    else:
        mid_dist = np.hypot(mx, my)
        ndx, ndy = (-mx / mid_dist, -my / mid_dist) if mid_dist > 0.001 \
            else (-dy / L, dx / L)
    cx, cy = mx + ndx * bulge, my + ndy * bulge

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        bx, by = bezier_pt(x1, y1, cx, cy, x2, y2, mid)
        if np.hypot(bx - x2, by - y2) > to_rad:
            lo = mid
        else:
            hi = mid
    t_stop = (lo + hi) / 2
    ex, ey = bezier_pt(x1, y1, cx, cy, x2, y2, t_stop)
    dtx, dty = bezier_dir(x1, y1, cx, cy, x2, y2, t_stop)

    ts = np.linspace(0, t_stop, max(15, int(L * 40)))
    pts = [bezier_pt(x1, y1, cx, cy, x2, y2, t) for t in ts]
    pd_ = [(Path.MOVETO, pts[0])] + [(Path.LINETO, p) for p in pts[1:]]
    codes, verts = zip(*pd_)
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), fc="none", ec=color, lw=lw,
        alpha=min(alpha, 0.85), zorder=2, capstyle="round", joinstyle="round"))

    arr_s = 0.012 + 0.004 * lw
    tip_x, tip_y = ex, ey
    bx, by = tip_x - dtx * arr_s * 2.5, tip_y - dty * arr_s * 2.5
    px, py = -dty, dtx
    lx, ly = bx + px * arr_s * 1.5, by + py * arr_s * 1.5
    rx, ry = bx - px * arr_s * 1.5, by - py * arr_s * 1.5
    ax.add_patch(mpatches.Polygon(
        [(tip_x, tip_y), (lx, ly), (rx, ry)], closed=True,
        fc=color, ec="none", alpha=alpha, zorder=7))


def draw_panel(fig, ax, G, pos, sizes, netpos, edge_w, edge_color,
               cmap, title, ptype):
    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    drawn = set()
    for u, v, d in sorted(G.edges(data=True),
                          key=lambda e: e[2].get("weight", 0)):
        if u == v or (v, u) in drawn:
            continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        lw = edge_w.get((u, v), 0.5)
        w = d.get("weight", 1)
        alpha = 0.25 + 0.70 * (w / max_w)
        has_rev = G.has_edge(v, u)
        chord = np.hypot(x2 - x1, y2 - y1)
        bulge = (max(0.30, 1.0 / (chord + 0.3)) if has_rev
                 else max(0.20, 0.6 / (chord + 0.3)))
        mid_dist = np.hypot((x1 + x2) / 2, (y1 + y2) / 2)
        bulge = min(bulge, mid_dist * 0.85)
        to_rad = np.sqrt(sizes.get(v, 500)) * 0.0025 + 0.03
        draw_curved_edge(ax, x1, y1, x2, y2, to_rad, lw,
                         edge_color, alpha, bulge)
        if has_rev:
            drawn.add((u, v))
            rw = G[v][u].get("weight", 1)
            rlw = edge_w.get((v, u), 0.5)
            ra = 0.25 + 0.70 * (rw / max_w)
            to_rad2 = np.sqrt(sizes.get(u, 500)) * 0.0025 + 0.03
            draw_curved_edge(ax, x2, y2, x1, y1, to_rad2, rlw,
                             edge_color, ra, abs(bulge) * 0.6, use_perp=True)

    for node in G.nodes():
        x, y = pos[node]
        s = sizes.get(node, 500)
        fc = cmap((max(-1, min(1, netpos.get(node, 0))) + 1) / 2)
        ax.scatter(x, y, s=s, c=[fc], alpha=0.92, ec="#555555",
                   lw=0.8, zorder=5)
        angle = atan2(y, x)
        lx, ly = 1.09 * cos(angle), 1.09 * sin(angle)
        ha = "center"
        if lx > 0.05:
            ha = "left"
        elif lx < -0.05:
            ha = "right"
        ax.annotate(node, (lx, ly), fontsize=8.5, fontweight="bold",
                    ha=ha, va="center", annotation_clip=False, zorder=10)

    bbox = ax.get_position()
    left_lab, right_lab = (("net importer", "net exporter") if ptype == "trade"
                           else ("net cited", "net citing"))
    ly = bbox.y1 + 0.01; lh = 0.045
    bw = bbox.width * 0.30; gap = bbox.width * 0.02

    a1 = fig.add_axes([bbox.x0, ly, bw, lh])
    a1.set_xlim(-0.5, 2.5); a1.set_ylim(-0.6, 0.6); a1.axis("off")
    a1.text(-0.3, 0.30, "Node size = total flow", fontsize=5.5, fontweight="bold")
    for i, (sv, sl) in enumerate(zip([200, 1200, 3500], ["Low", "Mid", "High"])):
        a1.scatter(i * 1.1, 0, s=sv / 12, c=edge_color, alpha=0.7,
                   ec="#555555", lw=0.5)
        a1.text(i * 1.1, -0.40, sl, fontsize=4.5, ha="center")

    a2 = fig.add_axes([bbox.x0 + bw + gap, ly, bw, lh])
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.axis("off")
    g = np.linspace(0, 1, 256).reshape(1, -1)
    a2.imshow(g, aspect="auto", cmap=cmap, extent=[0.05, 0.75, 0.15, 0.55])
    a2.text(0.05, 0.70, left_lab, fontsize=4.8, ha="left", va="bottom")
    a2.text(0.75, 0.70, right_lab, fontsize=4.8, ha="right", va="bottom")
    a2.text(0.40, 0.75, "Node colour", fontsize=5.5, fontweight="bold",
            ha="center")

    a3 = fig.add_axes([bbox.x0 + 2 * (bw + gap), ly, bw, lh])
    a3.set_xlim(0, 3); a3.set_ylim(0, 1); a3.axis("off")
    a3.text(0, 0.72, "Edge width = flow", fontsize=5.5, fontweight="bold")
    for i, (lv, lb) in enumerate([(0.3, "Low"), (2.5, "Mid"), (5.5, "High")]):
        a3.plot([0.2, 1.2], [0.42 - i * 0.22, 0.42 - i * 0.22],
                color=edge_color, lw=lv, alpha=0.7, solid_capstyle="round")
        a3.text(1.4, 0.42 - i * 0.22, lb, fontsize=4.5, va="center")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-1.20, 1.20); ax.set_ylim(-1.20, 1.20)
    ax.set_aspect("equal"); ax.axis("off")


for YEAR in YEARS:
    print(f"  {YEAR}...")
    t = trade[trade["year"] == YEAR]
    c = cite[cite["year"] == YEAR]
    if len(t) == 0:
        print(f"    no trade data for {YEAR}, skipping")
        continue

    tv = total_by_country(t, "trade_value")
    cv = total_by_country(c, "weight")
    rank_t = tv.rank(ascending=False)
    rank_c = cv.rank(ascending=False)
    all_idx = rank_t.index.union(rank_c.index)
    combined = (rank_t.reindex(all_idx).fillna(rank_t.max() + 5) +
                rank_c.reindex(all_idx).fillna(rank_c.max() + 5))
    nodes = combined.nsmallest(args.top_n).index.tolist()
    for anchor in ["CN", "US", "DE", "JP", "KR", "TW"]:
        if anchor in combined.index and anchor not in nodes:
            nodes.append(anchor)
    nodes = nodes[:args.top_n + 2]

    def build_graph(df, wcol):
        G = nx.DiGraph(); G.add_nodes_from(nodes)
        sub = df[df["source"].isin(nodes) & df["target"].isin(nodes)]
        sub = sub.nlargest(args.max_edges, wcol)
        for _, r in sub.iterrows():
            G.add_edge(r["source"], r["target"], weight=r[wcol])
        return G

    Gt = build_graph(t, "trade_value")
    Gc = build_graph(c, "weight")

    # layout: anchors at quarter positions
    n_nodes = len(nodes)
    def vol(node):
        return (sum(d["weight"] for _, _, d in Gt.edges(node, data=True)) +
                sum(d["weight"] for _, _, d in Gc.edges(node, data=True)))
    anchored = ["CN", "US", "DE", "JP"]
    anchor_pos = [0, n_nodes // 4, 2 * n_nodes // 4, 3 * n_nodes // 4]
    amap = {a: p for a, p in zip(anchored, anchor_pos) if a in nodes}
    remaining = sorted([n for n in nodes if n not in amap], key=vol, reverse=True)
    available = [i for i in range(n_nodes) if i not in amap.values()]
    ordered = [None] * n_nodes
    for a, p in amap.items():
        ordered[p] = a
    for i, n in enumerate(remaining):
        ordered[available[i]] = n
    pos = {}
    for i, n in enumerate(ordered):
        angle = 2 * pi * i / n_nodes - pi / 2
        pos[n] = (cos(angle), sin(angle))

    t_sizes = node_sizes(Gt, nodes)
    c_sizes = node_sizes(Gc, nodes)
    t_net = net_position(t, "trade_value", nodes)
    c_net = net_position(c, "weight", nodes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    t_tl = (f"Semiconductor trade ({YEAR})\nHS 8541+8542, "
            f"{Gt.number_of_edges()} top cross-border flows\narrow → importer")
    c_tl = (f"Semiconductor knowledge ({YEAR})\nCPC H01L citations, "
            f"{Gc.number_of_edges()} top cross-border flows\narrow → cited country")
    draw_panel(fig, ax1, Gt, pos, t_sizes, t_net, edge_widths(Gt),
               TRADE_EC, TRADE_CMAP, t_tl, "trade")
    draw_panel(fig, ax2, Gc, pos, c_sizes, c_net, edge_widths(Gc),
               CITE_EC, CITE_CMAP, c_tl, "cite")
    fig.suptitle(
        f"Semiconductors: Trade vs. Knowledge — {YEAR}"
        + (" (filing-office country)" if args.country == "auth"
           else " (inventor country)"),
        fontsize=14, fontweight="bold", y=0.995)
    plt.subplots_adjust(top=0.82, bottom=0.02, left=0.02, right=0.98,
                        wspace=0.05)
    out_path = f"../plots/semiconductor_side_by_side_{YEAR}{SUFFIX}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved -> {out_path}")

print("Done.")
