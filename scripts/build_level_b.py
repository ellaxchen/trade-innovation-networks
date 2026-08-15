"""
Level B — Industry × Country network builder.
Builds directed edges for trade (HS2 chapter) and citation (CPC section) networks
where nodes are country×industry pairs (e.g., "US_Electronics", "CN_Electronics").

Electronics is the default demo: HS85 (Electrical machinery) ↔ CPC H (Electricity).

Usage:
  python build_level_b.py                              # Electronics, 2015
  python build_level_b.py --sector vehicles            # Vehicles, 2015
  python build_level_b.py --year 2010 --top-n 15       # 2010, top 15 countries
  python build_level_b.py --list-sectors               # show available sector pairs
  python build_level_b.py --cross-border-only          # exclude self-loops

Output:
  ../processed/level_b_electronics_2015_trade.csv      # trade edges (Level B nodes)
  ../processed/level_b_electronics_2015_cite.csv       # citation edges (Level B nodes)
  ../plots/level_b_electronics_2015.png                # side-by-side visualization
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

# ---- Sector concordance: trade HS2 ↔ citation CPC section ----
SECTORS = {
    "electronics": {"hs2": "85", "hs_label": "Electronics (HS85)",
                    "cpc": "H",  "cpc_label": "Electricity (CPC H)", "short": "Elec"},
    "vehicles":    {"hs2": "87", "hs_label": "Vehicles (HS87)",
                    "cpc": "B",  "cpc_label": "Ops & Transport (CPC B)", "short": "Vehicle"},
    "pharma":      {"hs2": "30", "hs_label": "Pharma (HS30)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "short": "Pharma"},
    "machinery":   {"hs2": "84", "hs_label": "Machinery (HS84)",
                    "cpc": "F",  "cpc_label": "Engineering (CPC F)", "short": "Machine"},
    "oil_fuels":   {"hs2": "27", "hs_label": "Oil & Fuels (HS27)",
                    "cpc": "B",  "cpc_label": "Ops & Transport (CPC B)", "short": "Oil"},
    "iron_steel":  {"hs2": "72", "hs_label": "Iron & Steel (HS72)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "short": "Steel"},
    "plastics":    {"hs2": "39", "hs_label": "Plastics (HS39)",
                    "cpc": "C",  "cpc_label": "Chemistry (CPC C)", "short": "Plastic"},
    "apparel":     {"hs2": "62", "hs_label": "Apparel (HS62)",
                    "cpc": "D",  "cpc_label": "Textiles (CPC D)", "short": "Apparel"},
}

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# Color palettes for trade (blue) and citation (orange)
TRADE_CMAP = LinearSegmentedColormap.from_list(
    "trade", ["#B8D8F0", "#9AC8E6", "#7DB9DC", "#5BA3D9", "#3D8BC0", "#1E6BA8", "#0B2D5C"])
CITE_CMAP = LinearSegmentedColormap.from_list(
    "cite", ["#F5D5B8", "#EDC09E", "#E0A87A", "#D4915A", "#C87A4A", "#A85A2E", "#7A3512"])

parser = argparse.ArgumentParser()
parser.add_argument("--sector", type=str, default="electronics",
                    help="Sector key (electronics, vehicles, pharma, machinery, ...)")
parser.add_argument("--year", type=int, default=2015)
parser.add_argument("--top-n", type=int, default=12,
                    help="Number of top countries to include")
parser.add_argument("--max-edges", type=int, default=40)
parser.add_argument("--cross-border-only", action="store_true",
                    help="Exclude self-loops (domestic flows)")
parser.add_argument("--list-sectors", action="store_true",
                    help="Print available sector pairs and exit")
parser.add_argument("--no-plot", action="store_true",
                    help="Skip visualization, only write CSVs")
parser.add_argument("--edge-threshold", type=float, default=0,
                    help="Minimum edge weight fraction of max (0-1) to display")
args = parser.parse_args()

if args.list_sectors:
    print("Available sector pairs (Level B concordance):\n")
    print(f"{'Key':<14} {'Trade (HS2)':<22} {'Citation (CPC)':<24} {'Short':<10}")
    print("-" * 70)
    for key, s in SECTORS.items():
        print(f"  {key:<12} {s['hs_label']:<22} {s['cpc_label']:<24} {s['short']:<10}")
    print("\nNote: Some sectors share CPC sections (e.g., vehicles & oil both use CPC B).")
    print("This is the concordance problem described in PROJECT_GUIDE.md Part 2.")
    exit(0)

SECTOR = SECTORS.get(args.sector)
if SECTOR is None:
    print(f"Unknown sector '{args.sector}'. Use --list-sectors to see options.")
    exit(1)

YEAR = args.year
TOP_N = args.top_n
MAX_EDGES = args.max_edges
SHORT = SECTOR["short"]

# ---- Load country code maps ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
BACI_TO_ISO = dict(zip(cc["country_code"], cc["country_iso2"]))
NAME_TO_ISO = dict(zip(cc["country_name"], cc["country_iso2"]))

# ============================================================
# 1. Load and normalize trade data for the HS2 chapter
# ============================================================
print(f"\n{'='*60}")
print(f"Level B: {SECTOR['hs_label']} ↔ {SECTOR['cpc_label']} — {YEAR}")
print(f"{'='*60}")

print(f"\n[1/4] Loading trade data (HS{SECTOR['hs2']})...")
trade_cols = ["year", "exporter", "importer", "hs2", "trade_value"]
trade_raw = pd.read_parquet("../rawdata/baci_panel_hs6.parquet", columns=trade_cols)
# Filter year first (on int column — fast), THEN filter hs2 (string comparison on subset)
trade_raw = trade_raw[trade_raw["year"] == YEAR]
# hs2 is Int32 — filter by integer value
hs2_int = int(SECTOR["hs2"])
trade_raw = trade_raw[trade_raw["hs2"] == hs2_int]
print(f"  {len(trade_raw):,} trade rows for HS{SECTOR['hs2']} in {YEAR}")

# Aggregate to country-pair level
trade = (
    trade_raw.groupby(["exporter", "importer"], as_index=False)["trade_value"]
    .sum()
    .sort_values("trade_value", ascending=False)
)
del trade_raw

# Map BACI numeric codes → ISO2
trade["source"] = trade["exporter"].map(BACI_TO_ISO)
trade["target"] = trade["importer"].map(BACI_TO_ISO)
# Fallback: try name-based mapping for codes not in the lookup
trade["source"] = trade["source"].fillna(
    trade["exporter"].map(dict(zip(cc["country_code"], cc["country_name"]))).map(NAME_TO_ISO))
trade["target"] = trade["target"].fillna(
    trade["importer"].map(dict(zip(cc["country_code"], cc["country_name"]))).map(NAME_TO_ISO))
trade_before = len(trade)
trade = trade.dropna(subset=["source", "target"])
trade = trade[~trade["source"].isin(NON_COUNTRIES) & ~trade["target"].isin(NON_COUNTRIES)]
print(f"  {trade_before - len(trade):,} rows dropped (no ISO mapping or non-country)")
print(f"  {len(trade):,} directed country-pair flows")

total_trade = trade["trade_value"].sum()
cross_trade = trade[trade["source"] != trade["target"]]["trade_value"].sum()
print(f"  Total trade: ${total_trade/1e6:,.0f}M  |  Cross-border: ${cross_trade/1e6:,.0f}M "
      f"({cross_trade/total_trade*100:.1f}%)")

# ============================================================
# 2. Load and normalize citation data for the CPC section
# ============================================================
print(f"\n[2/4] Loading citation data (CPC {SECTOR['cpc']})...")
cite_all = pd.read_csv("../processed/country_citation_family_by_cpc.csv")
cite = cite_all[
    (cite_all["year"] == YEAR) &
    (cite_all["cpc_section"] == SECTOR["cpc"]) &
    (~cite_all["citing_country"].isin(NON_COUNTRIES)) &
    (~cite_all["cited_country"].isin(NON_COUNTRIES))
].copy()
cite = cite.rename(columns={
    "citing_country": "source", "cited_country": "target",
    "citation_count": "weight"
})
print(f"  {len(cite):,} country-pair citation flows for CPC {SECTOR['cpc']} in {YEAR}")

total_cite = cite["weight"].sum()
cross_cite = cite[cite["source"] != cite["target"]]["weight"].sum()
print(f"  Total citations: {total_cite:,}  |  Cross-border: {cross_cite:,} "
      f"({cross_cite/total_cite*100:.1f}%)")

# ============================================================
# 3. Identify common top countries and build Level B nodes
# ============================================================
print(f"\n[3/4] Building Level B nodes (country×industry)...")

# Compute total trade volume per country (source + target)
t_src = trade.groupby("source")["trade_value"].sum()
t_tgt = trade.groupby("target")["trade_value"].sum()
trade_vol = (t_src.add(t_tgt, fill_value=0)).sort_values(ascending=False)

# Compute total citation volume per country
c_src = cite.groupby("source")["weight"].sum()
c_tgt = cite.groupby("target")["weight"].sum()
cite_vol = (c_src.add(c_tgt, fill_value=0)).sort_values(ascending=False)

# Combined ranking: harmonic mean of ranks
rank_df = pd.DataFrame({
    "trade_vol": trade_vol, "cite_vol": cite_vol,
    "trade_rank": trade_vol.rank(ascending=False),
    "cite_rank": cite_vol.rank(ascending=False)
}).fillna(0)
rank_df["combined_rank"] = (rank_df["trade_rank"] + rank_df["cite_rank"]) / 2
rank_df = rank_df.sort_values("combined_rank")

# Select top countries that appear in BOTH networks
top_countries = []
for c in rank_df.index:
    if c in trade_vol.index and c in cite_vol.index:
        top_countries.append(c)
    if len(top_countries) >= TOP_N:
        break

print(f"  Top {len(top_countries)} countries in both networks:")
for i, c in enumerate(top_countries):
    tv = trade_vol.get(c, 0) / 1e6
    cv = cite_vol.get(c, 0)
    print(f"    {i+1:2d}. {c:3s}  trade ${tv:,.0f}M  |  citations {cv:,}")

# Create Level B node labels
def level_b_node(iso_code):
    """Return the Level B node label: 'US_Electronics' style."""
    return f"{iso_code}_{SHORT}"

# Print the trade and citation edges with Level B node labels
print(f"\n  Level B node examples:")
for c in top_countries[:5]:
    print(f"    {c} → {level_b_node(c)}")

# ============================================================
# 4. Build Level B edge lists and save
# ============================================================
print(f"\n[4/4] Building Level B edge lists...")

# Filter to top countries
trade_lb = trade[trade["source"].isin(top_countries) & trade["target"].isin(top_countries)].copy()
cite_lb  = cite[cite["source"].isin(top_countries) & cite["target"].isin(top_countries)].copy()

# Add Level B node labels
trade_lb["source_node"] = trade_lb["source"].apply(level_b_node)
trade_lb["target_node"] = trade_lb["target"].apply(level_b_node)
cite_lb["source_node"]  = cite_lb["source"].apply(level_b_node)
cite_lb["target_node"]  = cite_lb["target"].apply(level_b_node)

# Optional: cross-border only
if args.cross_border_only:
    trade_lb = trade_lb[trade_lb["source"] != trade_lb["target"]]
    cite_lb  = cite_lb[cite_lb["source"] != cite_lb["target"]]

# Save
trade_out = f"../processed/level_b_{args.sector}_{YEAR}_trade.csv"
cite_out  = f"../processed/level_b_{args.sector}_{YEAR}_cite.csv"
trade_lb[["source_node", "target_node", "source", "target", "trade_value"]].to_csv(
    trade_out, index=False)
cite_lb[["source_node", "target_node", "source", "target", "weight"]].to_csv(
    cite_out, index=False)
print(f"  Trade edges:  {len(trade_lb):,} → {trade_out}")
print(f"  Citation edges: {len(cite_lb):,} → {cite_out}")

# Metrics summary
print(f"\n  {'='*50}")
print(f"  Level B Summary — {SECTOR['hs_label']} ↔ {SECTOR['cpc_label']} ({YEAR})")
print(f"  {'='*50}")
print(f"  Nodes:  {len(top_countries)} country×{SHORT} pairs")
print(f"  Trade edges:      {len(trade_lb):,}  (${trade_lb['trade_value'].sum()/1e6:,.0f}M)")
print(f"  Citation edges:   {len(cite_lb):,}  ({cite_lb['weight'].sum():,} citations)")
if args.cross_border_only:
    print(f"  Mode: cross-border only")
else:
    t_self = trade_lb[trade_lb["source"] == trade_lb["target"]]["trade_value"].sum()
    c_self = cite_lb[cite_lb["source"] == cite_lb["target"]]["weight"].sum()
    print(f"  Domestic trade:   ${t_self/1e6:,.0f}M ({t_self/trade_lb['trade_value'].sum()*100:.1f}%)")
    print(f"  Domestic cites:   {c_self:,} ({c_self/cite_lb['weight'].sum()*100:.1f}%)")

# Top 5 trade edges
print(f"\n  Top 5 trade edges:")
for _, r in trade_lb.head(5).iterrows():
    print(f"    {r['source_node']} → {r['target_node']}: ${r['trade_value']/1e6:,.0f}M")

# Top 5 citation edges
print(f"\n  Top 5 citation edges:")
for _, r in cite_lb.nlargest(5, "weight").iterrows():
    print(f"    {r['source_node']} → {r['target_node']}: {r['weight']:,}")

# ============================================================
# 5. Visualization (optional)
# ============================================================
if args.no_plot:
    print("\nSkipping plot (--no-plot). Done.")
    exit(0)

print(f"\n[5/5] Generating Level B visualization...")

# ---- Network layout helpers ----
def build_graph(df, node_cols, weight_col, nodes, max_edges):
    """Build a networkx DiGraph from edge dataframe, filtered to nodes and top edges."""
    g = nx.DiGraph()
    g.add_nodes_from([level_b_node(n) for n in nodes])
    # Filter to nodes and take top edges
    subset = df[df["source"].isin(nodes) & df["target"].isin(nodes)]
    if args.cross_border_only:
        subset = subset[subset["source"] != subset["target"]]
    top = subset.nlargest(max_edges, weight_col)
    for _, r in top.iterrows():
        g.add_edge(level_b_node(r["source"]), level_b_node(r["target"]),
                   weight=r[weight_col])
    return g

def node_size_dict(G, nodes):
    """Map node to sqrt-scaled size based on total flow volume."""
    total = {n: 0 for n in [level_b_node(c) for c in nodes]}
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        total[u] = total.get(u, 0) + w
        total[v] = total.get(v, 0) + w
    vals = np.array([total[n] for n in total])
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {n: 500 for n in total}
    return {n: 120 + 1800 * (v - lo) / (hi - lo) for n, v in total.items()}

def net_position(df, nodes, weight_col):
    """Net position: (out - in) / (out + in), range [-1, 1]."""
    out_s = df[df["source"].isin(nodes)].groupby("source")[weight_col].sum()
    in_s  = df[df["target"].isin(nodes)].groupby("target")[weight_col].sum()
    result = {}
    for n in nodes:
        o = out_s.get(n, 0)
        i = in_s.get(n, 0)
        total = o + i
        result[n] = (o - i) / total if total > 0 else 0
    return result

def edge_width_dict(G):
    """Map edges to line widths scaled 0.3–4.0."""
    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    if not ws:
        return {}
    lo, hi = min(ws), max(ws)
    if hi == lo:
        return {e: 1.5 for e in G.edges()}
    return {(u, v): 0.3 + 3.7 * (d["weight"] - lo) / (hi - lo) for u, v, d in G.edges(data=True)}

def circle_layout(countries, iso_node_map):
    """Position Level B nodes on a circle, grouped by country."""
    n = len(countries)
    angles = {}
    for i, c in enumerate(countries):
        angles[c] = 2 * pi * i / n - pi / 2
    return {iso_node_map[c]: (cos(angles[c]), sin(angles[c])) for c in countries if c in iso_node_map}

def draw_level_b_panel(ax, G, pos_map, sizes, net_positions, edge_w, edge_color, cmap, title):
    """Draw one Level B network panel with curved edges and arrowheads."""
    mw = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    drawn = set()
    for u, v, d in sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0)):
        if u == v or (v, u) in drawn:
            continue
        if u not in pos_map or v not in pos_map:
            continue
        x1, y1 = pos_map[u]
        x2, y2 = pos_map[v]
        lw = edge_w.get((u, v), 0.5)
        w = d.get("weight", 1)
        alpha = 0.30 + 0.65 * (w / mw)
        has_rev = G.has_edge(v, u)
        bulge = 0.22 if has_rev else 0.06
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L < 0.01:
            continue
        ndx, ndy = -dy / L, dx / L
        cx, cy = mx + ndx * bulge * L, my + ndy * bulge * L
        path_data = [(Path.MOVETO, (x1, y1)), (Path.CURVE3, (cx, cy)), (Path.CURVE3, (x2, y2))]
        codes, verts = zip(*path_data)
        ax.add_patch(mpatches.PathPatch(
            Path(verts, codes), fc="none", ec=edge_color,
            lw=lw, alpha=min(alpha, 0.85), zorder=2, capstyle="round"))
        # Arrowhead at target
        tx, ty = 2 * (x2 - cx), 2 * (y2 - cy)
        tn = np.hypot(tx, ty)
        if tn > 0.001:
            tx, ty = tx / tn, ty / tn
        nr = np.sqrt(sizes.get(v, 500)) * 0.0013 + 0.018
        arr_s = 0.006 + 0.003 * lw
        tix, tiy = x2 - tx * nr, y2 - ty * nr
        px, py = -ty, tx
        bx, by = tix - tx * arr_s * 2, tiy - ty * arr_s * 2
        ax.add_patch(mpatches.Polygon(
            [(tix, tiy), (bx + px * arr_s, by + py * arr_s),
             (bx - px * arr_s, by - py * arr_s)],
            closed=True, fc=edge_color, alpha=alpha, zorder=6))
        # Reverse edge if bidirectional
        if has_rev:
            drawn.add((u, v))
            rw = G[v][u].get("weight", 1)
            rlw = edge_w.get((v, u), 0.5)
            ra = 0.30 + 0.65 * (rw / mw)
            rcx, rcy = mx + ndx * (-0.22) * L, my + ndy * (-0.22) * L
            r_path = [(Path.MOVETO, (x2, y2)), (Path.CURVE3, (rcx, rcy)), (Path.CURVE3, (x1, y1))]
            rc, rv = zip(*r_path)
            ax.add_patch(mpatches.PathPatch(
                Path(rv, rc), fc="none", ec=edge_color,
                lw=rlw, alpha=min(ra, 0.85), zorder=2, capstyle="round"))
            # Arrowhead for reverse
            rtx, rty = 2 * (x1 - rcx), 2 * (y1 - rcy)
            rtn = np.hypot(rtx, rty)
            if rtn > 0.001:
                rtx, rty = rtx / rtn, rty / rtn
            rnr = np.sqrt(sizes.get(u, 500)) * 0.0013 + 0.018
            rs = 0.006 + 0.003 * rlw
            rt2x, rt2y = x1 - rtx * rnr, y1 - rty * rnr
            rpx, rpy = -rty, rtx
            rb2x, rb2y = rt2x - rtx * rs * 2, rt2y - rty * rs * 2
            ax.add_patch(mpatches.Polygon(
                [(rt2x, rt2y), (rb2x + rpx * rs, rb2y + rpy * rs),
                 (rb2x - rpx * rs, rb2y - rpy * rs)],
                closed=True, fc=edge_color, alpha=ra, zorder=6))

    # Draw nodes
    for node in G.nodes():
        if node not in pos_map:
            continue
        x, y = pos_map[node]
        s = sizes.get(node, 500)
        # Extract ISO code for net position lookup
        iso = node.rsplit("_", 1)[0]
        nv = net_positions.get(iso, 0)
        fc = cmap((max(-1, min(1, nv)) + 1) / 2)
        ax.scatter(x, y, s=np.sqrt(s) * 5.5, c=[fc], alpha=0.92,
                   ec="#555555", lw=0.5, zorder=5)
        angle = atan2(y, x)
        r = 1.15
        lx, ly = r * cos(angle), r * sin(angle)
        ha = "center"
        if lx > 0.04:
            ha = "left"
        elif lx < -0.04:
            ha = "right"
        ax.annotate(node, (lx, ly), fontsize=5.5, fontweight="bold",
                    ha=ha, va="center", annotation_clip=False)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

# ---- Build graphs ----
G_trade = build_graph(trade_lb, ["source", "target"], "trade_value", top_countries, MAX_EDGES)
G_cite  = build_graph(cite_lb,  ["source", "target"], "weight",      top_countries, MAX_EDGES)

# ---- Shared circle layout ----
country_to_lb = {c: level_b_node(c) for c in top_countries}
lb_positions = circle_layout(top_countries, country_to_lb)

# ---- Node metrics ----
trade_sizes = node_size_dict(G_trade, top_countries)
cite_sizes  = node_size_dict(G_cite, top_countries)
trade_net   = net_position(trade_lb, top_countries, "trade_value")
cite_net    = net_position(cite_lb, top_countries, "weight")
trade_ew    = edge_width_dict(G_trade)
cite_ew     = edge_width_dict(G_cite)

# ---- Plot ----
fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(18, 10))

tv = trade_lb["trade_value"].sum() / 1e6
cv = cite_lb["weight"].sum()
draw_level_b_panel(
    ax_t, G_trade, lb_positions, trade_sizes, trade_net, trade_ew,
    "#3A7CA5", TRADE_CMAP,
    f"{SECTOR['hs_label']} — Trade\n{G_trade.number_of_edges()} edges, ${tv:,.0f}M total")
draw_level_b_panel(
    ax_c, G_cite, lb_positions, cite_sizes, cite_net, cite_ew,
    "#C87A4A", CITE_CMAP,
    f"{SECTOR['cpc_label']} — Citations\n{G_cite.number_of_edges()} edges, {cv:,.0f} citations")

mode_str = "cross-border only" if args.cross_border_only else "domestic + cross-border"
fig.suptitle(
    f"Level B: Industry × Country Networks — {SECTOR['hs_label']} ↔ {SECTOR['cpc_label']} ({YEAR})\n"
    f"Nodes = country×{SHORT} pairs  |  Top {len(top_countries)} countries  |  {mode_str}\n"
    f"Trade: arrow → importer  |  Citation: arrow → cited country",
    fontsize=11, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"../plots/level_b_{args.sector}_{YEAR}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  Plot saved → {out_path}")

print("\nDone! Level B edge CSVs and plot generated.")
print(f"\nNext steps:")
print(f"  1. Review the plot: {out_path}")
print(f"  2. Compare domestic vs cross-border patterns")
print(f"  3. Run with other sectors: --sector vehicles, --sector pharma")
print(f"  4. Run lead/lag at Level B: adapt lead_lag_analysis.py to use Level B edges")
