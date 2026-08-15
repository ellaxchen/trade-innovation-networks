"""
Compute PageRank centrality for each country/year in both networks.
Plot per-country panels with trade and citation PageRank series overlaid.

PageRank justification: Unlike eigenvector centrality, PageRank converges on directed
graphs that aren't strongly connected (common in trade/citation networks with dominant
hubs). The damping factor (α=0.85) ensures every node has a baseline score, preventing
dangling nodes from absorbing all centrality.

Usage:
  python centrality_analysis.py                        # app-level citations
  python centrality_analysis.py --family               # family-level (tls228)
  python centrality_analysis.py --top-n 15             # top 15 countries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--family", action="store_true",
                    help="Use family-level citation data (country_citation_family_by_year.csv)")
parser.add_argument("--top-n", type=int, default=16,
                    help="Number of top countries to plot")
parser.add_argument("--cross-border-only", action="store_true", default=True,
                    help="Exclude self-loops")
args = parser.parse_args()

ALPHA = 0.85  # PageRank damping factor
MAX_ITER = 200
TRADE_START, TRADE_END = 1995, 2024
CITE_START, CITE_END = 1995, 2013  # stop at 2013 (last reliable year)
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}

# ---- Load country codes ----
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
name_to_iso = dict(zip(cc["country_name"], cc["country_iso2"]))
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
iso_to_name["TW"] = "Taiwan"
# Clean names
CLEAN = {"Fed. Rep. of Germany (...1990)": "Germany",
         "Belgium-Luxembourg (...1998)": "Belgium",
         "Russian Federation": "Russia"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

# ============================================================
# 1. Load trade edges from pre-aggregated CSVs → PageRank per year
# ============================================================
print("[1/3] Computing trade PageRank (from pre-aggregated CSVs)...")
trade_pr = {}
for yr in range(TRADE_START, TRADE_END + 1):
    try:
        yr_df = pd.read_csv(f"../processed/trade_edges_{yr}.csv")
    except FileNotFoundError:
        trade_pr[yr] = {}
        continue
    # Map country names to ISO2 for consistent node IDs
    yr_df["source"] = yr_df["exporter_name"].map(name_to_iso)
    yr_df["target"] = yr_df["importer_name"].map(name_to_iso)
    yr_df = yr_df.dropna(subset=["source", "target"])
    yr_df = yr_df[~yr_df["source"].isin(NON_COUNTRIES) &
                  ~yr_df["target"].isin(NON_COUNTRIES)]
    if args.cross_border_only:
        yr_df = yr_df[yr_df["source"] != yr_df["target"]]

    G = nx.DiGraph()
    for _, r in yr_df.iterrows():
        w = float(r["trade_value"])
        if G.has_edge(r["source"], r["target"]):
            G[r["source"]][r["target"]]["weight"] += w
        else:
            G.add_edge(r["source"], r["target"], weight=w)

    if G.number_of_nodes() == 0:
        trade_pr[yr] = {}
        continue
    try:
        pr = nx.pagerank(G, alpha=ALPHA, max_iter=MAX_ITER, weight="weight")
        trade_pr[yr] = pr
    except Exception as e:
        print(f"  Trade PR failed for {yr}: {e}")
        trade_pr[yr] = {}
    if yr % 5 == 0:
        print(f"  {yr}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"top={max(pr, key=pr.get) if pr else '?'}")

# ============================================================
# 2. Load citation edges → PageRank per year
# ============================================================
cite_file = ("../processed/country_citation_family_by_year.csv"
             if args.family else
             "../processed/country_citation_edges_by_year.csv")
cite_label = "family" if args.family else "app"
print(f"\n[2/3] Computing citation PageRank ({cite_label}-level)...")

cite_all = pd.read_csv(cite_file)
cite_all = cite_all.rename(columns={
    "citing_country": "source", "cited_country": "target",
    "citation_count": "weight"
})
cite_all = cite_all[~cite_all["source"].isin(NON_COUNTRIES) &
                    ~cite_all["target"].isin(NON_COUNTRIES)]
cite_all = cite_all[cite_all["year"].between(CITE_START, CITE_END)]

if args.cross_border_only:
    cite_all = cite_all[cite_all["source"] != cite_all["target"]]

cite_pr = {}
for yr in range(CITE_START, CITE_END + 1):
    yr_df = cite_all[cite_all["year"] == yr]
    G = nx.DiGraph()
    for _, r in yr_df.iterrows():
        w = float(r["weight"])
        if G.has_edge(r["source"], r["target"]):
            G[r["source"]][r["target"]]["weight"] += w
        else:
            G.add_edge(r["source"], r["target"], weight=w)
    if G.number_of_nodes() == 0:
        cite_pr[yr] = {}
        continue
    try:
        pr = nx.pagerank(G, alpha=ALPHA, max_iter=MAX_ITER, weight="weight")
        cite_pr[yr] = pr
    except Exception as e:
        print(f"  Citation PageRank failed for {yr}: {e}")
        cite_pr[yr] = {}
    if yr % 3 == 0:
        print(f"  {yr}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"top={max(pr, key=pr.get) if pr else '?'}")

del cite_all

# ============================================================
# 3. Select top countries and build panel plot
# ============================================================
print(f"\n[3/3] Building per-country panels...")

# Select top countries by mean trade PageRank (1995-2024)
mean_trade = {}
for c in set().union(*(d.keys() for d in trade_pr.values())):
    vals = [trade_pr[yr].get(c, 0) for yr in range(TRADE_START, TRADE_END + 1)]
    mean_trade[c] = np.mean(vals) if vals else 0
top_countries = sorted(mean_trade, key=mean_trade.get, reverse=True)[:args.top_n]

print(f"  Top {args.top_n} countries: {', '.join(top_countries)}")

# Build plot
COLS = 4
ROWS = int(np.ceil(len(top_countries) / COLS))
fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.5, ROWS * 2.8))
axes = axes.flatten()

for idx, country in enumerate(top_countries):
    ax = axes[idx]
    name = iso_to_name.get(country, country)

    # Trade PR series
    t_years = list(range(TRADE_START, TRADE_END + 1))
    t_vals = [trade_pr[yr].get(country, 0) for yr in t_years]
    # Citation PR series
    c_years = list(range(CITE_START, CITE_END + 1))
    c_vals = [cite_pr[yr].get(country, 0) for yr in c_years]

    # Normalize each series to its own max for overlay comparison
    t_max = max(t_vals) if max(t_vals) > 0 else 1
    c_max = max(c_vals) if max(c_vals) > 0 else 1

    t_norm = [v / t_max for v in t_vals]
    c_norm = [v / c_max for v in c_vals]

    ax.plot(t_years, t_norm, color="#3A7CA5", linewidth=1.8, label="Trade PR")
    ax.plot(c_years, c_norm, color="#C87A4A", linewidth=1.8, label=f"Cite PR ({cite_label})")

    # Shade post-2013 citation gap
    ax.axvspan(2014, 2024, alpha=0.06, color="red")
    ax.axvline(x=2013.5, color="#aaa", linewidth=0.5, linestyle="--")

    # Find crossing point
    crossings = []
    for yr in c_years:
        tv = trade_pr[yr].get(country, 0) / t_max if t_max > 0 else 0
        cv = cite_pr[yr].get(country, 0) / c_max if c_max > 0 else 0
        crossings.append(tv - cv)

    # Mark where trade > cite vs cite > trade
    trade_lead_yrs = sum(1 for yr in c_years
                         if (trade_pr[yr].get(country, 0) / t_max) >
                            (cite_pr[yr].get(country, 0) / c_max))

    ax.set_title(f"{name} ({country})", fontsize=9, fontweight="bold")
    ax.set_xlim(1993, 2026)
    ax.set_ylim(-0.05, 1.12)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    if idx >= len(top_countries) - COLS:
        ax.set_xlabel("Year", fontsize=7)
    if idx % COLS == 0:
        ax.set_ylabel("PageRank (norm.)", fontsize=7)

    # Only legend on first panel
    if idx == 0:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

# Remove empty panels
for idx in range(len(top_countries), len(axes)):
    axes[idx].set_visible(False)

source_tag = "family-level" if args.family else "app-level"
fig.suptitle(
    f"PageRank Centrality: Trade vs. Patent Citations ({source_tag})\n"
    f"Each panel: blue = trade PageRank, orange = citation PageRank "
    f"(each normalized to its own max for comparison)\n"
    f"Citation data ends 2013 (red shading = no citation data). Cross-border only, α={ALPHA}.",
    fontsize=11, fontweight="bold", y=1.01
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"../plots/centrality_panels_{cite_label}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  Saved → {out_path}")

# ============================================================
# 4. Analysis: divergences and crossings
# ============================================================
print(f"\n=== Centrality Divergence Analysis ({cite_label}-level) ===")
print(f"Positive diff = trade PR (norm) > cite PR (norm); negative = cite leads\n")

for country in top_countries:
    name = iso_to_name.get(country, country)
    t_max = max(trade_pr[yr].get(country, 0) for yr in range(TRADE_START, TRADE_END + 1)) or 1
    c_max = max(cite_pr[yr].get(country, 0) for yr in range(CITE_START, CITE_END + 1)) or 1

    # Average diff 1995-2013
    diffs = []
    for yr in range(CITE_START, CITE_END + 1):
        tv = trade_pr[yr].get(country, 0) / t_max
        cv = cite_pr[yr].get(country, 0) / c_max
        diffs.append(tv - cv)
    avg_diff = np.mean(diffs)

    # Trend: is diff increasing or decreasing?
    if len(diffs) >= 10:
        early = np.mean(diffs[:5])
        late = np.mean(diffs[-5:])
        trend = late - early
    else:
        trend = 0

    # Latest values
    latest_trade = trade_pr[2013].get(country, 0) if 2013 in trade_pr else 0
    latest_cite = cite_pr[2013].get(country, 0) if 2013 in cite_pr else 0

    # Classification
    if avg_diff > 0.05:
        pathway = "Trade-leading"
    elif avg_diff < -0.05:
        pathway = "Knowledge-leading"
    else:
        pathway = "Balanced"

    trend_str = "trade gaining" if trend > 0.02 else ("cite gaining" if trend < -0.02 else "stable")
    print(f"  {country:3s} {name:<20s}  avg diff={avg_diff:+.3f}  "
          f"trend={trend:+.3f} ({trend_str:<15s})  pathway={pathway}")

print(f"\n=== Key observations ===")
print(f"  1. Compare this with lead_lag_analysis.py results (time-domain cross-correlation).")
print(f"  2. Countries where trade PR rises before cite PR → 'trade-first' pathway.")
print(f"  3. Countries where cite PR stays above trade PR → mature knowledge economies.")
print(f"  4. The normalized overlay reveals SHAPE divergence, not magnitude differences.")
if args.family:
    print(f"  5. Family-level data removes JP's multi-filing inflation — check if JP's")
    print(f"     cited centrality drops relative to app-level results.")
