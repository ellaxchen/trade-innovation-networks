"""
Semiconductor case study: who leads in semiconductors?

Three measures per country-year:
  1. Knowledge rank: PageRank (alpha=0.85) on the H01L family-level citation
     network (edges citing->cited, cross-border) — "cited centrality": whose
     semiconductor prior art the world builds on.
  2. Trade rank: PageRank on the HS 8541+8542 export network with REVERSED
     edges (importer->exporter) — "export centrality": who the world buys
     chips from (weighted by buyers' importance). Export share reported too.
  3. Cite ratio: H01L citations received / citations made (cross-border).

Country definitions:
  auth      appln_auth (filing office, proxy)
  inventor  first inventor with a known country (tls206/tls207)

Windows:
  core     1995-2013 (last fully reliable citation year)
  extended 1995-2017 (robustness only; 2014-2017 undercounted)

The hypothesis: trade and knowledge rankings disagree sharply — assembly
countries (KR, TW, CN, MY, SG) dominate exports while design countries
(US, JP) dominate cited knowledge. The disagreement IS the result.

Outputs (suffix omitted for auth/core):
  ../processed/semiconductor_analysis[_suffix].csv   per country-year panel
  ../processed/semiconductor_spearman[_suffix].csv
  ../plots/semiconductor_rank_bump[_suffix].png
  ../plots/semiconductor_cite_ratio[_suffix].png

Usage:
  python semiconductor_analysis.py                         # auth, core
  python semiconductor_analysis.py --country inventor
  python semiconductor_analysis.py --end-year 2017         # robustness
"""
import argparse

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--country", choices=["auth", "inventor"], default="auth")
parser.add_argument("--end-year", type=int, choices=[2013, 2017], default=2013)
args = parser.parse_args()

END = args.end_year
YEARS = list(range(1995, END + 1))
ALPHA = 0.85
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
SUFFIX = ""
if args.country == "inventor":
    SUFFIX += "_inventor"
if END != 2013:
    SUFFIX += f"_{END}"
CITE_FILE = ("../processed/semiconductor_citations_family_by_year.csv"
             if args.country == "auth"
             else "../processed/semiconductor_citations_family_by_year_inventor.csv")
COUNTRY_LABEL = "filing office" if args.country == "auth" else "inventor country"

# Reference categorical palette, slots 1-8 (validated light mode)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#ffffff"
WATCH = {"KR", "TW"}

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Rep. of Korea": "South Korea"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new

print(f"=== Semiconductor analysis: {COUNTRY_LABEL}, 1995-{END} ===")

print("[1/3] Loading semiconductor edges...")
cite = pd.read_csv(CITE_FILE)
cite = cite[(cite["year"] >= 1995) & (cite["year"] <= END)]
cite = cite[~cite["citing_country"].isin(NON_COUNTRIES) &
            ~cite["cited_country"].isin(NON_COUNTRIES)]
cite = cite[cite["citing_country"] != cite["cited_country"]]

trade = pd.read_csv("../processed/semiconductor_trade_edges_by_year.csv")
trade = trade[trade["year"].between(1995, END)]
trade["source"] = trade["exporter"].map(baci_to_iso)
trade["target"] = trade["importer"].map(baci_to_iso)
trade = trade.dropna(subset=["source", "target"])
trade = trade[~trade["source"].isin(NON_COUNTRIES) &
              ~trade["target"].isin(NON_COUNTRIES)]
trade = trade[trade["source"] != trade["target"]]

print("[2/3] Computing per-year centrality, ranks, ratios...")
rows = []
for yr in YEARS:
    c_yr = cite[cite["year"] == yr]
    Gc = nx.DiGraph()
    Gc.add_weighted_edges_from(
        c_yr[["citing_country", "cited_country", "citation_count"]].itertuples(
            index=False, name=None))
    cite_pr = nx.pagerank(Gc, alpha=ALPHA, weight="weight") if Gc.number_of_nodes() else {}

    t_yr = trade[trade["year"] == yr]
    Gt = nx.DiGraph()
    Gt.add_weighted_edges_from(
        t_yr[["target", "source", "trade_value"]].itertuples(index=False, name=None))
    trade_pr = nx.pagerank(Gt, alpha=ALPHA, weight="weight") if Gt.number_of_nodes() else {}
    total_exp = t_yr.groupby("source")["trade_value"].sum()
    world = total_exp.sum()
    exp_share = (total_exp / world).to_dict() if world else {}

    received = c_yr.groupby("cited_country")["citation_count"].sum()
    made = c_yr.groupby("citing_country")["citation_count"].sum()
    ratio = (received / made).to_dict()

    pr_sorted = sorted(cite_pr.items(), key=lambda kv: -kv[1])
    krank = {c: i + 1 for i, (c, _) in enumerate(pr_sorted)}
    tr_sorted = sorted(trade_pr.items(), key=lambda kv: -kv[1])
    trank = {c: i + 1 for i, (c, _) in enumerate(tr_sorted)}

    countries = set(cite_pr) | set(trade_pr)
    for c in countries:
        rows.append({
            "year": yr, "country": c,
            "cite_pr": cite_pr.get(c, 0.0),
            "trade_export_pr": trade_pr.get(c, 0.0),
            "export_share": exp_share.get(c, 0.0),
            "cite_ratio": ratio.get(c, np.nan),
            "knowledge_rank": krank.get(c, np.nan),
            "trade_rank": trank.get(c, np.nan),
        })

panel = pd.DataFrame(rows)
panel.to_csv(f"../processed/semiconductor_analysis{SUFFIX}.csv", index=False)
print(f"  {len(panel):,} country-year rows -> "
      f"../processed/semiconductor_analysis{SUFFIX}.csv")

cite_vol = cite.groupby("citing_country")["citation_count"].sum().add(
    cite.groupby("cited_country")["citation_count"].sum(), fill_value=0)
trade_vol = trade.groupby("source")["trade_value"].sum().add(
    trade.groupby("target")["trade_value"].sum(), fill_value=0)
combined = cite_vol.add(trade_vol, fill_value=0).sort_values(ascending=False)
top12 = list(combined.index[:12])
for c in WATCH:
    if c not in top12 and c in combined.index:
        top12.append(c)
presence = {}
for c in combined.index:
    k_yrs = panel[(panel["country"] == c) & panel["knowledge_rank"].notna()]["year"].nunique()
    t_yrs = panel[(panel["country"] == c) & panel["trade_rank"].notna()]["year"].nunique()
    presence[c] = min(k_yrs, t_yrs)
qualified = [c for c in top12 if presence.get(c, 0) >= 12]
colored8 = [c for c in qualified if c not in WATCH][:6] + [c for c in WATCH if c in qualified]
rest = [c for c in top12 if c not in colored8]
print(f"\n  Colored (8): {colored8}")
print(f"  Gray rest:   {rest}")

rank_corr = []
from scipy.stats import spearmanr
for yr in YEARS:
    sub = panel[panel["year"] == yr].dropna(subset=["knowledge_rank", "trade_rank"])
    if len(sub) < 8:
        continue
    rho_full = spearmanr(sub["knowledge_rank"], sub["trade_rank"]).statistic
    top15 = set(sub.nsmallest(15, "knowledge_rank")["country"]) | \
        set(sub.nsmallest(15, "trade_rank")["country"])
    tier = sub[sub["country"].isin(top15)]
    rho_top = spearmanr(tier["knowledge_rank"], tier["trade_rank"]).statistic \
        if len(tier) >= 8 else np.nan
    rank_corr.append({"year": yr, "spearman_full": rho_full,
                      "spearman_top15": rho_top,
                      "n_full": len(sub), "n_top15": len(tier)})
rc_df = pd.DataFrame(rank_corr)
rc_df.to_csv(f"../processed/semiconductor_spearman{SUFFIX}.csv", index=False)
print("\n  Spearman (full sample vs top-15 tier):")
for _, r in rc_df.iterrows():
    print(f"    {int(r['year'])}: full={r['spearman_full']:+.3f} "
          f"(n={int(r['n_full'])}), top15={r['spearman_top15']:+.3f} "
          f"(n={int(r['n_top15'])})")

# ============================================================
# Figure 1: bump chart
# ============================================================
print("\n[3/3] Drawing bump chart...")
fig = plt.figure(figsize=(11.5, 6.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.42, wspace=0.10,
                      left=0.06, right=0.96, top=0.88, bottom=0.10)

panels = [("Knowledge rank", "H01L citation PageRank (cited centrality)", "knowledge_rank", True),
          ("Trade rank", "HS 8541+8542 export PageRank (export centrality)", "trade_rank", False)]

for i, (title, sub, col, is_first) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(SURFACE)
    ranks = {yr: {} for yr in YEARS}
    for _, r in panel.iterrows():
        ranks[r["year"]][r["country"]] = r[col]

    for c in rest:
        ys = [ranks[yr].get(c, np.nan) for yr in YEARS]
        ax.plot(YEARS, ys, color=GRAY, lw=1.0, alpha=0.55, zorder=1)
    for j, c in enumerate(colored8):
        ys = [ranks[yr].get(c, np.nan) for yr in YEARS]
        emph = c in WATCH
        lw = 3.0 if emph else 1.8
        ax.plot(YEARS, ys, color=PALETTE[j], lw=lw, zorder=3 if emph else 2,
                solid_capstyle="round")
        if not np.isnan(ys[-1]) and ys[-1] <= 15:
            ax.scatter([YEARS[-1]], [ys[-1]], s=34 if emph else 18,
                       color=PALETTE[j], zorder=4 if emph else 3,
                       linewidths=1.2, edgecolors=SURFACE)
            ax.annotate(f"{c}  {int(ys[-1])}", xy=(YEARS[-1], ys[-1]),
                        xytext=(END + 0.5, ys[-1]),
                        fontsize=9.5 if emph else 8.5,
                        fontweight="bold" if emph else "normal",
                        color=PALETTE[j], va="center", annotation_clip=False)
    for c in rest:
        ys = [ranks[yr].get(c, np.nan) for yr in YEARS]
        if not np.isnan(ys[-1]) and ys[-1] <= 15:
            ax.annotate(f"{c} {int(ys[-1])}", xy=(YEARS[-1], ys[-1]),
                        xytext=(END + 0.5, ys[-1]), fontsize=8, color=GRAY,
                        va="center", annotation_clip=False)

    ax.set_ylim(15.8, 0.2)
    ax.set_xlim(1994.5, END + 3.8)
    ax.set_yticks(range(1, 16, 2))
    ax.set_xticks([y for y in [1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
                   if y <= END])
    ax.tick_params(axis="both", labelsize=8, colors=INK2, length=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title(f"{title} — {sub}", fontsize=10, color=INK, pad=8, loc="left")
    if is_first:
        ax.set_ylabel("rank (1 = top)", fontsize=8, color=INK2)

ax = fig.add_subplot(gs[1, :])
ax.set_facecolor(SURFACE)
ax.plot(rc_df["year"], rc_df["spearman_full"], color=GRAY, lw=1.8, zorder=2)
ax.plot(rc_df["year"], rc_df["spearman_top15"], color=PALETTE[0], lw=2.4, zorder=3)
ax.scatter(rc_df["year"], rc_df["spearman_top15"], s=16, color=PALETTE[0],
           zorder=4, linewidths=0.8, edgecolors=SURFACE)
ax.axhline(0, color=BASE, lw=1.0)
ax.set_ylim(-0.05, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlim(1994.5, END + 4.0)
ax.set_xticks([y for y in [1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
               if y <= END])
ax.tick_params(axis="both", labelsize=8, colors=INK2, length=0)
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_title("Rank agreement — Spearman rho between trade and knowledge ranks, per year",
             fontsize=10, color=INK, pad=6, loc="left")
last = rc_df.iloc[-1]
ax.annotate(f"top-15 tier: rho = {last['spearman_top15']:+.2f}",
            xy=(END, last["spearman_top15"]),
            xytext=(END + 0.6, last["spearman_top15"] + 0.09),
            fontsize=8.5, color=PALETTE[0], annotation_clip=False)
ax.annotate(f"all countries: rho = {last['spearman_full']:+.2f}",
            xy=(END, last["spearman_full"]),
            xytext=(END + 0.6, last["spearman_full"] - 0.14),
            fontsize=8.5, color=GRAY, annotation_clip=False)
ax.plot([], [], color=GRAY, lw=1.8, label="all countries")
ax.plot([], [], color=PALETTE[0], lw=2.4, label="top-15 tier (union of both top-15s)")
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=1)

fig.suptitle(f"Who leads in semiconductors? Trade rank vs knowledge rank, "
             f"1995–{END} — {COUNTRY_LABEL} country definition",
             fontsize=12, fontweight="bold", color=INK, x=0.06, ha="left")
fig.text(0.06, 0.006,
         "Colored: top-8 by combined volume in both networks. Bold = Korea and Taiwan. "
         "Rank 1 = top; off-panel lines are ranks >15.\n"
         "Knowledge = H01L citation PageRank; trade = HS 8541+8542 export PageRank "
         "(reversed edges).",
         fontsize=7.6, color=INK2)
fig.savefig(f"../plots/semiconductor_rank_bump{SUFFIX}.png", dpi=200, facecolor=SURFACE)
print(f"  Saved -> ../plots/semiconductor_rank_bump{SUFFIX}.png")
plt.close(fig)

# ============================================================
# Figure 2: citations received / made ratio
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for j, c in enumerate(colored8):
    ys = []
    for yr in YEARS:
        sub = panel[(panel["year"] == yr) & (panel["country"] == c)]
        ys.append(sub["cite_ratio"].iloc[0]
                  if len(sub) and not np.isnan(sub["cite_ratio"].iloc[0]) else np.nan)
    emph = c in WATCH
    ax.plot(YEARS, ys, color=PALETTE[j], lw=3.0 if emph else 1.8,
            zorder=3 if emph else 2)
    if not np.isnan(ys[-1]):
        ax.annotate(f"{c}", xy=(END, ys[-1]), xytext=(END + 0.5, ys[-1]),
                    fontsize=9, fontweight="bold" if emph else "normal",
                    color=PALETTE[j], va="center", annotation_clip=False)
ax.set_yscale("log", base=2)
ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.axhline(1, color=BASE, lw=1.1)
ax.annotate("net knowledge source", xy=(1995.2, 1.15), fontsize=8, color=INK2)
ax.annotate("net knowledge absorber", xy=(1995.2, 0.87), fontsize=8, color=INK2)
ax.set_ylim(0.05, 10)
ax.set_xlim(1994.5, END + 3.5)
ax.set_xticks([y for y in [1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
               if y <= END])
ax.tick_params(axis="both", labelsize=8, colors=INK2, length=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_title("H01L citations received / made (cross-border), log2 scale",
             fontsize=10.5, color=INK, loc="left", pad=8)
fig.suptitle("Semiconductor knowledge balance", fontsize=12.5, fontweight="bold",
             color=INK, x=0.125, ha="left")
fig.text(0.125, 0.015,
         "Ratio > 1 = cited more than citing (net source of semiconductor knowledge). "
         f"Bold = Korea and Taiwan. Family-level, 1995–{END} ({COUNTRY_LABEL}).",
         fontsize=7.8, color=INK2)
fig.savefig(f"../plots/semiconductor_cite_ratio{SUFFIX}.png", dpi=200,
            facecolor=SURFACE, bbox_inches="tight")
print(f"  Saved -> ../plots/semiconductor_cite_ratio{SUFFIX}.png")
plt.close(fig)

# ============================================================
# Summary
# ============================================================
print(f"\n=== {END} snapshot ===")
snap = panel[panel["year"] == END].sort_values("knowledge_rank")
print(f"  {'Country':<10}{'know rank':>10}{'trade rank':>11}{'exp share':>10}{'cite ratio':>11}")
for _, r in snap.head(14).iterrows():
    if r["country"] in top12:
        print(f"  {r['country']:<10}{int(r['knowledge_rank']):>10}{int(r['trade_rank']):>11}"
              f"{r['export_share']*100:>9.1f}%{r['cite_ratio']:>11.2f}")

print("\n=== Watch list: Korea and Taiwan rank trajectories ===")
for c in sorted(WATCH):
    sub = panel[panel["country"] == c].sort_values("year")
    kr = " -> ".join(str(int(v)) for v in sub["knowledge_rank"].iloc[::3])
    tr = " -> ".join(str(int(v)) for v in sub["trade_rank"].iloc[::3])
    print(f"  {c}  knowledge rank (every 3y): {kr}")
    print(f"  {c}  trade rank     (every 3y): {tr}")

print("\nDone.")
