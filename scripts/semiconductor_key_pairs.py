"""
Key bilateral pairs in semiconductors: trade dependence vs knowledge dependence.

For each ordered pair A -> B:
  trade share     = A's semiconductor exports to B / A's total semiconductor exports
  citation share  = A's H01L citations to B / A's total H01L citations

Panels: four pairs (US-CN, US-TW, US-KR, JP-KR) x two networks
(trade, citations), both directions in each panel.

Output: ../plots/semiconductor_key_pairs[_inventor].png
        ../processed/semiconductor_key_pairs[_inventor].csv

Usage: python semiconductor_key_pairs.py [--country auth|inventor] [--end-year 2013]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--country", choices=["auth", "inventor"], default="auth")
parser.add_argument("--end-year", type=int, default=2013)
args = parser.parse_args()

END = args.end_year
YEARS = list(range(1995, END + 1))
SUFFIX = "" if args.country == "auth" else "_inventor"
if END != 2013:
    SUFFIX += f"_{END}"
CITE_FILE = ("../processed/semiconductor_citations_family_by_year.csv"
             if args.country == "auth"
             else "../processed/semiconductor_citations_family_by_year_inventor.csv")
COUNTRY_LABEL = "filing office" if args.country == "auth" else "inventor country"

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
PAIRS = [("US", "CN"), ("US", "TW"), ("US", "KR"), ("JP", "KR")]
NAMES = {"US": "United States", "CN": "China", "TW": "Taiwan",
         "KR": "South Korea", "JP": "Japan"}

SURFACE = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
C_AB = "#2a78d6"; C_BA = "#eb6834"

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"

trade = pd.read_csv("../processed/semiconductor_trade_edges_by_year.csv")
trade["source"] = trade["exporter"].map(baci_to_iso)
trade["target"] = trade["importer"].map(baci_to_iso)
trade = trade.dropna(subset=["source", "target"])
trade = trade[~trade["source"].isin(NON_COUNTRIES) &
              ~trade["target"].isin(NON_COUNTRIES)]
trade = trade[trade["source"] != trade["target"]]
trade = trade[trade["year"].between(1995, END)]

cite = pd.read_csv(CITE_FILE)
cite = cite[~cite["citing_country"].isin(NON_COUNTRIES) &
            ~cite["cited_country"].isin(NON_COUNTRIES)]
cite = cite[cite["citing_country"] != cite["cited_country"]]
cite = cite[cite["year"].between(1995, END)]


def series(df, src, dst, wcol, a, b, years):
    """Share of A's total flow that goes to B, by year."""
    vals = []
    for yr in years:
        d = df[df["year"] == yr]
        total = d.loc[d[src] == a, wcol].sum()
        pair = d.loc[(d[src] == a) & (d[dst] == b), wcol].sum()
        vals.append(100.0 * pair / total if total > 0 else np.nan)
    return vals


rows = []
fig, axes = plt.subplots(4, 2, figsize=(10.5, 11.5), facecolor=SURFACE,
                         sharex=True)
for i, (a, b) in enumerate(PAIRS):
    ta = series(trade, "source", "target", "trade_value", a, b, YEARS)
    tb = series(trade, "source", "target", "trade_value", b, a, YEARS)
    ca = series(cite, "citing_country", "cited_country", "citation_count", a, b, YEARS)
    cb = series(cite, "citing_country", "cited_country", "citation_count", b, a, YEARS)

    for j, (ys_a, ys_b, title, fmt) in enumerate([
            (ta, tb, "Trade: share of sender's exports", "trade"),
            (ca, cb, "Knowledge: share of sender's citations", "cite")]):
        ax = axes[i, j]
        ax.set_facecolor(SURFACE)
        ax.plot(YEARS, ys_a, color=C_AB, lw=2.2, label=f"{a} → {b}")
        ax.plot(YEARS, ys_b, color=C_BA, lw=2.2, label=f"{b} → {a}")
        ax.set_xlim(1994.5, END + 2.5)
        ax.set_xticks([y for y in [1995, 2000, 2005, 2010, 2013] if y <= END])
        ax.tick_params(axis="both", labelsize=7.5, colors=INK2, length=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_title(f"{a}–{b}: {title}", fontsize=8.5, color=INK, loc="left")
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
        # label the final value
        for ys, color, lab in [(ys_a, C_AB, a), (ys_b, C_BA, b)]:
            if len(ys) and not np.isnan(ys[-1]):
                ax.annotate(f"{ys[-1]:.1f}%", xy=(YEARS[-1], ys[-1]),
                            xytext=(END + 0.4, ys[-1]), fontsize=7,
                            color=color, va="center", annotation_clip=False)
        for yr, ya, yb in zip(YEARS, ys_a, ys_b):
            rows.append({"pair": f"{a}-{b}", "network": fmt, "year": yr,
                         "a_to_b_share": ya, "b_to_a_share": yb})

for i in range(4):
    axes[i, 0].set_ylabel("percent", fontsize=7.5, color=INK2)
for j in range(2):
    axes[-1, j].set_xlabel("year", fontsize=7.5, color=INK2)

fig.suptitle(
    "Key semiconductor pairs: trade dependence and knowledge dependence\n"
    f"{COUNTRY_LABEL} country definition, 1995–{END}",
    fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
fig.text(0.02, 0.008,
         "Trade share = bilateral exports / sender's total semiconductor exports. "
         "Citation share = bilateral H01L citations / sender's total H01L citations.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
out_png = f"../plots/semiconductor_key_pairs{SUFFIX}.png"
fig.savefig(out_png, dpi=200, facecolor=SURFACE)
plt.close(fig)
pd.DataFrame(rows).to_csv(f"../processed/semiconductor_key_pairs{SUFFIX}.csv",
                          index=False)
print(f"Saved -> {out_png}")
for pair in ["US-CN", "US-TW", "US-KR", "JP-KR"]:
    sub = pd.DataFrame(rows)
    sub = sub[(sub["pair"] == pair) & (sub["year"] == END)]
    print(f"  {pair} at {END}:")
    for _, r in sub.iterrows():
        print(f"    {r['network']:<6} {r['a_to_b_share']:>6.1f}% / "
              f"{r['b_to_a_share']:>6.1f}%")
print("Done.")
