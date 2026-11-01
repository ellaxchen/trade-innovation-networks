"""
Compare the filing-office and inventor-country semiconductor panels.

Answers: does the country-assignment choice change the knowledge hierarchy?

Inputs:
  ../processed/semiconductor_analysis.csv           (filing office, 1995-2013)
  ../processed/semiconductor_analysis_inventor.csv  (inventor country, 1995-2013)

Outputs:
  ../processed/semiconductor_country_definition_comparison.csv
  ../plots/semiconductor_knowledge_rank_auth_vs_inventor.png

Usage: python compare_country_definitions.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SURFACE = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAJORS = ["US", "JP", "CN", "KR", "TW", "DE"]

auth = pd.read_csv("../processed/semiconductor_analysis.csv")
inv = pd.read_csv("../processed/semiconductor_analysis_inventor.csv")

rows = []
for yr in sorted(set(auth["year"]) & set(inv["year"])):
    a = auth[auth["year"] == yr][["country", "knowledge_rank"]]
    b = inv[inv["year"] == yr][["country", "knowledge_rank"]]
    m = a.merge(b, on="country", suffixes=("_auth", "_inv")).dropna()
    if len(m) < 8:
        continue
    rho = spearmanr(m["knowledge_rank_auth"], m["knowledge_rank_inv"]).statistic
    top10_a = set(a.nsmallest(10, "knowledge_rank")["country"])
    top10_b = set(b.nsmallest(10, "knowledge_rank")["country"])
    rows.append({
        "year": yr,
        "spearman_knowledge": round(float(rho), 3),
        "top10_overlap": len(top10_a & top10_b),
        "n_common": len(m),
    })

cmp = pd.DataFrame(rows)
cmp.to_csv("../processed/semiconductor_country_definition_comparison.csv", index=False)

print("=== Knowledge-rank agreement between country definitions ===")
print(cmp.to_string(index=False))
print(f"\n  Spearman: min {cmp.spearman_knowledge.min():.3f}, "
      f"max {cmp.spearman_knowledge.max():.3f}, "
      f"mean {cmp.spearman_knowledge.mean():.3f}")
print(f"  Top-10 overlap: {cmp.top10_overlap.min()}-{cmp.top10_overlap.max()} of 10")

print("\n=== 2013 knowledge order ===")
for label, df in [("filing office", auth), ("inventor", inv)]:
    s = df[df["year"] == 2013].nsmallest(10, "knowledge_rank")
    print(f"  {label}: " + ", ".join(
        f"{r.country}({int(r.knowledge_rank)})" for _, r in s.iterrows()))

# ---- figure: knowledge rank trajectories, both definitions ----
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), facecolor=SURFACE,
                         sharey=True)
for ax, c in zip(axes.ravel(), MAJORS):
    ax.set_facecolor(SURFACE)
    for df, color, lab, ls in [
            (auth, PALETTE[0], "filing office", "-"),
            (inv, PALETTE[1], "inventor country", "--")]:
        sub = df[df["country"] == c].sort_values("year")
        ax.plot(sub["year"], sub["knowledge_rank"], color=color, lw=2.0,
                linestyle=ls, label=lab)
    ax.set_ylim(16, 0.5)
    ax.set_xlim(1994.5, 2013.5)
    ax.set_xticks([1995, 2000, 2005, 2010, 2013])
    ax.set_title(c, fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=INK2, length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for ax in axes[:, 0]:
    ax.set_ylabel("knowledge rank (1 = top)", fontsize=8, color=INK2)
axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Country definition check: knowledge rank, filing office vs inventor country",
             fontsize=11.5, fontweight="bold", color=INK, x=0.02, ha="left")
fig.text(0.02, 0.01,
         "1995–2013, semiconductor cut (H01L). Inventor country = first inventor "
         "with a known country; coverage is limited by PATSTAT person records.",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig("../plots/semiconductor_knowledge_rank_auth_vs_inventor.png",
            dpi=200, facecolor=SURFACE)
print("\nSaved -> ../processed/semiconductor_country_definition_comparison.csv")
print("Saved -> ../plots/semiconductor_knowledge_rank_auth_vs_inventor.png")
