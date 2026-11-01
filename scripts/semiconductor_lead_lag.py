"""
Semiconductor lead-lag: trade export centrality vs H01L citation centrality.

Mirrors the project's changes-based methodology (lead_lag_changes.py +
changes_bootstrap.py), applied to the semiconductor cut:
  trade    = HS 8541+8542, PageRank on REVERSED edges (importer->exporter)
             = export centrality (the case study's trade measure)
  knowledge= CPC H01L family citations, PageRank citing->cited

Convention: positive lag = trade leads (corr(trade[t], cite[t+lag])).
Levels are reported for completeness but are trend-dominated; the changes
columns are the informative ones.

Windows:
  core     1995-2013  (last fully reliable citation year)
  extended 1995-2017  (robustness; 2014-2017 undercounted)

Country definitions:
  auth      appln_auth (filing office, proxy)
  inventor  first inventor with a known country (tls206/tls207)

Outputs:
  ../processed/semiconductor_lead_lag_summary.csv
  ../processed/semiconductor_lead_lag_bootstrap.csv
  ../plots/semiconductor_lead_lag_correlogram_auth.png
  ../plots/semiconductor_lead_lag_correlogram_inventor.png

Usage: python semiconductor_lead_lag.py [--draws 1000]
"""
import argparse
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--block", type=int, default=3)
args = parser.parse_args()

LAGS = list(range(-5, 6))
ALPHA = 0.85
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU"}
WINDOWS = {"core": list(range(1995, 2014)), "extended": list(range(1995, 2018))}
VERSIONS = {
    "auth": "../processed/semiconductor_citations_family_by_year.csv",
    "inventor": "../processed/semiconductor_citations_family_by_year_inventor.csv",
}
# allow an auth-only run while the inventor edge file is still being built
VERSIONS = {k: v for k, v in VERSIONS.items() if os.path.exists(v)}
missing = [k for k in ("auth", "inventor") if k not in VERSIONS]
if missing:
    print(f"NOTE: skipping missing version(s): {', '.join(missing)}")
WATCH = ["US", "CN", "JP", "KR", "TW", "DE"]
rng = np.random.default_rng(42)

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"
iso_to_name = dict(zip(cc["country_iso2"], cc["country_name"]))
CLEAN = {"Rep. of Korea": "South Korea", "Russian Federation": "Russia",
         "China, Hong Kong SAR": "Hong Kong",
         "Fed. Rep. of Germany (...1990)": "Germany"}
for old, new in CLEAN.items():
    for k, v in iso_to_name.items():
        if v == old:
            iso_to_name[k] = new
iso_to_name["TW"] = "Taiwan"

# ---------------------------------------------------------------
# Trade edges (once)
# ---------------------------------------------------------------
print("[1/4] Loading semiconductor trade edges...")
trade_raw = pd.read_csv("../processed/semiconductor_trade_edges_by_year.csv")
trade_raw["source"] = trade_raw["exporter"].map(baci_to_iso)
trade_raw["target"] = trade_raw["importer"].map(baci_to_iso)
trade_raw = trade_raw.dropna(subset=["source", "target"])
trade_raw = trade_raw[~trade_raw["source"].isin(NON_COUNTRIES) &
                      ~trade_raw["target"].isin(NON_COUNTRIES)]
trade_raw = trade_raw[trade_raw["source"] != trade_raw["target"]]
print(f"  {len(trade_raw):,} mapped flows, years "
      f"{int(trade_raw.year.min())}-{int(trade_raw.year.max())}")


def pagerank_by_year(df, weight_col, src_col, dst_col, years):
    out = {}
    for yr in years:
        sub = df[df["year"] == yr]
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            sub[[src_col, dst_col, weight_col]].itertuples(index=False, name=None))
        out[yr] = nx.pagerank(G, alpha=ALPHA, weight="weight") if G.number_of_nodes() else {}
    return out


def best_lag(ts, cs, changes=False):
    if changes:
        ts, cs = np.diff(ts), np.diff(cs)
    best, best_r = 0, np.nan
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(best_r) or abs(r) > abs(best_r):
                best, best_r = k, r
    return best, best_r


def correlogram(ts, cs, changes=True):
    if changes:
        ts, cs = np.diff(ts), np.diff(cs)
    out = {}
    for k in LAGS:
        if k > 0:
            x, y = np.array(ts[:-k]), np.array(cs[k:])
        elif k < 0:
            x, y = np.array(ts[-k:]), np.array(cs[:k])
        else:
            x, y = np.array(ts), np.array(cs)
        if len(x) >= 10 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def moving_block_sample(n, b):
    idx = []
    while len(idx) < n:
        start = rng.integers(0, n - b + 1)
        idx.extend(range(start, start + b))
    return np.array(idx[:n])


def bootstrap_series(dts, dcs, sample_lag, sample_r, draws, block):
    n = len(dts)
    lag_counts = {k: 0 for k in LAGS}
    r_at_sample = []
    for _ in range(draws):
        yi = moving_block_sample(n, block)
        bts, bcs = dts[yi], dcs[yi]
        bl, _ = best_lag(bts, bcs, changes=False)  # already differenced
        lag_counts[bl] += 1
        k = sample_lag
        if k > 0:
            x, y = bts[:-k], bcs[k:]
        elif k < 0:
            x, y = bts[-k:], bcs[:k]
        else:
            x, y = bts, bcs
        if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
            r_at_sample.append(np.corrcoef(x, y)[0, 1])
    r_at_sample = np.array(r_at_sample)
    lo, hi = np.percentile(r_at_sample, [2.5, 97.5])
    top3 = sorted(lag_counts.items(), key=lambda kv: -kv[1])[:3]
    p_zero = (float(np.mean(r_at_sample <= 0)) if sample_r > 0
              else float(np.mean(r_at_sample >= 0)))
    return {
        "lag_share": lag_counts[sample_lag] / draws,
        "zero_share": lag_counts[0] / draws,
        "r_lo": lo, "r_hi": hi, "p_signflip": p_zero,
        "top3": ", ".join(f"{k}({v/draws:.0%})" for k, v in top3),
    }


# ---------------------------------------------------------------
# Per version / window
# ---------------------------------------------------------------
summary_rows, boot_rows, corr_data = [], [], {}

for version, path in VERSIONS.items():
    print(f"\n[2/4] Version: {version} ({path})")
    cite = pd.read_csv(path)
    cite = cite[~cite["citing_country"].isin(NON_COUNTRIES) &
                ~cite["cited_country"].isin(NON_COUNTRIES)]
    cite = cite[cite["citing_country"] != cite["cited_country"]]

    for window, years in WINDOWS.items():
        print(f"  window {window}: {years[0]}-{years[-1]}")
        t = trade_raw[trade_raw["year"].isin(years)]
        c = cite[cite["year"].isin(years)]

        # export centrality: reversed edges importer -> exporter
        trade_pr = pagerank_by_year(t, "trade_value", "target", "source", years)
        cite_pr = pagerank_by_year(c, "citation_count",
                                   "citing_country", "cited_country", years)

        trade_vol = t.groupby("source")["trade_value"].sum().add(
            t.groupby("target")["trade_value"].sum(), fill_value=0)
        cite_vol = c.groupby("citing_country")["citation_count"].sum().add(
            c.groupby("cited_country")["citation_count"].sum(), fill_value=0)
        combined = trade_vol.add(cite_vol, fill_value=0)

        eligible = []
        for country in combined.sort_values(ascending=False).index:
            t_yrs = sum(1 for yr in years if country in trade_pr.get(yr, {}))
            c_yrs = sum(1 for yr in years if country in cite_pr.get(yr, {}))
            if t_yrs >= 5 and c_yrs >= 5:
                eligible.append(country)
            if len(eligible) >= 12:
                break

        print(f"    countries: {', '.join(eligible)}")
        for country in eligible:
            ts = np.array([trade_pr[yr].get(country, 0) for yr in years])
            cs = np.array([cite_pr[yr].get(country, 0) for yr in years])
            if np.std(ts) < 1e-12 or np.std(cs) < 1e-12:
                continue
            ll, lr = best_lag(ts, cs, changes=False)
            dl, dr = best_lag(ts, cs, changes=True)
            dts, dcs = np.diff(ts), np.diff(cs)
            r0 = correlogram(ts, cs, changes=True).get(0, np.nan)
            name = iso_to_name.get(country, country)
            summary_rows.append({
                "version": version, "window": window,
                "country": country, "name": name,
                "level_lag": ll, "level_r": round(lr, 3),
                "change_lag": dl, "change_r": round(dr, 3),
                "change_r_lag0": round(r0, 3),
                "n_years": len(years),
            })
            b = bootstrap_series(dts, dcs, dl, dr, args.draws, args.block)
            boot_rows.append({
                "version": version, "window": window,
                "country": country, "name": name,
                "sample_lag": dl, "sample_r": round(dr, 3),
                "lag_share": round(b["lag_share"], 3),
                "zero_share": round(b["zero_share"], 3),
                "r_lo": round(float(b["r_lo"]), 3),
                "r_hi": round(float(b["r_hi"]), 3),
                "p_signflip": round(b["p_signflip"], 3),
                "top3": b["top3"],
            })

        # store correlograms for the core-window figure
        if window == "core":
            corr_data[version] = {}
            for country in eligible:
                ts = [trade_pr[yr].get(country, 0) for yr in years]
                cs = [cite_pr[yr].get(country, 0) for yr in years]
                if np.std(ts) > 1e-12 and np.std(cs) > 1e-12:
                    corr_data[version][country] = correlogram(ts, cs, changes=True)

summary = pd.DataFrame(summary_rows)
boot = pd.DataFrame(boot_rows)
summary.to_csv("../processed/semiconductor_lead_lag_summary.csv", index=False)
boot.to_csv("../processed/semiconductor_lead_lag_bootstrap.csv", index=False)
print(f"\n[3/4] Saved {len(summary)} summary rows -> "
      f"../processed/semiconductor_lead_lag_summary.csv")
print(f"      Saved {len(boot)} bootstrap rows -> "
      f"../processed/semiconductor_lead_lag_bootstrap.csv")

# ---------------------------------------------------------------
# Figure: changes correlograms, core window
# ---------------------------------------------------------------
print("[4/4] Drawing correlograms...")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300"]
SURFACE = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"
GRID = "#e1e0d9"; BASE = "#c3c2b7"

for version, cmap in corr_data.items():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for ax, country in zip(axes.ravel(), WATCH):
        ax.set_facecolor(SURFACE)
        prof = cmap.get(country)
        if prof:
            ks = sorted(prof)
            ax.plot(ks, [prof[k] for k in ks], color=PALETTE[0], lw=2.2,
                    marker="o", ms=3.5)
        ax.axhline(0, color=BASE, lw=0.9)
        ax.axvline(0, color=GRID, lw=1.0)
        ax.set_title(f"{country} — {iso_to_name.get(country, country)}",
                     fontsize=9, color=INK, loc="left")
        ax.set_xticks([-5, -3, -1, 1, 3, 5])
        ax.set_xlim(-5.4, 5.4)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(labelsize=7, colors=INK2, length=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(BASE); ax.spines["bottom"].set_color(BASE)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("r (year-over-year changes)", fontsize=8, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel("lag (years) — positive = trade leads", fontsize=8, color=INK2)
    fig.suptitle(f"Semiconductor lead-lag correlograms — {version} country definition",
                 fontsize=11.5, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01,
             "Year-over-year changes, 1995–2013. Levels are trend-dominated "
             "(r ≈ 0.99 at every lag); changes are the informative diagnostic.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = f"../plots/semiconductor_lead_lag_correlogram_{version}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved -> {out}")

print("\nDone.")
