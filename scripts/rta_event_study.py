"""
Staggered event study: do knowledge flows (family-level citations) rise after
two countries enter a regional trade agreement (RTA)?

Design (descriptive, audited style — honest about limits):
  1. Detect RTA entry events from the BACI rta column (pair-year 0→1
     transitions), event years 1998-2008 so each event has >=3 pre and
     >=5 post years inside the citation window.
  2. Outcome: growth of directed citation flows on the pair,
     pre (t-3..t-1) vs post (t+1..t+5), event year dropped.
  3. Controls: pairs that never enter an RTA in the window, matched on
     pre-period citation volume (tercile matching).
  4. Report event-pair growth vs matched-control growth, with t-tests,
     plus the China-RTA sub-events separately (with the coverage caveat).

Usage: python rta_event_study.py
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from collections import defaultdict

NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU", "EA"}
CN_CODE = 156

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"

# ============================================================
# 1. Pair-year RTA matrix from the BACI panel
# ============================================================
print("[1/3] Extracting pair-year RTA status...")
rta = {}          # (year, exporter, importer) -> max rta (0 default included)
ever_rta = set()  # pairs with rta=1 in some year
all_pairs = set()
pf = pq.ParquetFile("../rawdata/baci_panel_hs6.parquet")
for batch in pf.iter_batches(batch_size=2_000_000,
                             columns=["year", "exporter", "importer", "rta"]):
    b = batch.to_pandas()
    b = b[b["year"].between(1995, 2013)]
    if len(b) == 0:
        continue
    b["rta"] = b["rta"].fillna(0)
    g = b.groupby(["year", "exporter", "importer"])["rta"].max()
    for (yr, e, i), v in g.items():
        key = (int(yr), int(e), int(i))
        vv = 0 if pd.isna(v) else int(v)
        rta[key] = max(rta.get(key, 0), vv)   # ALWAYS set (keep 0s too)
        all_pairs.add((e, i))
        if vv > 0:
            ever_rta.add((e, i))

# Convert to pair -> year series
pair_rta = defaultdict(dict)
for (yr, e, i), v in rta.items():
    pair_rta[(e, i)][yr] = int(v)
del rta
print(f"  {len(all_pairs):,} directed pairs total, {len(ever_rta):,} ever in an RTA")

# ============================================================
# 2. Detect RTA entry events
# ============================================================
print("[2/3] Detecting RTA entry events...")
events = []  # (exporter, importer, entry_year)
for (e, i), series in pair_rta.items():
    yrs = sorted(series)
    if len(yrs) < 6:
        continue
    # entry: first year with rta=1 preceded by >=2 zeros
    for t in range(1998, 2009):
        if t not in series:
            continue
        if series[t] == 1 and series.get(t - 1, 0) == 0 and series.get(t - 2, 0) == 0:
            # require post-period observations exist
            if any(y > t for y in yrs):
                events.append((e, i, t))
            break  # one event per directed pair

print(f"  {len(events)} RTA entry events (1998-2008)")

# ============================================================
# 3. Citation outcome + matched controls
# ============================================================
print("[3/3] Computing knowledge-flow growth and controls...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[cite.year.between(1995, 2013)]
cite = cite[cite.citing_country != cite.cited_country]

def pair_citations(iso_i, iso_j, years):
    """Sum of citations i -> j in given years."""
    return cite[(cite.citing_country == iso_i) & (cite.cited_country == iso_j) &
                (cite.year.isin(years))]["citation_count"].sum()

def growth_for(iso_i, iso_j, t):
    pre = pair_citations(iso_i, iso_j, range(t - 3, t))
    post = pair_citations(iso_i, iso_j, range(t + 1, t + 6))
    return np.log1p(post) - np.log1p(pre), pre

event_rows = []
for (e, i, t) in events:
    iso_e = baci_to_iso.get(e)
    iso_i = baci_to_iso.get(i)
    if not iso_e or not iso_i or iso_e in NON_COUNTRIES or iso_i in NON_COUNTRIES:
        continue
    if iso_e == iso_i:
        continue
    g, pre = growth_for(iso_e, iso_i, t)
    event_rows.append({"iso_e": iso_e, "iso_i": iso_i, "t": t,
                       "growth": g, "pre_vol": pre})

ev = pd.DataFrame(event_rows)
print(f"  {len(ev)} usable events (ISO-mapped, non-country)")

# Controls: never-RTA pairs = all pairs minus ever-RTA pairs
never_rta = all_pairs - ever_rta
print(f"  {len(never_rta):,} never-RTA pairs available as controls")

# Vectorized citation pair lookup: (i, j) -> (pre_sum 1999-2001, post_sum 2003-2007)
cite_pre = cite[cite.year.between(1999, 2001)].groupby(
    ["citing_country", "cited_country"])["citation_count"].sum()
cite_post = cite[cite.year.between(2003, 2007)].groupby(
    ["citing_country", "cited_country"])["citation_count"].sum()
cite_lookup = {}
for (i, j) in set(cite_pre.index) | set(cite_post.index):
    cite_lookup[(i, j)] = (int(cite_pre.get((i, j), 0)), int(cite_post.get((i, j), 0)))

def lookup_growth(i, j):
    pre, post = cite_lookup.get((i, j), (0, 0))
    return np.log1p(post) - np.log1p(pre), pre

# sample controls: never-RTA pairs with pre-period flow
control_rows = []
for (e, i) in never_rta:
    iso_e = baci_to_iso.get(e)
    iso_i = baci_to_iso.get(i)
    if not iso_e or not iso_i or iso_e in NON_COUNTRIES or iso_i in NON_COUNTRIES:
        continue
    if iso_e == iso_i:
        continue
    g, pre = lookup_growth(iso_e, iso_i)
    if pre > 0:  # require some pre-period flow
        control_rows.append({"iso_e": iso_e, "iso_i": iso_i, "growth": g, "pre_vol": pre})

ctrl = pd.DataFrame(control_rows)
print(f"  {len(ctrl):,} control pairs (never-RTA, with pre-flow)")

# tercile matching on pre volume
if len(ev) and len(ctrl):
    terciles = np.quantile(ev["pre_vol"], [1/3, 2/3])
    def terc(v):
        return 0 if v <= terciles[0] else (1 if v <= terciles[1] else 2)
    ev = ev.assign(terc=ev["pre_vol"].apply(terc))
    ctrl = ctrl.assign(terc=ctrl["pre_vol"].apply(terc))
    matched = []
    for t_ in range(3):
        ctrl_t = ctrl[ctrl.terc == t_]
        if len(ctrl_t) == 0:
            continue
        # sample with replacement to match event counts
        n_ev = len(ev[ev.terc == t_])
        if n_ev:
            matched.append(ctrl_t.sample(n=n_ev * 3, replace=True, random_state=42))
    matched_ctrl = pd.concat(matched)
else:
    matched_ctrl = ctrl

print("\n=== Event study: knowledge flows around RTA entry ===")
print(f"  Event pairs: n={len(ev)}, growth {ev['growth'].mean():+.3f} "
      f"(median {ev['growth'].median():+.3f})")
print(f"  Matched controls: n={len(matched_ctrl)}, growth {matched_ctrl['growth'].mean():+.3f} "
      f"(median {matched_ctrl['growth'].median():+.3f})")
if len(ev) and len(matched_ctrl):
    t_stat, p_val = stats.ttest_ind(ev["growth"], matched_ctrl["growth"])
    print(f"  t-test: t={t_stat:+.2f}, p={p_val:.3f}")
    print(f"  DiD (event - control): {ev['growth'].mean() - matched_ctrl['growth'].mean():+.3f}")

# China-specific events
cn_events = ev[(ev.iso_e == "CN") | (ev.iso_i == "CN")]
print(f"\n=== China RTA events (post-WTO) ===")
print(f"  n={len(cn_events)}")
for _, r in cn_events.iterrows():
    print(f"    {r['iso_e']} -> {r['iso_i']} entered {r['t']}: growth {r['growth']:+.3f}")
if len(cn_events):
    print(f"  mean growth {cn_events['growth'].mean():+.3f}")
    print("  CAVEAT: post-2005 outcomes sit in the family-level coverage ramp;")
    print("  treat China-specific event estimates as descriptive upper bounds.")

ev.to_csv("../processed/rta_events.csv", index=False)
print("\nSaved → ../processed/rta_events.csv")
