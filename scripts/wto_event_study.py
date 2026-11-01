"""
Event study: China's WTO accession (Dec 2001) as a trade-liberalization shock.
REVISED after independent audit (2026-08-15). Changes vs v1:
  - Dose-response on tariff cuts DROPPED as primary evidence: the BACI tariff
    columns have no post-2003 coverage for China-as-importer, and zero coverage
    for US/EU/India/Brazil as China's partners; measured cuts fail the trade
    validation (r = +0.086). Documented as a negative result instead.
  - DiD rebuilt: pair-level growth, control = same partners' flows with other
    major hubs (US/JP/DE), self-citations excluded, CN self-pair excluded,
    same pre/post windows for both sides.
  - Timing caveats documented (US PNTR changed applied rates ~not at all;
    China's cuts pre-announced; shock = uncertainty removal + phase-in).

Usage: python wto_event_study.py
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

CN_CODE = 156
NON_COUNTRIES = {"EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU", "EA"}
HUBS = ["US", "JP", "DE"]  # control destinations for the partner-based DiD

cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_iso = dict(zip(cc["country_code"], cc["country_iso2"]))
baci_to_iso[490] = "TW"

# ============================================================
# 1. Document the tariff-data failure (audit findings)
# ============================================================
print("[1/3] Tariff data coverage check (the negative result)...")
pf = pq.ParquetFile("../rawdata/baci_panel_hs6.parquet")
TARIFF_COLS = ["year", "exporter", "importer", "trade_value",
               "mfn_tariff_wavg", "pref_tariff_wavg"]
n_mfn = {}
n_pref = {}
for batch in pf.iter_batches(batch_size=2_000_000, columns=TARIFF_COLS):
    b = batch.to_pandas()
    b = b[(b["exporter"] == CN_CODE) | (b["importer"] == CN_CODE)]
    b = b[b["year"].between(1999, 2005)]
    if len(b) == 0:
        continue
    for yr in range(1999, 2006):
        m = b[(b.year == yr) & (b.importer == CN_CODE)]
        e = b[(b.year == yr) & (b.exporter == CN_CODE)]
        n_mfn[yr] = n_mfn.get(yr, 0) + int(m["mfn_tariff_wavg"].notna().sum())
        n_pref[yr] = n_pref.get(yr, 0) + int(m["pref_tariff_wavg"].notna().sum())

print("  China-as-IMPORTER tariff coverage (mfn non-null rows, 1999-2005):")
for yr in range(1999, 2006):
    print(f"    {yr}: mfn={n_mfn.get(yr, 0):>8,}  pref={n_pref.get(yr, 0):>8,}")
print("  → post-2003 coverage is essentially zero; the 'in' dose-response")
print("    cannot be measured with this data (audit finding #1).")

# ============================================================
# 2. Corrected DiD: pair-level growth, hub-based control
# ============================================================
print("\n[2/3] Corrected diff-in-diff...")
cite = pd.read_csv("../processed/country_citation_family_by_year.csv")
cite = cite[cite.year.between(1995, 2013)]
cite = cite[~cite.citing_country.isin(NON_COUNTRIES) &
            ~cite.cited_country.isin(NON_COUNTRIES)]
cite = cite[cite.citing_country != cite.cited_country]  # exclude self-citations

PRE = range(1997, 2002)
POST = range(2003, 2008)

def pair_growth(df, i, j):
    """log growth of directed pair (i -> j), pre vs post."""
    pre = df[(df.citing_country == i) & (df.cited_country == j) &
             (df.year.isin(PRE))]["citation_count"].sum()
    post = df[(df.citing_country == i) & (df.cited_country == j) &
              (df.year.isin(POST))]["citation_count"].sum()
    return np.log1p(post) - np.log1p(pre)

# treatment partners: countries that have BOTH pre/post flows with CN, in either direction
partners_in = set(cite[(cite.citing_country == "CN") & (cite.year.isin(PRE))]["cited_country"]) & \
              set(cite[(cite.citing_country == "CN") & (cite.year.isin(POST))]["cited_country"])
partners_out = set(cite[(cite.cited_country == "CN") & (cite.year.isin(PRE))]["citing_country"]) & \
               set(cite[(cite.cited_country == "CN") & (cite.year.isin(POST))]["citing_country"])
partners_in.discard("CN")
partners_out.discard("CN")

def hub_control_growth(p, direction):
    """Growth of partner p's flows with hubs (US/JP/DE), excluding CN."""
    gs = []
    for h in HUBS:
        if direction == "in":  # control for 'CN cites p': other citers of p among hubs
            g = pair_growth(cite, h, p)
        else:                  # control for 'p cites CN': p's citing of hubs
            g = pair_growth(cite, p, h)
        gs.append(g)
    return float(np.mean(gs))

print(f"  Partners with pre+post flows: in={len(partners_in)}, out={len(partners_out)}")

rows = []
for p in sorted(partners_in):
    treat = pair_growth(cite, "CN", p)
    ctrl = hub_control_growth(p, "in")
    rows.append({"direction": "in", "partner": p, "treat": treat, "control": ctrl,
                 "did": treat - ctrl})
for p in sorted(partners_out):
    treat = pair_growth(cite, p, "CN")
    ctrl = hub_control_growth(p, "out")
    rows.append({"direction": "out", "partner": p, "treat": treat, "control": ctrl,
                 "did": treat - ctrl})

did_df = pd.DataFrame(rows)
print("\n  === Corrected DiD (pair-level, hub controls, no self-citations) ===")
for direction in ["in", "out"]:
    sub = did_df[did_df.direction == direction]
    if len(sub) == 0:
        continue
    label = "Knowledge INTO China (CN cites partner)" if direction == "in" \
        else "Knowledge OUT of China (partner cites CN)"
    print(f"\n  {label}  (n={len(sub)})")
    print(f"    CN-pair growth:   {sub.treat.mean():+.3f}")
    print(f"    Hub-pair growth:  {sub.control.mean():+.3f}")
    print(f"    DiD (within-partner): {sub.did.mean():+.3f}")

# simple significance: t-test on the did column
from scipy import stats
print("\n  === Significance (one-sample t on within-partner DiD) ===")
for direction in ["in", "out"]:
    sub = did_df[did_df.direction == direction]
    if len(sub) < 5:
        continue
    t_stat, p_val = stats.ttest_1samp(sub["did"], 0)
    print(f"    {direction}: mean={sub.did.mean():+.3f}, t={t_stat:+.2f}, "
          f"p={p_val:.3f}, n={len(sub)}")

# ============================================================
# 3. Honest interpretation
# ============================================================
print("\n[3/3] Interpretation (per audit) ==============================")
print("""
  Timing caveat: US applied MFN rates to China continuously since 1980 under
  Jackson-Vanik waivers; PNTR (Oct 2000, effective Jan 2002) removed renewal
  UNCERTAINTY, not tariff levels. China's own cuts were pre-announced in the
  1999-2001 accession agreements. The 2001 event is therefore a phase-in of
  committed liberalization plus an uncertainty shock — a weak design basis
  for a level-based tariff dose-response.

  Outcome caveat: family-level CN citation counts rise ~4x from 1997-2001 to
  2003-2007 for coverage/stock reasons unrelated to tariffs; the series is
  non-monotone (2004 peak, 2007 trough). The DiD here should be read as
  descriptive, not causal.

  Verdict: the DiD estimates knowledge-flow growth on China pairs relative to
  the same partners' hub links. The tariff dose-response remains UNMEASURED
  with this data (negative result documented above).
""")
did_df.to_csv("../processed/wto_did_corrected.csv", index=False)
print("Saved → ../processed/wto_did_corrected.csv")
