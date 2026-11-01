"""
Semiconductor trade extraction: HS 8541 + 8542 (HS92).

8541 = diodes, transistors, semiconductor devices, LEDs, mounted piezo crystals
8542 = electronic integrated circuits and microassemblies

Loads the BACI panel once with a column filter, keeps 8541xx/8542xx flows,
aggregates to (exporter, importer, year) and (exporter, year).

Outputs:
  ../processed/semiconductor_trade_edges_by_year.csv  (source, target, year, trade_value)
  ../processed/semiconductor_trade_by_exporter_year.csv (exporter, year, trade_value)

Usage: python semiconductor_trade.py
"""
import pandas as pd

print("[1/2] Loading BACI panel (column-filtered)...")
trade = pd.read_parquet(
    "../rawdata/baci_panel_hs6.parquet",
    columns=["year", "exporter", "importer", "hs6", "trade_value"]
)
print(f"  {len(trade):,} rows loaded")

# hs6 is a string column in this parquet — filter on the first four digits
semi = trade[trade["hs6"].str[:4].isin(["8541", "8542"])]
del trade
print(f"  {len(semi):,} semiconductor (8541/8542) rows")

print("[2/2] Aggregating...")
edges = semi.groupby(["year", "exporter", "importer"], as_index=False)["trade_value"].sum()
by_exp = semi.groupby(["year", "exporter"], as_index=False)["trade_value"].sum()

edges.to_csv("../processed/semiconductor_trade_edges_by_year.csv", index=False)
by_exp.to_csv("../processed/semiconductor_trade_by_exporter_year.csv", index=False)
print(f"  Saved {len(edges):,} directed edges → ../processed/semiconductor_trade_edges_by_year.csv")
print(f"  Saved {len(by_exp):,} exporter-year rows → ../processed/semiconductor_trade_by_exporter_year.csv")

# Top exporters, selected years (values in thousands USD)
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
baci_to_name = dict(zip(cc["country_code"], cc["country_name"]))
by_exp["name"] = by_exp["exporter"].map(baci_to_name)
print("\n=== Top 10 semiconductor exporters by year (USD billions) ===")
for yr in [1995, 2000, 2005, 2010, 2013, 2024]:
    sub = by_exp[by_exp["year"] == yr].nlargest(10, "trade_value")
    top = "  ".join(f"{r.name[:14]}:{r.trade_value/1e6:,.1f}" for r in sub.itertuples())
    print(f"  {yr}: {top}")

print("\nDone.")
