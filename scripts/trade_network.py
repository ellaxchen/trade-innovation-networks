"""
Generate trade edge CSVs for one or more years from BACI panel data.
Loads the parquet file ONCE then filters per year.
Usage: python trade_network.py              # default: 5-year snapshots
       python trade_network.py --year 2020  # single year
       python trade_network.py --all-years  # every year 1995-2024
"""
import pandas as pd
import argparse
import sys

COLUMNS = ["year", "exporter", "importer", "trade_value"]
TOP_N = 200

# Country code lookup for readable names
try:
    countries = pd.read_csv("../rawdata/country_codes_V202601.csv")
    COUNTRY_MAP = dict(zip(countries["country_code"], countries["country_name"]))
except Exception:
    COUNTRY_MAP = {}

# Parse args
parser = argparse.ArgumentParser(description="Generate trade network edge CSVs")
parser.add_argument("--year", type=int, help="Single year to process")
parser.add_argument("--all-years", action="store_true", help="Every year 1995-2024")
args = parser.parse_args()

if args.year:
    years = [args.year]
elif args.all_years:
    years = list(range(1995, 2025))
else:
    years = [1995, 2000, 2005, 2010, 2015, 2020, 2024]

# Load data once
print(f"Loading trade data ({', '.join(map(str, years))})...", file=sys.stderr)
df = pd.read_parquet("../rawdata/baci_panel_hs6.parquet", columns=COLUMNS)
print(f"Loaded {len(df):,} rows", file=sys.stderr)

for year in years:
    yr = df[df["year"] == year]
    edges = (
        yr.groupby(["exporter", "importer"], as_index=False)["trade_value"]
        .sum()
        .sort_values("trade_value", ascending=False)
    )
    top = edges.head(TOP_N).copy()
    top["year"] = year
    top["exporter_name"] = top["exporter"].map(COUNTRY_MAP)
    top["importer_name"] = top["importer"].map(COUNTRY_MAP)

    out_path = f"../processed/trade_edges_{year}.csv"
    top.to_csv(out_path, index=False)
    top_pct = (top["trade_value"].sum() / edges["trade_value"].sum()) * 100
    print(f"{year}: {len(edges):,} flows, top {TOP_N} = {top_pct:.1f}% of value -> {out_path}")
