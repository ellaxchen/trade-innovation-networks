"""
Generate sector-level trade edge CSVs by HS2 chapter.
Filters baci_panel_hs6.parquet to a specific HS chapter before aggregation.

Usage:
  python trade_by_sector.py                        # default sectors, 2015
  python trade_by_sector.py --year 2015 --hs2 85   # electronics only
  python trade_by_sector.py --all-sectors          # all major sectors
"""
import pandas as pd
import argparse

COLUMNS = ["year", "exporter", "importer", "hs2", "trade_value"]
TOP_N = 200

# Major HS2 sectors with CPC counterparts
SECTORS = {
    "85": "Electronics",
    "87": "Vehicles",
    "30": "Pharmaceuticals",
    "84": "Machinery",
    "27": "Oil & Fuels",
    "72": "Iron & Steel",
    "39": "Plastics",
    "62": "Apparel",
}

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2015)
parser.add_argument("--hs2", type=str, help="Single HS2 chapter")
parser.add_argument("--all-sectors", action="store_true")
args = parser.parse_args()

YEAR = args.year

# Load country names
cc = pd.read_csv("../rawdata/country_codes_V202601.csv")
COUNTRY_MAP = dict(zip(cc["country_code"], cc["country_name"]))
NAME_TO_ISO = dict(zip(cc["country_name"], cc["country_iso2"]))

# Determine which sectors to process
if args.hs2:
    sectors_to_run = {args.hs2: SECTORS.get(args.hs2, f"HS{args.hs2}")}
else:
    sectors_to_run = dict(list(SECTORS.items())[:6])  # default: first 6

print(f"Loading trade data for {YEAR}...")
df = pd.read_parquet("../rawdata/baci_panel_hs6.parquet", columns=COLUMNS)
df = df[df["year"] == YEAR]
df["hs2"] = df["hs2"].astype(str).str.zfill(2)
print(f"  {len(df):,} rows for {YEAR}")

for hs2, name in sectors_to_run.items():
    sector_df = df[df["hs2"] == hs2]
    if len(sector_df) == 0:
        print(f"  HS{hs2} ({name}): no data")
        continue

    edges = (
        sector_df.groupby(["exporter", "importer"], as_index=False)["trade_value"]
        .sum()
        .sort_values("trade_value", ascending=False)
    )

    top = edges.head(TOP_N).copy()
    top["year"] = year = YEAR
    top["hs2"] = hs2
    top["exporter_name"] = top["exporter"].map(COUNTRY_MAP)
    top["importer_name"] = top["importer"].map(COUNTRY_MAP)

    out_path = f"../processed/trade_edges_{YEAR}_hs{hs2}.csv"
    top.to_csv(out_path, index=False)

    top_pct = (top["trade_value"].sum() / edges["trade_value"].sum()) * 100
    total_val = edges["trade_value"].sum() / 1e6
    print(f"  HS{hs2} {name}: {len(edges):,} flows, ${total_val:,.0f}M total, "
          f"top {TOP_N} = {top_pct:.1f}% -> {out_path}")

print("Done.")
