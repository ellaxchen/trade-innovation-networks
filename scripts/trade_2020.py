import pandas as pd

cols = ["year", "exporter", "importer", "trade_value"]
df = pd.read_parquet('../rawdata/baci_panel_hs6.parquet', columns=cols)
edges = df[df['year'] == 2020].groupby(['exporter','importer']).sum().reset_index()
edges.nlargest(200, 'trade_value').to_csv('../processed/trade_edges_2020.csv', index=False)
print(f"Saved {len(edges)} trade edges from 2020 (only top 200 written to CSV)")