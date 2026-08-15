"""
Analyze trade network evolution across years.
Computes centrality metrics and identifies structural changes.
Usage: python trade_evolution.py
"""
import pandas as pd
import networkx as nx
import numpy as np

YEARS = [1995, 2000, 2005, 2010, 2015, 2020, 2024]

print("Computing trade network metrics across years...\n")

metrics = []
for year in YEARS:
    df = pd.read_csv(f"../processed/trade_edges_{year}.csv")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["exporter_name"], row["importer_name"], weight=row["trade_value"])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Centrality metrics
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="weight", k=min(n_nodes, 100))
    pagerank = nx.pagerank(G, weight="weight")

    # Top importers (by in-degree centrality)
    top_importers = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    # Top exporters (by out-degree centrality)
    top_exporters = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    # Most central (by betweenness)
    top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
    # Top PageRank
    top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]

    # Total trade value
    total_value = sum(d["weight"] for _, _, d in G.edges(data=True))

    metrics.append({
        "year": year,
        "nodes": n_nodes,
        "edges": n_edges,
        "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
        "total_value_B": total_value / 1e9,
        "top_importer": top_importers[0][0] if top_importers else "",
        "top_exporter": top_exporters[0][0] if top_exporters else "",
        "hub_country": top_between[0][0] if top_between else "",
        "top_pagerank": top_pr[0][0] if top_pr else "",
        "top5_importers": top_importers,
        "top5_exporters": top_exporters,
    })

# ---------- print evolution table ----------
print(f"{'Year':<6} {'Nodes':<7} {'Edges':<7} {'Density':<10} {'Total Value (B$)':<18} {'Top Exporter':<20} {'Top Importer':<20} {'Hub':<15}")
print("-" * 115)
for m in metrics:
    print(f"{m['year']:<6} {m['nodes']:<7} {m['edges']:<7} {m['density']:<10.4f} {m['total_value_B']:<18.1f} {m['top_exporter']:<20} {m['top_importer']:<20} {m['hub_country']:<15}")

# ---------- key trends ----------
print("\n--- Key Trends ---")
first, last = metrics[0], metrics[-1]
print(f"Node growth: {first['nodes']} -> {last['nodes']} (+{last['nodes'] - first['nodes']})")
print(f"Trade value growth: ${first['total_value_B']:.0f}B -> ${last['total_value_B']:.0f}B ({last['total_value_B']/first['total_value_B']:.1f}x)")
print(f"Network density: {first['density']:.4f} -> {last['density']:.4f}")

# ---------- top players over time ----------
print("\n--- Top Importers by Year ---")
for m in metrics:
    countries = ", ".join(f"{c} ({v:.3f})" for c, v in m["top5_importers"])
    print(f"  {m['year']}: {countries}")

print("\n--- Top Exporters by Year ---")
for m in metrics:
    countries = ", ".join(f"{c} ({v:.3f})" for c, v in m["top5_exporters"])
    print(f"  {m['year']}: {countries}")

# ---------- China vs USA story ----------
print("\n--- China vs USA: Export Centrality ---")
for year in YEARS:
    df = pd.read_csv(f"../processed/trade_edges_{year}.csv")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["exporter_name"], row["importer_name"], weight=row["trade_value"])

    out_c = nx.out_degree_centrality(G)
    pr = nx.pagerank(G, weight="weight")
    china_exp = out_c.get("China", 0)
    usa_exp = out_c.get("USA", 0)
    china_pr = pr.get("China", 0)
    usa_pr = pr.get("USA", 0)
    print(f"  {year}: China out-centrality={china_exp:.4f}, USA={usa_exp:.4f} | China PageRank={china_pr:.4f}, USA={usa_pr:.4f}")
