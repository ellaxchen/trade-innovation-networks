import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the CSV file
edges = pd.read_csv('../processed/trade_edges_2020.csv')

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for index, row in edges.iterrows():
    G.add_edge(row['exporter'], row['importer'], weight=row['trade_value'])

# Set up figure
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Draw nodes
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')

# Draw edges with weight labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show plot
plt.title('2020 Trade Network (Top 200 Flows)')
plt.show()