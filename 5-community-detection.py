import networkx as nx
from torch_geometric.datasets import TUDataset
import community as community_louvain
import matplotlib.pyplot as plt

from utils.seed import set_random_seed

set_random_seed(0)

dataset = TUDataset(root='data/TUDataset', name='AIDS')

num_graphs = 10
fig, axes = plt.subplots(2, 5, figsize=(18, 10))


for i in range(num_graphs):
    data = dataset[i]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    partition = community_louvain.best_partition(G)
    community_colors = [partition[node] for node in G.nodes]
    labels = data.y.numpy()
    label_colors = ['red' if label == 0 else 'blue' for label in labels]

    ax = axes[i // 5, i % 5]

    pos = nx.spring_layout(G)
    ax.set_title(f'Graph {i + 1}: Community Detection')
    nx.draw(G, pos, node_color=community_colors, with_labels=True, node_size=100, cmap=plt.cm.jet, font_size=8, ax=ax, font_color='white')

    # ax2 = ax.twinx()
    # ax2.set_title('Node Classification (Red: 0, Blue: 1)')
    # nx.draw(G, pos, node_color=label_colors, with_labels=True, node_size=50, font_size=8, ax=ax2)

plt.show()
