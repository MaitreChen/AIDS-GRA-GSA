import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

dataset = TUDataset(root='data/TUDataset', name='AIDS')

num_graphs = len(dataset)
num_features = dataset.num_features
num_classes = dataset.num_classes

num_nodes = [data.num_nodes for data in dataset]
num_edges = [data.num_edges for data in dataset]

data = dataset[1]

edge_index = data.edge_index.numpy()
node_features = data.x.numpy()
labels = data.y.item()

G = nx.Graph()
G.add_edges_from(zip(edge_index[0], edge_index[1]))

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(12):
    data = dataset[i]

    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    degree_sequence = [G.degree(node) for node in G.nodes()]
    degree_values, degree_counts = np.unique(degree_sequence, return_counts=True)

    axes[i].bar(degree_values, degree_counts, color='skyblue')
    # axes[i].set_title(f'Graph {i + 1} Node Degree Distribution')
    axes[i].set_xlabel('Degree')

    axes[i].set_ylabel('Frequency')

    # Add id
    axes[i].text(-0.1, 1.05, f'({chr(97 + i)})', transform=axes[i].transAxes, fontsize=12, verticalalignment='top',
                 horizontalalignment='right')

plt.show()
