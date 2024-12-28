import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.datasets import TUDataset
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

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.histplot(list(degree_centrality.values()), kde=True, color='green', bins=30, ax=axes[0])
axes[0].set_title('Degree Centrality Distribution')
axes[0].set_xlabel('Degree Centrality')
axes[0].set_ylabel('Frequency')
axes[0].text(-0.2, 1.05, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold')

sns.histplot(list(betweenness_centrality.values()), kde=True, color='red', bins=30, ax=axes[1])
axes[1].set_title('Betweenness Centrality Distribution')
axes[1].set_xlabel('Betweenness Centrality')
axes[1].set_ylabel('Frequency')
axes[1].text(-0.2, 1.05, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold')

sns.histplot(list(closeness_centrality.values()), kde=True, color='blue', bins=30, ax=axes[2])
axes[2].set_title('Closeness Centrality Distribution')
axes[2].set_xlabel('Closeness Centrality')
axes[2].set_ylabel('Frequency')
axes[2].text(-0.2, 1.05, '(c)', transform=axes[2].transAxes, fontsize=16, fontweight='bold')
plt.show()
