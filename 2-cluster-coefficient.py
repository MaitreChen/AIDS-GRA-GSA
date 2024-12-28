import matplotlib.pyplot as plt
import seaborn as sns
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

clustering_coeffs = list(nx.clustering(G).values())

plt.figure(figsize=(8, 5))
sns.histplot(clustering_coeffs, kde=True, color='orange', bins=30)
plt.title('Clustering Coefficient Distribution')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.show()

clustering_coeffs_all_graphs = []

for data in dataset:
    edge_index = data.edge_index.numpy()  # (2, num_edges)

    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    clustering_coeffs = list(nx.clustering(G).values())
    clustering_coeffs_all_graphs.extend(clustering_coeffs)

# Collect clustering coefficients for all graphs in the dataset
clustering_coeffs_all_graphs = []

# Loop over each graph in the dataset
for data in dataset:
    edge_index = data.edge_index.numpy()  # (2, num_edges)

    # Create NetworkX graph
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))  # Add edges

    # Calculate clustering coefficients
    clustering_coeffs = list(nx.clustering(G).values())
    clustering_coeffs_all_graphs.extend(clustering_coeffs)  # Add to the list

# Set up the visualization
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
sns.kdeplot(clustering_coeffs_all_graphs, shade=True, color='C0', lw=2)
plt.title('Kernel Density Estimation of Clustering Coefficient')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Density')

plt.subplot(2, 2, 2)
plt.hist(clustering_coeffs_all_graphs, bins=30, color='skyblue', edgecolor='black')
plt.title('Clustering Coefficient Distribution for All Graphs')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
sns.violinplot(data=clustering_coeffs_all_graphs, color='C2')
plt.title('Violin Plot of Clustering Coefficient Distribution')
plt.ylabel('Clustering Coefficient')

plt.subplot(2, 2, 4)
plt.hist(clustering_coeffs_all_graphs, bins=50, color='C3', edgecolor='black', log=True)
plt.title('Log-scaled Histogram of Clustering Coefficient Distribution')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency (Log Scale)')
plt.show()
