from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
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


network_density = nx.density(G)

print(f'Network Density: {network_density:.4f}')


network_densities = []

for i in range(num_graphs):
    data = dataset[i]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    network_density = nx.density(G)
    network_densities.append(network_density)


for i, density in enumerate(network_densities):
    print(f'Graph {i + 1} Network Density: {density:.4f}')

plt.figure(figsize=(20, 6))
plt.plot(range(1, num_graphs+1), network_densities, marker='o', linestyle='-', color='b', label='Network Density')

plt.title('Network Density of Each Graph in AIDS Dataset', fontsize=16)
plt.xlabel('Graph Index', fontsize=12)
plt.ylabel('Network Density', fontsize=12)
plt.xticks(range(1, num_graphs+1, max(1, num_graphs // 10)))
plt.grid(True)
plt.legend()
plt.show()