import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.datasets import TUDataset
import networkx as nx

# Load the AIDS dataset
dataset = TUDataset(root='data/TUDataset', name='AIDS')

# Get dataset information
num_graphs = len(dataset)
num_features = dataset.num_features
num_classes = dataset.num_classes

# Print dataset information
print(f'Dataset: {dataset}')
print(f'Number of graphs: {num_graphs}')
print(f'Number of features: {num_features}')
print(f'Number of classes: {num_classes}')

# Get the distribution of node and edge counts
num_nodes = [data.num_nodes for data in dataset]
num_edges = [data.num_edges for data in dataset]

# Select the first graph for visualization
data = dataset[1]

# Convert the graph to a NetworkX graph
# Create edges using data.edge_index and nodes using data.x
edge_index = data.edge_index.numpy()  # (2, num_edges)
node_features = data.x.numpy()  # (num_nodes, num_features)
labels = data.y.item()  # Get the label (for coloring)

# Create the NetworkX graph
G = nx.Graph()
G.add_edges_from(zip(edge_index[0], edge_index[1]))  # Add edges

# Add features to each node
for i, feature in enumerate(node_features):
    G.nodes[i]['feature'] = feature  # Store each node's feature

# Set node colors based on labels (simplified example using a uniform color)
node_color = [labels] * len(G.nodes)  # Set the node color based on the label

# Set layout
pos = nx.spring_layout(G, seed=42)  # Use spring layout algorithm

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    node_color=node_color,
    cmap=plt.cm.Blues,  # Color map
    font_size=10,
    font_weight='bold',
    edge_color='gray',  # Edge color
    width=1.0,  # Edge width
    alpha=0.7  # Node transparency
)
plt.title('Graph Visualization of the First AIDS Data Graph', fontsize=15)
plt.show()

# Set font style and size to be more suitable for Nature style
sns.set_context('talk', font_scale=1.2)  # Larger font size
sns.set_style('whitegrid')  # White grid background

# Set Matplotlib default font to a more simple font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

# 1. Graph Structure Visualization - Node and Edge Count Distributions
plt.figure(figsize=(20, 5))

# Node count distribution
plt.subplot(1, 2, 1)
sns.histplot(num_nodes, kde=True, color='C0', bins=20)
# plt.title('Node Count Distribution')
plt.xlabel('Number of Nodes')
plt.ylabel('Frequency')

# Edge count distribution
plt.subplot(1, 2, 2)
sns.histplot(num_edges, kde=True, color='C1', bins=20)
# plt.title('Edge Count Distribution')
plt.xlabel('Number of Edges')
plt.ylabel('Frequency')

plt.savefig('distribution.png', bbox_inches='tight', dpi=400)
plt.show()

# 2. Feature Distribution - Check the distribution of node features
# Assuming the features are scalar or a feature vector for each node (dimension num_features)
# Here we will visualize the features for the first graph
first_graph = dataset[0]
node_features = first_graph.x.numpy()

# Visualize the node feature distribution for the first graph
plt.figure(figsize=(8, 5))
sns.histplot(node_features.flatten(), kde=True, color='C2', bins=30)
plt.title('Node Feature Distribution (First Graph)')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()

# 3. Class Distribution Visualization
labels = [data.y.item() for data in dataset]

plt.figure(figsize=(8, 5))
sns.countplot(x=labels, palette='Set2', legend=False)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
