import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import TUDataset


def set_default_node_options():
    """Set default node display options."""
    return {
        'node_size': 500,
        'node_color': 'skyblue',
        'alpha': 0.7,
        'linewidths': 1,
    }


def set_default_edge_options():
    """Set default edge display options."""
    return {
        'width': 1.0,
        'alpha': 0.7,
        'edge_color': 'gray',
    }


def set_default_node_labels():
    """Set default node label display options."""
    return {
        'font_size': 12,
        'font_weight': 'bold',
        'font_color': 'white',  # White color for node labels
    }


def draw_graph(G, pos, node_options=None, edge_options=None, node_labels=None, edge_labels=None):
    """
    Draw the graph and display it with various custom options.

    G: NetworkX graph
    pos: Node positions (layout)
    node_options: Options for nodes (color, size, transparency, etc.)
    edge_options: Options for edges (color, width, transparency, etc.)
    node_labels: Options for node labels (font size, weight, color)
    edge_labels: Edge labels (if provided)
    """
    # Set default options if not provided
    if node_options is None:
        node_options = set_default_node_options()
    if edge_options is None:
        edge_options = set_default_edge_options()
    if node_labels is None:
        node_labels = set_default_node_labels()

    # Draw nodes with custom options
    nx.draw_networkx_nodes(G, pos, **node_options)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, **node_labels)

    # Draw edges with custom options
    nx.draw_networkx_edges(G, pos, **edge_options)

    # Draw edge labels if available
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Hide axis for a cleaner plot
    plt.axis('off')
    plt.gca().set_aspect('equal')

    # Save the plot as a PNG file
    save_path = '../results/graph3_visualization.png'
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=400)

    # Show the plot
    plt.show()


# Load the AIDS dataset
dataset = TUDataset(root='data/TUDataset', name='AIDS')

# Select the third graph for visualization
data = dataset[3]

# Convert the graph to a NetworkX graph
edge_index = data.edge_index.numpy()  # (2, num_edges)
node_features = data.x.numpy()  # (num_nodes, num_features)
labels = data.y.item()  # Get the label (for coloring)

# Create a NetworkX graph
G = nx.Graph()
G.add_edges_from(zip(edge_index[0], edge_index[1]))  # Add edges

# Add features to each node
for i, feature in enumerate(node_features):
    G.nodes[i]['feature'] = feature  # Store each node's feature

# Set node color based on label (simplified example with uniform color)
node_color = [labels] * len(G.nodes)  # Use a single color for all nodes in this example

# Set layout for node positions
pos = nx.spring_layout(G, seed=42)  # Use spring layout algorithm

# Draw the graph with customized options
plt.figure(figsize=(10, 8))
draw_graph(
    G, pos,
    node_options={'node_color': node_color, 'node_size': 700, 'alpha': 0.9},
    edge_options={'edge_color': 'gray', 'width': 1.2, 'alpha': 0.6},
    node_labels={'font_size': 12, 'font_weight': 'bold', 'font_color': 'white'}
)
