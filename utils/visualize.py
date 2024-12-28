import matplotlib.pyplot as plt
import networkx as nx


def visualize_attention_weights(attention_weights, edge_index, num_nodes):
    # Extract attention weights, attention_weights is a tuple, the second element contains the weights
    attention_weights = attention_weights[1]  # Get the attention weight part

    # Average the attention weights across multiple heads for each edge
    attention_weights = attention_weights.mean(dim=1)  # Average over multiple heads for each edge

    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Get the edge connection information
    edge_list = edge_index.t().tolist()

    # Add the attention weights to the graph edges
    for i, (src, dst) in enumerate(edge_list):
        weight = attention_weights[i].item()  # Get the average attention weight for each edge
        G.add_edge(src, dst, weight=weight)

    # Use networkx to draw the graph
    pos = nx.spring_layout(G)  # Use spring layout algorithm to determine node positions
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Format edge labels to two decimal places
    edge_labels = {k: f'{v:.2f}' for k, v in edge_weights.items()}

    # Draw the graph
    plt.figure(figsize=(12, 12))  # Set the figure size
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, font_weight='bold', node_color='skyblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Attention Weights Visualization")
    plt.show()
