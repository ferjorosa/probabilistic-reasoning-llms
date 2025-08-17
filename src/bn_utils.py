import matplotlib.pyplot as plt
import networkx as nx

def draw_bayesian_network(model, node_size=3000, node_color='lightblue', font_size=12):
    """
    Draws a Bayesian network (pgmpy) with an automatic hierarchical layout.
    Parents are placed above children.
    """
    # Convert BayesianNetwork edges to NetworkX DiGraph
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    
    # Topological sort of nodes
    topo_order = list(nx.topological_sort(G))
    
    # Assign y-levels based on topological layers
    layers = {}
    for node in topo_order:
        parents = list(G.predecessors(node))
        if parents:
            layers[node] = max(layers[p] for p in parents) + 1
        else:
            layers[node] = 0  # root nodes at top layer

    # Assign x-positions to spread nodes horizontally
    layer_nodes = {}
    for node, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(node)

    pos = {}
    for layer, nodes in layer_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (i - n/2, -layer)  # center horizontally, invert y-axis for top-down

    # Draw the network
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color,
            font_size=font_size, font_weight='bold', arrows=True)
    plt.title("Bayesian Network Structure (Hierarchical Layout)")
    plt.show()