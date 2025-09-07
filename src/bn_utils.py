import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, Dict, Any

def _draw_digraph_hierarchical(G: nx.DiGraph, 
                              title: str = "Network Structure (Hierarchical Layout)",
                              node_size: int = 3000, 
                              node_color: str = 'lightblue', 
                              font_size: int = 12,
                              figsize: Tuple[int, int] = (10, 6),
                              show_treewidth: bool = True) -> Dict[str, Any]:
    """
    Core method to draw a NetworkX DiGraph with hierarchical layout.
    Parents are placed above children.
    
    Args:
        G: NetworkX DiGraph to draw
        title: Plot title
        node_size: Size of nodes
        node_color: Color of nodes
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display treewidth in title (default: True)
        
    Returns:
        Dictionary with layout info and treewidth (if computed)
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a NetworkX DiGraph")
    
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG (Directed Acyclic Graph)")
    
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

    # Compute treewidth if requested
    display_title = title
    result = {
        "positions": pos,
        "layers": layers,
        "layer_nodes": layer_nodes
    }
    
    if show_treewidth:
        try:
            from networkx.algorithms.approximation import treewidth
            # Convert to undirected for treewidth computation
            undirected_G = G.to_undirected()
            width, decomposition = treewidth.treewidth_min_degree(undirected_G)
            result["treewidth"] = {"width": width, "decomposition": decomposition}
            display_title = f"{title} (Treewidth ≈ {width})"
        except ImportError:
            print("Warning: Could not compute treewidth. NetworkX approximation module not available.")

    # Draw the network
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color,
            font_size=font_size, font_weight='bold', arrows=True)
    plt.title(display_title)
    plt.show()
    
    return result

def draw_bayesian_network(model, 
                         node_size: int = 3000, 
                         node_color: str = 'lightblue', 
                         font_size: int = 12,
                         figsize: Tuple[int, int] = (10, 6),
                         show_treewidth: bool = True):
    """
    Draws a Bayesian network (pgmpy) with an automatic hierarchical layout.
    Parents are placed above children.
    
    Args:
        model: pgmpy BayesianNetwork or LinearGaussianBayesianNetwork
        node_size: Size of nodes
        node_color: Color of nodes  
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display approximate treewidth (default: True)
        
    Returns:
        Dictionary with layout info and treewidth (if computed)
    """
    # Convert BayesianNetwork edges to NetworkX DiGraph
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    
    return _draw_digraph_hierarchical(
        G, 
        title="Bayesian Network Structure (Hierarchical Layout)",
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        figsize=figsize,
        show_treewidth=show_treewidth
    )

def draw_networkx_graph(G: nx.DiGraph,
                       title: Optional[str] = None,
                       node_size: int = 3000,
                       node_color: str = 'lightgreen', 
                       font_size: int = 12,
                       figsize: Tuple[int, int] = (10, 6),
                       show_treewidth: bool = True):
    """
    Draws a NetworkX DiGraph with an automatic hierarchical layout.
    Parents are placed above children.
    
    Args:
        G: NetworkX DiGraph to draw
        title: Plot title (auto-generated if None)
        node_size: Size of nodes
        node_color: Color of nodes
        font_size: Font size for labels
        figsize: Figure size (width, height)
        show_treewidth: Whether to compute and display approximate treewidth (default: True)
        
    Returns:
        Dictionary with layout info and treewidth (if computed)
    """
    if title is None:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        title = f"NetworkX Graph ({n_nodes} nodes, {n_edges} edges)"
    
    return _draw_digraph_hierarchical(
        G,
        title=title,
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        figsize=figsize,
        show_treewidth=show_treewidth
    )