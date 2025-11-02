import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, Dict, Any

def num_edges(bn):
    # For pgmpy BayesianModel, the edges can be accessed by .edges
    return len(list(bn.edges()))

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

def compute_average_markov_blanket_size(bn):
    # Compute Markov blanket for each node: parents, children, and other parents of children
    node_blankets = []
    for node in bn.nodes():
        parents = set(bn.predecessors(node))
        children = set(bn.successors(node))
        other_parents = set()
        for child in children:
            other_parents.update(bn.predecessors(child))
        # Markov blanket = parents ∪ children ∪ (other parents of children) minus the node itself
        blanket = parents | children | other_parents
        blanket.discard(node)
        node_blankets.append(len(blanket))
    if node_blankets:
        return sum(node_blankets) / len(node_blankets)
    else:
        return 0.0

def is_barren(node, bn, query_vars, evidence_nodes, barren_vars):
    """Check if a node is barren: not target/evidence, has no descendants, or all descendants are barren."""
    if node in query_vars or node in evidence_nodes:
        return False  # Targets and evidence are never barren
    
    children = list(bn.get_children(node))
    if len(children) == 0:
        return True  # Leaf node is barren
    
    # Check if all descendants are barren
    descendants = set()
    stack = [node]
    visited = {node}
    while stack:
        current = stack.pop()
        for child in bn.get_children(current):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                stack.append(child)
    
    # Remove query and evidence from descendants check
    descendants = descendants - set(query_vars) - set(evidence_nodes)
    
    if len(descendants) == 0:
        return True
    
    # All descendants must be barren
    all_descendants_barren = True
    for desc in descendants:
        if desc not in barren_vars:
            all_descendants_barren = False
            break
    
    return all_descendants_barren

def identify_barren_nodes(bn, query_vars, evidence_nodes, barren_vars):
    """
    Iteratively identify barren nodes in the Bayesian Network.
    Updates barren_vars in place and returns the set.
    """
    changed = True
    while changed:
        changed = False
        for node in bn.nodes():
            if node in query_vars or node in evidence_nodes or node in barren_vars:
                continue

            if is_barren(node, bn, query_vars, evidence_nodes, barren_vars):
                barren_vars.add(node)
                changed = True
    return barren_vars


def find_conditionally_independent_vars(bn, query_vars, evidence_nodes):
    """
    Return a set of variables in `bn` that are independent of all `query_vars` given `evidence_nodes`.
    """
    independent_vars = set()
    for var in bn.nodes():
        if var in query_vars or var in evidence_nodes:
            continue  # Skip query and evidence variables

        # Check if this variable is independent of all query variables given evidence
        is_independent = True
        for query_var in query_vars:
            # Use d-separation: variable is independent of query_var given evidence
            if bn.is_dconnected(var, query_var, observed=evidence_nodes):
                is_independent = False
                break

        if is_independent:
            independent_vars.add(var)
    return independent_vars


def compute_query_complexity(bn, target_nodes, evidence_nodes, verbose=False):
    """
    Compute query complexity by first removing independent and barren nodes, then computing variable elimination complexity.
    
    This function:
    1. Creates a copy of the BN
    2. Identifies and removes variables independent of targets given evidence
    3. Identifies and removes barren nodes
    4. Computes variable elimination complexity on the reduced network
    
    Parameters:
    - bn: Bayesian network (pgmpy DiscreteBayesianNetwork)
    - target_nodes: List of target/query variable names
    - evidence_nodes: List of evidence variable names (or empty list)
    - verbose: If True, print detailed progress information
    
    Returns:
    - dict: Complexity metrics including induced width, total cost, max factor size, etc.
    """
    from pgmpy.models import BayesianNetwork
    from pgmpy.inference.EliminationOrder import WeightedMinFill
    from pgmpy.inference import VariableElimination
    import numpy as np
    
    # Step 1: Create a copy of the network (work on copy to avoid modifying original)
    # We'll build the reduced network from scratch
    if verbose:
        print(f"Original network: {len(bn.nodes())} nodes, {bn.number_of_edges()} edges")
    
    # Step 2: Find conditionally independent variables
    independent_vars = find_conditionally_independent_vars(bn, target_nodes, evidence_nodes)
    if verbose:
        print(f"Found {len(independent_vars)} independent variables: {sorted(independent_vars)}")
    
    # Step 3: Find barren nodes (start with empty set, function updates it)
    barren_vars = set()
    identify_barren_nodes(bn, target_nodes, evidence_nodes, barren_vars)
    if verbose:
        print(f"Found {len(barren_vars)} barren variables: {sorted(barren_vars)}")
    
    # Step 4: Create reduced network
    vars_to_remove = independent_vars | barren_vars
    vars_to_keep = set(bn.nodes()) - vars_to_remove
    
    if verbose:
        print(f"Removing {len(vars_to_remove)} variables, keeping {len(vars_to_keep)} variables")
        print(f"Variables to keep: {sorted(vars_to_keep)}")
    
    # Create a new Bayesian network with only kept variables
    reduced_bn = BayesianNetwork()
    reduced_bn.add_nodes_from(vars_to_keep)
    
    # Add edges that connect kept variables
    for edge in bn.edges():
        u, v = edge
        if u in vars_to_keep and v in vars_to_keep:
            reduced_bn.add_edge(u, v)
    
    # Copy CPDs for kept variables (only those with parents that are also kept)
    for node in vars_to_keep:
        try:
            cpd = bn.get_cpds(node)
            parents = list(cpd.variables)
            parents.remove(node)  # Remove the node itself from parents list
            
            # Check if all parents are kept
            if all(p in vars_to_keep for p in parents):
                reduced_bn.add_cpds(cpd)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not copy CPD for {node}: {e}")
    
    if verbose:
        print(f"Reduced network: {len(reduced_bn.nodes())} nodes, {reduced_bn.number_of_edges()} edges")
    
    # Step 5: Compute complexity on reduced network
    # Ensure the model is valid
    reduced_bn.check_model()
    
    # Get cardinalities
    card = reduced_bn.get_cardinality()
    if verbose:
        print(f"Variable cardinalities: {dict(card)}")
    
    # Identify which variables need to be eliminated vs kept for the query
    all_vars = set(reduced_bn.nodes())
    target_vars_set = set(target_nodes)
    evidence_vars_set = set(evidence_nodes)
    
    # Variables that must be kept until the end (target variables)
    keep_vars = target_vars_set & all_vars  # Intersection to only include vars in reduced network
    
    # Variables that can be eliminated (all others in reduced network)
    eliminate_vars = all_vars - keep_vars
    
    if verbose:
        print(f"Variables to keep (targets): {sorted(keep_vars)}")
        print(f"Variables to eliminate: {sorted(eliminate_vars)}")
        print(f"Evidence variables: {sorted(evidence_vars_set & all_vars)}")
    
    # Handle evidence by reducing cardinalities
    # Evidence variables are instantiated, so they don't contribute to factor sizes
    effective_card = card.copy()
    for evar in evidence_vars_set & all_vars:
        effective_card[evar] = 1  # Evidence variables are fixed, so cardinality = 1
    
    if verbose:
        print(f"Effective cardinalities (after evidence): {dict(effective_card)}")
    
    # Create elimination orderer and get optimal order for variables to eliminate
    if eliminate_vars:
        orderer = WeightedMinFill(reduced_bn)
        elim_order = orderer.get_elimination_order(nodes=list(eliminate_vars))
    else:
        elim_order = []  # No variables to eliminate
    
    if verbose:
        print(f"Elimination order (variables to eliminate): {elim_order}")
        if elim_order:
            complete_elim_order = elim_order + list(keep_vars)
            print(f"Complete elimination order: {complete_elim_order}")
    
    # Calculate induced width for the elimination order
    if elim_order:
        # For induced width calculation, we need to create a complete elimination order
        # that includes all variables, with target variables at the end
        complete_elim_order = elim_order + list(keep_vars)
        ve = VariableElimination(reduced_bn)
        induced_width = ve.induced_width(complete_elim_order)
    else:
        complete_elim_order = list(keep_vars)  # Only target variables
        induced_width = 0  # No elimination needed
    
    if verbose:
        print(f"Induced width: {induced_width}")
    
    # Simulate variable elimination to compute cost metrics
    cost = 0
    max_factor_size = 0
    moral = reduced_bn.to_markov_model()  # moralized undirected graph
    
    # Track factor sizes for each elimination step
    factor_sizes = []
    
    for step, x in enumerate(elim_order):
        nbrs = list(moral.neighbors(x))
        
        # Size of the intermediate factor created when eliminating x
        # Use effective cardinalities (evidence variables have cardinality 1)
        size = 1
        for v in nbrs + [x]:
            size *= effective_card[v]
        
        cost += size
        max_factor_size = max(max_factor_size, size)
        factor_sizes.append(size)
        
        if verbose:
            print(f"Step {step+1}: Eliminating {x}, neighbors: {nbrs}, factor size: {size}")
        
        # Connect neighbors (fill-in) and remove x
        for i in range(len(nbrs)):
            for j in range(i+1, len(nbrs)):
                moral.add_edge(nbrs[i], nbrs[j])
        moral.remove_node(x)
    
    # Calculate final factor size (the remaining target variables)
    if keep_vars:
        # The final factor contains all remaining target variables
        final_factor_size = 1
        for v in keep_vars:
            final_factor_size *= effective_card[v]
        cost += final_factor_size
        max_factor_size = max(max_factor_size, final_factor_size)
        if verbose:
            print(f"Final factor (target variables): {sorted(keep_vars)}, size: {final_factor_size}")
    
    # Calculate additional metrics
    num_vars = len(reduced_bn.nodes())
    num_edges = reduced_bn.number_of_edges()
    
    # Query-specific metrics
    num_target_vars = len(target_nodes)
    num_evidence_vars = len(evidence_nodes)
    num_eliminated_vars = len(elim_order)
    
    # Complexity metrics
    complexity_metrics = {
        'original_num_vars': len(bn.nodes()),
        'reduced_num_vars': num_vars,
        'num_independent_vars': len(independent_vars),
        'num_barren_vars': len(barren_vars),
        'num_vars_removed': len(vars_to_remove),
        'num_edges': num_edges,
        'num_target_vars': num_target_vars,
        'num_evidence_vars': num_evidence_vars,
        'num_eliminated_vars': num_eliminated_vars,
        'elimination_order': elim_order,
        'complete_elimination_order': complete_elim_order,
        'induced_width': induced_width,
        'total_cost': cost,
        'max_factor_size': max_factor_size,
        'avg_factor_size': cost / len(elim_order) if elim_order else 0,
        'factor_sizes': factor_sizes,
        'log_total_cost': np.log2(cost) if cost > 0 else 0,
        'log_max_factor_size': np.log2(max_factor_size) if max_factor_size > 0 else 0,
        'keep_vars': sorted(keep_vars),
        'eliminate_vars': sorted(eliminate_vars),
    }
    
    if verbose:
        print(f"\nQuery Complexity Summary:")
        print(f"  Original variables: {len(bn.nodes())}")
        print(f"  Variables removed (independent + barren): {len(vars_to_remove)}")
        print(f"  Reduced network variables: {num_vars}")
        print(f"  Variables eliminated: {num_eliminated_vars}/{num_vars}")
        print(f"  Target variables kept: {sorted(keep_vars)}")
        print(f"  Induced width: {induced_width}")
        print(f"  Total factor work: {cost:,}")
        print(f"  Max intermediate factor size: {max_factor_size:,}")
        print(f"  Average factor size: {cost / len(elim_order) if elim_order else 0:.1f}")
        print(f"  Log2(total cost): {np.log2(cost):.2f}")
        print(f"  Log2(max factor size): {np.log2(max_factor_size):.2f}")
    
    return complexity_metrics