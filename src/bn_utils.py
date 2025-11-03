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
    descendants = descendants 
    
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
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference.EliminationOrder import WeightedMinFill
    from pgmpy.inference import VariableElimination
    from pgmpy.factors.discrete import TabularCPD
    import numpy as np
    
    # Step 1: Create a copy of the network (work on copy to avoid modifying original)
    # We'll build the reduced network from scratch
    if verbose:
        print(f"Original network: {len(bn.nodes())} nodes, {bn.number_of_edges()} edges")
    
    # Step 2: Find conditionally independent variables
    independent_vars = find_conditionally_independent_vars(bn, target_nodes, evidence_nodes)
    if verbose:
        print(f"Found {len(independent_vars)} independent variables: {sorted(independent_vars)}")
    
    # Step 3: Create intermediate network with independent variables removed
    # This intermediate network will be used to find barren nodes
    vars_after_independent_removal = set(bn.nodes()) - independent_vars
    
    # Create intermediate network without independent variables
    intermediate_bn = DiscreteBayesianNetwork()
    intermediate_bn.add_nodes_from(vars_after_independent_removal)
    
    # Add edges that connect kept variables (excluding independent vars)
    for edge in bn.edges():
        u, v = edge
        if u in vars_after_independent_removal and v in vars_after_independent_removal:
            intermediate_bn.add_edge(u, v)
    
    # Copy CPDs for variables that remain (handling removed parents)
    for node in vars_after_independent_removal:
        cpd_added = False
        try:
            cpd = bn.get_cpds(node)
            parents = list(cpd.variables)
            parents.remove(node)  # Remove the node itself from parents list
            
            # Check if all parents are kept (after removing independent vars)
            if all(p in vars_after_independent_removal for p in parents):
                intermediate_bn.add_cpds(cpd)
                cpd_added = True
            else:
                # Some parents were removed - create marginal CPD (same logic as below)
                kept_parents = [p for p in parents if p in vars_after_independent_removal]
                original_card = bn.get_cardinality()
                node_card = original_card[node]
                cpd_values = cpd.values.copy()
                cpd_variable_order = list(cpd.variables)
                
                axes_to_sum = []
                for i, var in enumerate(cpd_variable_order[1:], start=1):
                    if var not in vars_after_independent_removal:
                        axes_to_sum.append(i)
                
                if axes_to_sum:
                    marginal_values = np.sum(cpd_values, axis=tuple(axes_to_sum))
                else:
                    marginal_values = cpd_values
                
                # Get state names from original CPD, preserving only for kept variables
                original_state_names = cpd.state_names
                state_names = {node: original_state_names[node]}
                for parent in kept_parents:
                    if parent in original_state_names:
                        state_names[parent] = original_state_names[parent]
                
                if len(kept_parents) == 0:
                    marginal_values = marginal_values.flatten()
                    if marginal_values.sum() > 0:
                        marginal_values = marginal_values / marginal_values.sum()
                    else:
                        marginal_values = np.ones(node_card) / node_card
                    marginal_cpd = TabularCPD(variable=node, variable_card=node_card, 
                                             values=marginal_values.reshape(-1, 1),
                                             state_names=state_names)
                    intermediate_bn.add_cpds(marginal_cpd)
                    cpd_added = True
                else:
                    kept_parent_indices = [cpd_variable_order.index(p) for p in kept_parents]
                    desired_order = [0]
                    for kept_parent in kept_parents:
                        orig_pos = cpd_variable_order.index(kept_parent)
                        num_removed_before = sum(1 for i, v in enumerate(cpd_variable_order[1:orig_pos], 1) 
                                                   if v not in vars_after_independent_removal)
                        current_pos = orig_pos - num_removed_before
                        desired_order.append(current_pos)
                    
                    marginal_values = np.transpose(marginal_values, axes=desired_order)
                    kept_parents_cards = [original_card[p] for p in kept_parents]
                    marginal_values = marginal_values / marginal_values.sum(axis=0, keepdims=True)
                    
                    marginal_cpd = TabularCPD(variable=node, variable_card=node_card,
                                             values=marginal_values,
                                             evidence=kept_parents,
                                             evidence_card=kept_parents_cards,
                                             state_names=state_names)
                    intermediate_bn.add_cpds(marginal_cpd)
                    cpd_added = True
        except Exception as e:
            if verbose:
                print(f"Warning: Could not copy CPD for {node} in intermediate network: {e}")
        
        if not cpd_added:
            try:
                node_card = bn.get_cardinality()[node]
                uniform_values = np.ones((node_card, 1)) / node_card
                # Try to get state names from original CPD
                state_names = {}
                try:
                    original_cpd = bn.get_cpds(node)
                    if node in original_cpd.state_names:
                        state_names = {node: original_cpd.state_names[node]}
                except:
                    pass  # If we can't get state names, create CPD without them
                
                fallback_cpd = TabularCPD(variable=node, variable_card=node_card,
                                          values=uniform_values,
                                          state_names=state_names if state_names else None)
                intermediate_bn.add_cpds(fallback_cpd)
            except Exception as e2:
                if verbose:
                    print(f"Error creating fallback CPD for {node} in intermediate network: {e2}")
    
    if verbose:
        print(f"After removing independent variables: {len(intermediate_bn.nodes())} nodes, {intermediate_bn.number_of_edges()} edges")
    
    # Step 4: Find barren nodes in the intermediate network (after independent vars removed)
    barren_vars = set()
    identify_barren_nodes(intermediate_bn, target_nodes, evidence_nodes, barren_vars)
    if verbose:
        print(f"Found {len(barren_vars)} barren variables: {sorted(barren_vars)}")
    
    # Step 5: Create final reduced network (after removing both independent and barren nodes)
    vars_to_remove = independent_vars | barren_vars
    vars_to_keep = set(bn.nodes()) - vars_to_remove
    
    if verbose:
        print(f"Removing {len(vars_to_remove)} variables (independent: {len(independent_vars)}, barren: {len(barren_vars)}), keeping {len(vars_to_keep)} variables")
        print(f"Variables to keep: {sorted(vars_to_keep)}")
    
    # Create the final reduced network
    # We can build it from the intermediate network (just remove barren nodes from it)
    reduced_bn = DiscreteBayesianNetwork()
    reduced_bn.add_nodes_from(vars_to_keep)
    
    # Add edges from intermediate network (which already has independent vars removed)
    # Only include edges connecting nodes that remain after removing barren nodes
    for edge in intermediate_bn.edges():
        u, v = edge
        if u in vars_to_keep and v in vars_to_keep:
            reduced_bn.add_edge(u, v)
    
    # Copy CPDs for kept variables from the intermediate network
    # Since intermediate network already has independent vars removed and CPDs adjusted,
    # we just need to handle any additional parents that were removed as barren nodes
    for node in vars_to_keep:
        cpd_added = False
        try:
            # Get CPD from intermediate network (which already has independent vars handled)
            cpd = intermediate_bn.get_cpds(node)
            parents = list(cpd.variables)
            parents.remove(node)  # Remove the node itself from parents list
            
            # Check if all parents are kept (after removing barren nodes)
            if all(p in vars_to_keep for p in parents):
                reduced_bn.add_cpds(cpd)
                cpd_added = True
            else:
                # Some parents were removed - create marginal CPD
                # Get the kept parents
                kept_parents = [p for p in parents if p in vars_to_keep]
                
                # Get cardinalities for all parents
                intermediate_card = intermediate_bn.get_cardinality()
                node_card = intermediate_card[node]
                
                # Get the CPD values - shape is [node_card, parent1_card, parent2_card, ...]
                # The order matches cpd.variables: [node, parent1, parent2, ...]
                cpd_values = cpd.values.copy()
                cpd_variable_order = list(cpd.variables)  # [node, parent1, parent2, ...]
                
                # Find indices of removed parents (skip index 0 which is the node)
                axes_to_sum = []
                for i, var in enumerate(cpd_variable_order[1:], start=1):  # start=1 because 0 is the node
                    if var not in vars_to_keep:
                        axes_to_sum.append(i)
                
                # Sum over removed parent dimensions
                if axes_to_sum:
                    marginal_values = np.sum(cpd_values, axis=tuple(axes_to_sum))
                else:
                    marginal_values = cpd_values
                
                # Get state names from intermediate CPD, preserving only for kept variables
                intermediate_state_names = cpd.state_names
                state_names = {node: intermediate_state_names[node]}
                for parent in kept_parents:
                    if parent in intermediate_state_names:
                        state_names[parent] = intermediate_state_names[parent]
                
                # Now handle the result based on whether we have kept parents
                if len(kept_parents) == 0:
                    # No parents kept - create marginal distribution
                    marginal_values = marginal_values.flatten()
                    # Normalize to ensure it sums to 1
                    if marginal_values.sum() > 0:
                        marginal_values = marginal_values / marginal_values.sum()
                    else:
                        # If sum is zero, use uniform
                        marginal_values = np.ones(node_card) / node_card
                    marginal_cpd = TabularCPD(variable=node, variable_card=node_card, 
                                             values=marginal_values.reshape(-1, 1),
                                             state_names=state_names)
                    reduced_bn.add_cpds(marginal_cpd)
                    cpd_added = True
                else:
                    # We have some kept parents - need to reorder dimensions
                    # After summing, remaining dimensions are [node, kept_parent1, kept_parent2, ...]
                    # but in the order they appeared in the original CPD
                    # We need to reorder to match kept_parents order
                    
                    # Find current positions of kept parents in the summed array
                    # Original positions of kept parents (in cpd_variable_order)
                    kept_parent_original_positions = [cpd_variable_order.index(p) for p in kept_parents]
                    
                    # After summing, some dimensions were removed, so positions shifted
                    # Build new order: [node(0), then kept parents in desired order]
                    # The node is always at position 0
                    desired_order = [0]  # Start with node
                    
                    # For each kept parent in desired order, find its current position
                    # after summing (which is its original position minus number of removed parents before it)
                    removed_before = [p for p in cpd_variable_order[1:] if p not in vars_to_keep]
                    
                    for kept_parent in kept_parents:
                        orig_pos = cpd_variable_order.index(kept_parent)
                        # Count how many removed parents came before this position
                        num_removed_before = sum(1 for i, v in enumerate(cpd_variable_order[1:orig_pos], 1) 
                                                   if v not in vars_to_keep)
                        current_pos = orig_pos - num_removed_before
                        desired_order.append(current_pos)
                    
                    # Transpose to reorder dimensions
                    marginal_values = np.transpose(marginal_values, axes=desired_order)
                    
                    # Get cardinalities for kept parents (in the order we want)
                    kept_parents_cards = [intermediate_card[p] for p in kept_parents]
                    
                    # Normalize across the node dimension for each parent configuration
                    marginal_values = marginal_values / marginal_values.sum(axis=0, keepdims=True)
                    
                    marginal_cpd = TabularCPD(variable=node, variable_card=node_card,
                                             values=marginal_values,
                                             evidence=kept_parents,
                                             evidence_card=kept_parents_cards,
                                             state_names=state_names)
                    reduced_bn.add_cpds(marginal_cpd)
                    cpd_added = True
        except Exception as e:
            if verbose:
                print(f"Warning: Could not copy CPD for {node}: {e}")
                import traceback
                traceback.print_exc()
        
        # If we didn't add a CPD (either because of exception or other issue), create fallback
        if not cpd_added:
            try:
                node_card = intermediate_bn.get_cardinality()[node]
                uniform_values = np.ones((node_card, 1)) / node_card
                # Try to get state names from intermediate network or original BN
                state_names = {}
                try:
                    # First try from intermediate network
                    intermediate_cpd = intermediate_bn.get_cpds(node)
                    if node in intermediate_cpd.state_names:
                        state_names = {node: intermediate_cpd.state_names[node]}
                except:
                    try:
                        # Fallback to original BN
                        original_cpd = bn.get_cpds(node)
                        if node in original_cpd.state_names:
                            state_names = {node: original_cpd.state_names[node]}
                    except:
                        pass  # If we can't get state names, create CPD without them
                
                fallback_cpd = TabularCPD(variable=node, variable_card=node_card,
                                          values=uniform_values,
                                          state_names=state_names if state_names else None)
                reduced_bn.add_cpds(fallback_cpd)
                if verbose:
                    print(f"Added uniform fallback CPD for {node}")
            except Exception as e2:
                if verbose:
                    print(f"Error creating fallback CPD for {node}: {e2}")
                raise
    
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
        try:
            induced_width = ve.induced_width(complete_elim_order)
        except ValueError:
            # Handle case where induced graph has no cliques (empty graph or isolated nodes)
            # This happens when the network has no edges or all nodes are disconnected
            # In this case, the induced width is 0 (no fill-in edges needed)
            induced_width = 0
            if verbose:
                print("Warning: Induced graph has no cliques, setting induced width to 0")
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


def main():
    """
    Test the compute_query_complexity function with a simple example.
    """
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        import numpy as np
    except ImportError:
        print("Error: pgmpy is required to run this test. Install with: pip install pgmpy")
        return
    
    print("=" * 80)
    print("Testing compute_query_complexity function")
    print("=" * 80)
    print()
    
    # Create a simple test Bayesian network
    # Structure: A -> B -> C, A -> D, B -> E
    # This gives us a small network where we can test independence and barren nodes
    print("Creating test Bayesian network...")
    model = DiscreteBayesianNetwork([
        ('A', 'B'),
        ('B', 'C'),
        ('A', 'D'),
        ('B', 'E'),
    ])
    
    # Add CPDs
    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.6], [0.4]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.8, 0.3], [0.2, 0.7]],
                       evidence=['A'], evidence_card=[2])
    cpd_c = TabularCPD(variable='C', variable_card=2,
                       values=[[0.9, 0.2], [0.1, 0.8]],
                       evidence=['B'], evidence_card=[2])
    cpd_d = TabularCPD(variable='D', variable_card=2,
                       values=[[0.7, 0.4], [0.3, 0.6]],
                       evidence=['A'], evidence_card=[2])
    cpd_e = TabularCPD(variable='E', variable_card=2,
                       values=[[0.5, 0.3], [0.5, 0.7]],
                       evidence=['B'], evidence_card=[2])
    
    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e)
    model.check_model()
    
    print(f"Created network with {len(model.nodes())} nodes: {sorted(model.nodes())}")
    print(f"Network edges: {list(model.edges())}")
    print()
    
    # Test Case 1: Query C with evidence on A
    print("=" * 80)
    print("Test Case 1: Query C given evidence A")
    print("=" * 80)
    target_nodes = ['C']
    evidence_nodes = ['A']
    
    result1 = compute_query_complexity(model, target_nodes, evidence_nodes, verbose=True)
    
    print("\n" + "=" * 80)
    print("Key Results:")
    print("=" * 80)
    print(f"Original variables: {result1['original_num_vars']}")
    print(f"Independent variables removed: {result1['num_independent_vars']}")
    print(f"Barren variables removed: {result1['num_barren_vars']}")
    print(f"Total variables removed: {result1['num_vars_removed']}")
    print(f"Reduced network variables: {result1['reduced_num_vars']}")
    print(f"Induced width: {result1['induced_width']}")
    print(f"Total cost: {result1['total_cost']:,}")
    print(f"Max factor size: {result1['max_factor_size']:,}")
    print()
    
    # Test Case 2: Query C with no evidence (should have different independent/barren nodes)
    print("=" * 80)
    print("Test Case 2: Query C with no evidence")
    print("=" * 80)
    target_nodes = ['C']
    evidence_nodes = []
    
    result2 = compute_query_complexity(model, target_nodes, evidence_nodes, verbose=True)
    
    print("\n" + "=" * 80)
    print("Key Results:")
    print("=" * 80)
    print(f"Original variables: {result2['original_num_vars']}")
    print(f"Independent variables removed: {result2['num_independent_vars']}")
    print(f"Barren variables removed: {result2['num_barren_vars']}")
    print(f"Total variables removed: {result2['num_vars_removed']}")
    print(f"Reduced network variables: {result2['reduced_num_vars']}")
    print(f"Induced width: {result2['induced_width']}")
    print(f"Total cost: {result2['total_cost']:,}")
    print(f"Max factor size: {result2['max_factor_size']:,}")
    print()
    
    # Test Case 3: Query A with evidence on C (reverse direction)
    print("=" * 80)
    print("Test Case 3: Query A given evidence C")
    print("=" * 80)
    target_nodes = ['A']
    evidence_nodes = ['C']
    
    result3 = compute_query_complexity(model, target_nodes, evidence_nodes, verbose=True)
    
    print("\n" + "=" * 80)
    print("Key Results:")
    print("=" * 80)
    print(f"Original variables: {result3['original_num_vars']}")
    print(f"Independent variables removed: {result3['num_independent_vars']}")
    print(f"Barren variables removed: {result3['num_barren_vars']}")
    print(f"Total variables removed: {result3['num_vars_removed']}")
    print(f"Reduced network variables: {result3['reduced_num_vars']}")
    print(f"Induced width: {result3['induced_width']}")
    print(f"Total cost: {result3['total_cost']:,}")
    print(f"Max factor size: {result3['max_factor_size']:,}")
    print()
    
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()