"""
Graph Generation for Probabilistic Reasoning Experiments

This module provides tools for generating graphs and DAGs with controlled treewidth
for evaluating the probabilistic reasoning capabilities of Large Language Models (LLMs).

The key insight is that treewidth correlates with exact inference hardness in Bayesian
networks, making it a crucial parameter for systematic evaluation of LLM performance
on probabilistic reasoning tasks.

REPRODUCIBILITY NOTES:
- Seeds are set for both numpy.random and Python's random module
- Reproducibility is guaranteed on the same system with same library versions
- Cross-platform reproducibility may vary due to:
  * Different Python versions (3.8 vs 3.12)
  * Different NetworkX versions (3.x vs 4.x)  
  * Different NumPy versions (1.x vs 2.x)
  * Different architectures (x86_64 vs ARM64)
- For maximum reproducibility, use identical environments (Docker recommended)

Main Functions:
    - generate_node_names(): Creates node names using different strategies
    - generate_graph_with_target_treewidth(): Creates graphs with approximate treewidth
    - undirected_to_dag(): Converts undirected graphs to DAGs using various methods
    - generate_dag_with_treewidth(): Main function combining graph generation and DAG conversion
    - analyze_graph_properties(): Utility function for analyzing generated graphs

Key Design Decisions:
    - Use 'random' or 'topological' DAG conversion methods to preserve treewidth
    - Avoid 'bfs' and 'dfs' methods as they create spanning trees (treewidth = 1)
    - Iterative method provides approximate treewidth with diverse structures
    - Support multiple node naming strategies to test LLM robustness

Example Usage:
    >>> # Generate a single DAG with simple node names
    >>> dag, achieved_tw, metadata = generate_dag_with_treewidth(
    ...     n_nodes=10, target_treewidth=3, node_naming='simple'
    ... )
    >>> print(f"Nodes: {list(dag.nodes())[:3]}")  # ['V0', 'V1', 'V2']
    
    >>> # Generate with confusing names to test LLM robustness
    >>> dag, achieved_tw, metadata = generate_dag_with_treewidth(
    ...     n_nodes=8, target_treewidth=2, node_naming='confusing'
    ... )
    >>> print(f"Confusing nodes: {list(dag.nodes())[:2]}")  # ['X_7a4f2b', 'Q_9c1e8d']

Author: Generated for LLM probabilistic reasoning research
"""

import networkx as nx
import numpy as np
import random
import string
from typing import List, Tuple, Optional, Dict, Any
from networkx.algorithms.approximation import treewidth


def generate_node_names(n_nodes: int, 
                       strategy: str = 'simple',
                       seed: Optional[int] = None) -> List[str]:
    """
    Generate node names using different strategies for LLM experiments.
    
    Different naming strategies can help test how node names affect LLM
    probabilistic reasoning performance.
    
    Args:
        n_nodes: Number of node names to generate
        strategy: Naming strategy ('simple', 'confusing', 'semantic', 'mixed')
        seed: Random seed for reproducible name generation
        
    Returns:
        List of node names
        
    Strategies:
        - 'simple': V0, V1, V2, ... (clear and systematic)
        - 'confusing': X_445aFa, S_af3a34, ... (random alphanumeric)
        - 'semantic': meaningful names like 'Rain', 'Sprinkler', 'WetGrass'
        - 'mixed': combination of different strategies
        
    Example:
        >>> names = generate_node_names(3, 'simple')
        >>> print(names)
        ['V0', 'V1', 'V2']
        
        >>> names = generate_node_names(3, 'confusing', seed=42)
        >>> print(names)
        ['X_7a4f2b', 'Q_9c1e8d', 'Z_3b6a9f']
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if strategy == 'simple':
        return [f'V{i}' for i in range(n_nodes)]
    
    elif strategy == 'confusing':
        names = []
        prefixes = list(string.ascii_uppercase)
        for i in range(n_nodes):
            prefix = np.random.choice(prefixes)
            # Generate random alphanumeric suffix
            chars = string.ascii_lowercase + string.digits
            suffix = ''.join(np.random.choice(list(chars), size=6))
            names.append(f'{prefix}_{suffix}')
        return names
    
    elif strategy == 'semantic':
        # Common semantic names for Bayesian networks
        semantic_names = [
            'Rain', 'Sprinkler', 'WetGrass', 'Cloudy', 'Season',
            'Temperature', 'Humidity', 'Wind', 'Pressure', 'Visibility',
            'Traffic', 'Accident', 'Weather', 'Road', 'Time',
            'Age', 'Gender', 'Income', 'Education', 'Health',
            'Smoking', 'Exercise', 'Diet', 'Stress', 'Sleep',
            'Disease', 'Symptom', 'Treatment', 'Recovery', 'Test',
            'Cause', 'Effect', 'Factor', 'Outcome', 'Risk',
            'Signal', 'Noise', 'Data', 'Model', 'Prediction'
        ]
        
        if n_nodes <= len(semantic_names):
            return list(np.random.choice(semantic_names, size=n_nodes, replace=False))
        else:
            # If we need more names than available, cycle through and add numbers
            base_names = list(np.random.choice(semantic_names, size=len(semantic_names), replace=False))
            names = base_names.copy()
            counter = 1
            while len(names) < n_nodes:
                for base_name in base_names:
                    if len(names) >= n_nodes:
                        break
                    names.append(f'{base_name}{counter}')
                counter += 1
            return names[:n_nodes]
    
    elif strategy == 'mixed':
        # Mix of different strategies
        names = []
        strategies = ['simple', 'confusing', 'semantic']
        
        for i in range(n_nodes):
            chosen_strategy = np.random.choice(strategies)
            if chosen_strategy == 'simple':
                names.append(f'V{i}')
            elif chosen_strategy == 'confusing':
                prefix = np.random.choice(list(string.ascii_uppercase))
                chars = string.ascii_lowercase + string.digits
                suffix = ''.join(np.random.choice(list(chars), size=4))
                names.append(f'{prefix}_{suffix}')
            else:  # semantic
                semantic_options = ['Factor', 'Node', 'Variable', 'Element', 'Component']
                base = np.random.choice(semantic_options)
                names.append(f'{base}{i}')
        
        return names
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'simple', 'confusing', 'semantic', or 'mixed'")


def relabel_graph_nodes(G: nx.Graph, 
                       node_names: List[str]) -> nx.Graph:
    """
    Relabel graph nodes with custom names.
    
    Args:
        G: NetworkX graph with default node labels (0, 1, 2, ...)
        node_names: List of new node names (must match number of nodes)
        
    Returns:
        New graph with relabeled nodes
        
    Raises:
        ValueError: If number of names doesn't match number of nodes
    """
    if len(node_names) != G.number_of_nodes():
        raise ValueError(f"Number of names ({len(node_names)}) must match number of nodes ({G.number_of_nodes()})")
    
    # Create mapping from old labels to new names
    old_nodes = sorted(G.nodes())  # Ensure consistent ordering
    mapping = {old_nodes[i]: node_names[i] for i in range(len(old_nodes))}
    
    return nx.relabel_nodes(G, mapping)


def generate_graph_with_target_treewidth(n_nodes: int, 
                                       target_treewidth: int, 
                                       max_iterations: int = 1000,
                                       seed: Optional[int] = None) -> Tuple[nx.Graph, int, int]:
    """
    Generate a graph trying to achieve a specific treewidth using iterative approach.
    
    This method uses a heuristic approach: starts with a random tree (treewidth=1)
    and iteratively adds edges while monitoring treewidth approximations until
    the target is reached or max_iterations is exceeded.
    
    **Advantages:**
    - More diverse graph structures than deterministic methods
    - Can produce more "natural" looking networks
    - Good for testing robustness across different topologies
    
    **Disadvantages:**
    - Only approximate treewidth (uses approximation algorithms)
    - May not reach exact target treewidth
    - Success depends on target difficulty
    
    Args:
        n_nodes: Number of nodes in the graph
        target_treewidth: Desired treewidth (will try to approximate this)
        max_iterations: Maximum number of generation attempts (default: 1000)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_graph, achieved_treewidth, difference_from_target)
        - best_graph: NetworkX Graph with closest achieved treewidth
        - achieved_treewidth: Actual treewidth of the returned graph
        - difference_from_target: |achieved_treewidth - target_treewidth|
    """
    # Set seed once at the beginning for reproducible results
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    best_graph = None
    best_treewidth = float('inf')
    best_diff = float('inf')
    
    for iteration in range(max_iterations):
        # Start with a random tree (treewidth = 1)
        G = nx.random_labeled_tree(n_nodes)
        
        # Iteratively add edges to increase treewidth
        max_edge_additions = n_nodes * (n_nodes - 1) // 2 - (n_nodes - 1)  # Max possible edges - tree edges
        
        for _ in range(max_edge_additions):
            # Compute current treewidth
            current_width, _ = treewidth.treewidth_min_degree(G)
            
            if current_width >= target_treewidth:
                break
            
            # Try to add a random edge
            nodes = list(G.nodes())
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                u, v = np.random.choice(nodes, 2, replace=False)
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    break
                attempts += 1
            
            if attempts >= 50:  # No more edges can be added
                break
        
        # Check final treewidth
        final_width, _ = treewidth.treewidth_min_degree(G)
        diff = abs(final_width - target_treewidth)
        
        if diff < best_diff:
            best_diff = diff
            best_treewidth = final_width
            best_graph = G.copy()
        
        if diff == 0:  # Exact match found
            break
    
    return best_graph, best_treewidth, best_diff


def undirected_to_dag(G: nx.Graph, 
                     method: str = 'random',
                     root: Optional[int] = None,
                     seed: Optional[int] = None) -> nx.DiGraph:
    """
    Convert an undirected graph to a DAG using various methods.
    
    **IMPORTANT:** Choice of method significantly affects treewidth preservation:
    - 'bfs'/'dfs': Create spanning trees → treewidth = 1 (Naive Bayes structure)
    - 'random'/'topological': Preserve all edges → maintain original treewidth
    
    **Method Details:**
    - 'bfs': Breadth-first spanning tree (good for hierarchical structures)
    - 'dfs': Depth-first spanning tree (good for deep hierarchical structures)  
    - 'random': Random edge orientation while avoiding cycles (preserves complexity)
    - 'topological': Random node ordering with consistent edge directions (preserves complexity)
    
    **Recommendation for Treewidth Experiments:**
    Use 'random' or 'topological' to preserve the treewidth of the original graph.
    Use 'bfs' or 'dfs' only if you specifically want simple tree structures.
    
    Args:
        G: Undirected NetworkX graph to convert
        method: Conversion method ('bfs', 'dfs', 'random', 'topological')
                Default: 'random' (recommended for treewidth preservation)
        root: Root node for tree-based methods ('bfs'/'dfs'). Chosen randomly if None.
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX DiGraph that is a DAG
        
    Raises:
        ValueError: If method is not one of the supported options
        
    Example:
        >>> # Preserve treewidth (recommended for experiments)
        >>> dag = undirected_to_dag(graph, method='random', seed=42)
        
        >>> # Create simple tree structure  
        >>> tree_dag = undirected_to_dag(graph, method='bfs', root=0)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if not nx.is_connected(G):
        # Handle each connected component separately
        dag_edges = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            sub_dag = undirected_to_dag(subgraph, method, root, seed)
            dag_edges.extend(sub_dag.edges())
        return nx.DiGraph(dag_edges)
    
    dag_edges = []
    
    if method == 'bfs':
        if root is None:
            root = np.random.choice(list(G.nodes()))
        # Use BFS to create a directed tree
        for edge in nx.bfs_edges(G, root):
            dag_edges.append(edge)
            
    elif method == 'dfs':
        if root is None:
            root = np.random.choice(list(G.nodes()))
        # Use DFS to create a directed tree
        for edge in nx.dfs_edges(G, root):
            dag_edges.append(edge)
            
    elif method == 'random':
        # Randomly orient edges while maintaining DAG property
        edges = list(G.edges())
        np.random.shuffle(edges)
        
        temp_dag = nx.DiGraph()
        temp_dag.add_nodes_from(G.nodes())
        
        for u, v in edges:
            # Try both orientations and pick one that doesn't create a cycle
            for edge_candidate in [(u, v), (v, u)]:
                temp_dag.add_edge(*edge_candidate)
                if nx.is_directed_acyclic_graph(temp_dag):
                    dag_edges.append(edge_candidate)
                    break
                else:
                    temp_dag.remove_edge(*edge_candidate)
                    
    elif method == 'topological':
        # Create a random topological ordering and orient edges accordingly
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        node_order = {node: i for i, node in enumerate(nodes)}
        
        for u, v in G.edges():
            if node_order[u] < node_order[v]:
                dag_edges.append((u, v))
            else:
                dag_edges.append((v, u))
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bfs', 'dfs', 'random', or 'topological'")
    
    return nx.DiGraph(dag_edges)


def generate_dag_with_treewidth(n_nodes: int,
                               target_treewidth: int,
                               dag_method: str = 'random',
                               max_iterations: int = 1000,
                               node_naming: str = 'simple',
                               seed: Optional[int] = None) -> Tuple[nx.DiGraph, int, Dict[str, Any]]:
    """
    Generate a DAG with approximately the target treewidth.
    
    This function combines graph generation with DAG conversion to produce
    directed acyclic graphs suitable for Bayesian network experiments.
    
    **Recommended Method Combination:**
    - dag_method='random': Preserves treewidth, diverse DAG structure
    - dag_method='topological': Preserves treewidth, ordered structure  
    
    **WARNING:** Avoid dag_method='bfs' or 'dfs' as they reduce treewidth to 1!
    
    Args:
        n_nodes: Number of nodes in the final DAG
        target_treewidth: Desired treewidth (approximate)
        dag_method: DAG conversion method ('random', 'topological', 'bfs', 'dfs')
                   Default: 'random' (recommended for treewidth preservation)
        max_iterations: Maximum iterations for treewidth search (default: 1000)
        node_naming: Node naming strategy ('simple', 'confusing', 'semantic', 'mixed')
                    Default: 'simple' (V0, V1, V2, ...)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dag, achieved_treewidth, metadata_dict) where:
        - dag: NetworkX DiGraph (the generated DAG)
        - achieved_treewidth: Final treewidth of the DAG's underlying undirected graph
        - metadata_dict: Dictionary with generation details and statistics
        
    Raises:
        ValueError: If target_treewidth >= n_nodes or invalid method names
        
    Example:
        >>> # Generate DAG with approximate treewidth 3 (recommended)
        >>> dag, tw, meta = generate_dag_with_treewidth(
        ...     n_nodes=10, target_treewidth=3, dag_method='random'
        ... )
        >>> print(f"Target: 3, Achieved: {tw}")
        Target: 3, Achieved: 3
        
        >>> # Generate with confusing node names to test LLM robustness
        >>> dag, tw, meta = generate_dag_with_treewidth(
        ...     n_nodes=8, target_treewidth=2, node_naming='confusing'
        ... )
        >>> print(list(dag.nodes())[:3])
        ['X_7a4f2b', 'Q_9c1e8d', 'Z_3b6a9f']
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if target_treewidth >= n_nodes:
        raise ValueError(f"target_treewidth ({target_treewidth}) must be less than n_nodes ({n_nodes})")
    
    metadata = {
        'dag_method': dag_method,
        'target_treewidth': target_treewidth,
        'n_nodes': n_nodes,
        'max_iterations': max_iterations,
        'node_naming': node_naming
    }
    
    # Generate base undirected graph using iterative method
    base_graph, achieved_treewidth, diff = generate_graph_with_target_treewidth(
        n_nodes, target_treewidth, max_iterations, seed=seed
    )
    metadata['treewidth_difference'] = diff
    metadata['exact_treewidth'] = (diff == 0)
    
    # Convert to DAG
    dag = undirected_to_dag(base_graph, dag_method, seed=seed)
    
    # Apply node naming strategy
    if node_naming != 'default':  # 'default' keeps numeric labels 0, 1, 2, ...
        node_names = generate_node_names(n_nodes, node_naming, seed=seed)
        dag = relabel_graph_nodes(dag, node_names)
        metadata['node_names'] = node_names
    
    # Verify final treewidth of the DAG's underlying undirected graph
    final_undirected = dag.to_undirected()
    final_treewidth, _ = treewidth.treewidth_min_degree(final_undirected)
    
    metadata['final_treewidth'] = final_treewidth
    metadata['base_graph_edges'] = base_graph.number_of_edges()
    metadata['dag_edges'] = dag.number_of_edges()
    
    return dag, final_treewidth, metadata


def generate_experimental_dataset(n_nodes_list: List[int],
                                treewidth_list: List[int],
                                n_samples: int = 5,
                                dag_method: str = 'random',
                                max_iterations: int = 1000,
                                node_naming: str = 'simple',
                                base_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a dataset of DAGs for systematic LLM probabilistic reasoning experiments.
    
    This function creates multiple samples for each (n_nodes, treewidth) combination
    using different seeds to ensure structural diversity while maintaining reproducibility.
    
    Args:
        n_nodes_list: List of node counts to test (e.g., [5, 8, 10, 15])
        treewidth_list: List of treewidths to test (e.g., [1, 2, 3, 4])
        n_samples: Number of different structures per (n_nodes, treewidth) combination
        dag_method: DAG conversion method ('random' recommended)
        max_iterations: Maximum iterations for treewidth search per sample
        node_naming: Node naming strategy ('simple', 'confusing', 'semantic', 'mixed')
        base_seed: Base seed for reproducibility (each sample gets base_seed + offset)
        
    Returns:
        List of dictionaries, each containing:
        - 'dag': Generated NetworkX DiGraph
        - 'n_nodes': Number of nodes
        - 'target_treewidth': Target treewidth
        - 'achieved_treewidth': Actual achieved treewidth
        - 'sample_id': Sample number (0 to n_samples-1)
        - 'seed': Seed used for this sample
        - 'metadata': Detailed generation metadata
        
    Example:
        >>> # Generate experimental dataset
        >>> results = generate_experimental_dataset(
        ...     n_nodes_list=[5, 8], 
        ...     treewidth_list=[1, 2],
        ...     n_samples=3,
        ...     base_seed=42
        ... )
        >>> print(f"Generated {len(results)} DAGs")
        Generated 12 DAGs
        
        >>> # Each combination has different structures
        >>> for r in results[:6]:  # First 6 results
        ...     print(f"Nodes: {r['n_nodes']}, TW: {r['target_treewidth']}, "
        ...           f"Sample: {r['sample_id']}, Seed: {r['seed']}")
    """
    results = []
    sample_counter = 0
    
    for n_nodes in n_nodes_list:
        for target_tw in treewidth_list:
            if target_tw >= n_nodes:
                print(f"Skipping treewidth {target_tw} for {n_nodes} nodes (must be < n_nodes)")
                continue
                
            for sample in range(n_samples):
                # Use spaced seeds to ensure good randomness separation
                current_seed = base_seed + sample_counter * 17  # Prime number for good distribution
                
                try:
                    dag, achieved_tw, metadata = generate_dag_with_treewidth(
                        n_nodes, target_tw, dag_method, max_iterations, node_naming, current_seed
                    )
                    
                    result = {
                        'dag': dag,
                        'n_nodes': n_nodes,
                        'target_treewidth': target_tw,
                        'achieved_treewidth': achieved_tw,
                        'sample_id': sample,
                        'seed': current_seed,
                        'metadata': metadata
                    }
                    results.append(result)
                    sample_counter += 1
                    
                except Exception as e:
                    print(f"Error generating DAG with {n_nodes} nodes, treewidth {target_tw}, "
                          f"sample {sample}, seed {current_seed}: {e}")
                    continue
    
    return results


# Utility functions for analysis
def analyze_graph_properties(G: nx.Graph) -> Dict[str, Any]:
    """
    Analyze various structural and complexity properties of a graph.
    
    This utility function computes key properties useful for understanding
    the characteristics of generated graphs, particularly for experimental
    analysis of LLM probabilistic reasoning performance.
    
    Args:
        G: NetworkX Graph or DiGraph to analyze
        
    Returns:
        Dictionary containing graph properties:
        - 'n_nodes': Number of nodes
        - 'n_edges': Number of edges  
        - 'is_connected': Whether the underlying undirected graph is connected
        - 'density': Edge density (ratio of actual to possible edges)
        - 'treewidth': Approximate treewidth (None if computation fails)
        - 'tree_decomposition_size': Number of nodes in tree decomposition
        - 'is_dag': Whether the graph is a DAG (DiGraph only)
        - 'max_path_length': Longest path length (DAG only)
        - 'topological_levels': Number of topological levels (DAG only)
        
    Example:
        >>> dag, _, _ = generate_dag_with_treewidth(10, 3)
        >>> props = analyze_graph_properties(dag)
        >>> print(f"Nodes: {props['n_nodes']}, Treewidth: {props['treewidth']}")
        Nodes: 10, Treewidth: 3
        >>> print(f"Is DAG: {props['is_dag']}, Levels: {props['topological_levels']}")
        Is DAG: True, Levels: 4
    """
    undirected_G = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    
    properties = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'is_connected': nx.is_connected(undirected_G),
        'density': nx.density(G),
    }
    
    # Compute treewidth
    try:
        width, decomposition = treewidth.treewidth_min_degree(undirected_G)
        properties['treewidth'] = width
        properties['tree_decomposition_size'] = len(decomposition.nodes())
    except Exception as e:
        properties['treewidth'] = None
        properties['treewidth_error'] = str(e)
    
    # DAG-specific properties
    if isinstance(G, nx.DiGraph):
        properties['is_dag'] = nx.is_directed_acyclic_graph(G)
        if properties['is_dag']:
            properties['max_path_length'] = nx.dag_longest_path_length(G)
            properties['topological_levels'] = len(list(nx.topological_generations(G)))
    
    return properties