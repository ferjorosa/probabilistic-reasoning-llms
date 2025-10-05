"""
DAG Configuration Generation for Probabilistic Reasoning Experiments

This module handles the generation of base DAG configurations with different 
structural parameters. It provides clean separation between:
- DAG structure generation (topology)
- Naming strategy application  
- CPT parameter configuration

The configurations generated here serve as the foundation for systematic
ablation studies on Bayesian network properties.

Author: Generated for LLM probabilistic reasoning research
"""

import itertools
import math
from typing import List, Dict, Any
from tqdm import tqdm


def get_proportional_treewidths(n_nodes: int, treewidth_fractions: List[float]) -> List[int]:
    """
    Generate treewidths as proportions of the number of nodes.
    
    Args:
        n_nodes: Number of nodes in the network
        treewidth_fractions: List of fractions (0.0 to 1.0) to multiply by n_nodes
        
    Returns:
        List of treewidth values computed as fractions of n_nodes
        
    Example:
        >>> get_proportional_treewidths(20, [0.1, 0.2, 0.3, 0.5])
        [2, 4, 6, 10]
        >>> get_proportional_treewidths(50, [0.1, 0.2, 0.3, 0.5])
        [5, 10, 15, 25]
    """
    min_tw = 1 # tree structure
    max_tw = n_nodes - 1
    
    treewidths = []
    for fraction in treewidth_fractions:
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"Treewidth fraction {fraction} must be between 0.0 and 1.0")
        
        tw = max(min_tw, int(n_nodes * fraction))
        tw = min(tw, max_tw)  # Ensure it doesn't exceed maximum possible
        treewidths.append(tw)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_treewidths = []
    for tw in treewidths:
        if tw not in seen:
            seen.add(tw)
            unique_treewidths.append(tw)
    
    return unique_treewidths


def generate_base_dag_configs(
    n_nodes_list: List[int] = [7, 11, 15],
    treewidths: List[int] = None, 
    treewidth_fractions: List[float] = None,
    dag_methods: List[str] = ['random'],
    samples_per_config: int = 2,
    base_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate base DAG configurations with unique structural parameters.
    
    Each configuration represents a unique DAG topology that will later be
    used with different naming strategies for ablation testing.
    
    Args:
        n_nodes_list: List of node counts to test
        treewidths: List of specific treewidth values (mutually exclusive with treewidth_fractions)
        treewidth_fractions: List of fractions (0.0-1.0) for proportional scaling (mutually exclusive with treewidths)
        dag_methods: List of DAG generation methods ('random', 'topological', etc.)
        samples_per_config: Number of different DAG structures per parameter combination
        base_seed: Base seed for reproducible DAG structure generation
        
    Returns:
        List of dictionaries, each containing:
        - dag_id: Unique identifier for this DAG structure
        - n_nodes: Number of nodes
        - target_treewidth: Target treewidth
        - dag_method: DAG generation method
        - structural_seed: Seed used for generating this DAG structure
        - sample_idx: Sample index within this parameter combination
        
    Example:
        >>> # Using specific treewidth values
        >>> configs = generate_base_dag_configs(
        ...     n_nodes_list=[10, 20], 
        ...     treewidths=[2, 4, 6],
        ...     samples_per_config=2
        ... )
        
        >>> # Using proportional scaling (recommended for varying node counts)
        >>> configs = generate_base_dag_configs(
        ...     n_nodes_list=[10, 30, 50], 
        ...     treewidth_fractions=[0.1, 0.2, 0.3, 0.5],
        ...     samples_per_config=2
        ... )
        >>> # For n_nodes=10: treewidths=[2, 2, 3, 5] (after deduplication: [2, 3, 5])
        >>> # For n_nodes=50: treewidths=[5, 10, 15, 25]
    """
    # Validate arguments - exactly one of treewidths or treewidth_fractions must be provided
    if treewidths is not None and treewidth_fractions is not None:
        raise ValueError("Cannot specify both 'treewidths' and 'treewidth_fractions'. Use one or the other.")
    
    if treewidths is None and treewidth_fractions is None:
        raise ValueError("Need to specify either 'treewidths' or 'treewidth_fractions'")
    
    configs = []
    dag_counter = 1
    
    # Calculate total combinations for progress bar
    valid_combinations = []
    
    if treewidth_fractions is not None:
        # Use proportional scaling
        for n_nodes, dag_method in itertools.product(n_nodes_list, dag_methods):
            proportional_tws = get_proportional_treewidths(n_nodes, treewidth_fractions)
            for treewidth in proportional_tws:
                valid_combinations.append((n_nodes, treewidth, dag_method))
    else:
        # Use provided specific treewidths
        for n_nodes, treewidth, dag_method in itertools.product(n_nodes_list, treewidths, dag_methods):
            if treewidth < n_nodes:  # Skip invalid combinations
                valid_combinations.append((n_nodes, treewidth, dag_method))
    
    total_configs = len(valid_combinations) * samples_per_config
    
    # Generate all combinations of structural parameters with progress bar
    with tqdm(total=total_configs, desc="Generating DAG configs") as pbar:
        for n_nodes, treewidth, dag_method in valid_combinations:
            # Generate multiple samples for each parameter combination
            for sample_idx in range(samples_per_config):
                # Create deterministic but well-separated seed for this configuration
                structural_seed = base_seed + dag_counter * 1000 + sample_idx * 17
                
                config = {
                    'dag_id': f'dag_{dag_counter:04d}',
                    'n_nodes': n_nodes,
                    'target_treewidth': treewidth,
                    'dag_method': dag_method,
                    'structural_seed': structural_seed,
                    'sample_idx': sample_idx
                }
                configs.append(config)
                dag_counter += 1
                pbar.update(1)
    
    return configs
