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
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def generate_base_dag_configs(
    n_nodes_list: List[int] = [7, 11, 15],
    treewidths: List[int] = [2, 3, 4], 
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
        treewidths: List of target treewidths  
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
        >>> configs = generate_base_dag_configs(
        ...     n_nodes_list=[7, 11], 
        ...     treewidths=[2, 3],
        ...     samples_per_config=2
        ... )
        >>> print(f"Generated {len(configs)} base DAG configurations")
        Generated 8 base DAG configurations
        >>> print(configs[0])
        {'dag_id': 'dag_0001', 'n_nodes': 7, 'target_treewidth': 2, ...}
    """
    configs = []
    dag_counter = 1
    
    # Calculate total combinations for progress bar
    valid_combinations = []
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
