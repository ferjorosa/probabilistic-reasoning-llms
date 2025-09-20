"""
Naming Variant Generation for Probabilistic Reasoning Experiments

This module handles the generation of naming variants for DAG configurations.
It takes base DAG configurations and applies different naming strategies to 
create variants with identical topology but different node names.

This enables clean ablation testing where the graph structure remains constant
while node naming varies (simple, confusing, semantic, etc.).

Author: Generated for LLM probabilistic reasoning research
"""

from typing import List, Dict, Any
from tqdm import tqdm

from src.graph_generation import generate_dag_with_treewidth


def generate_naming_variants(
    dag_configs: List[Dict[str, Any]],
    naming_strategies: List[str] = ['simple', 'confusing', 'semantic']
) -> List[Dict[str, Any]]:
    """
    Generate naming variants for each base DAG configuration.
    
    For each base DAG configuration, this function generates the same DAG structure
    with different naming strategies. This enables clean ablation testing where
    the graph topology is identical but node names vary.
    
    Args:
        dag_configs: List of base DAG configurations from generate_base_dag_configs()
        naming_strategies: List of naming strategies to apply to each DAG
        
    Returns:
        List of dictionaries, each containing:
        - dag_id: Same as the base DAG (links variants to same structure)
        - naming_variant_id: Unique identifier for this DAG+naming combination
        - naming_strategy: The naming strategy used ('simple', 'confusing', 'semantic')
        - dag: The actual NetworkX DiGraph with applied naming
        - achieved_treewidth: Actual treewidth achieved
        - All original dag_config fields (n_nodes, target_treewidth, etc.)
        
    Example:
        >>> base_configs = generate_base_dag_configs(n_nodes_list=[7], treewidths=[2], samples_per_config=1)
        >>> naming_variants = generate_naming_variants(base_configs)
        >>> print(f"Generated {len(naming_variants)} naming variants")
        Generated 3 naming variants  # 1 base config * 3 naming strategies
        >>> # All variants with same dag_id have identical graph structure
        >>> same_structure = [v for v in naming_variants if v['dag_id'] == 'dag_0001']
        >>> print(f"DAG dag_0001 has {len(same_structure)} naming variants")
        DAG dag_0001 has 3 naming variants
    """
    naming_variants = []
    variant_counter = 1
    
    total_variants = len(dag_configs) * len(naming_strategies)
    
    # Generate naming variants with progress bar
    with tqdm(total=total_variants, desc="Generating naming variants") as pbar:
        for dag_config in dag_configs:
            # Generate the same DAG structure with different naming strategies
            for naming_strategy in naming_strategies:
                try:
                    # Generate DAG with the same structural parameters and seed, but different naming
                    dag, achieved_treewidth, metadata = generate_dag_with_treewidth(
                        n_nodes=dag_config['n_nodes'],
                        target_treewidth=dag_config['target_treewidth'],
                        dag_method=dag_config['dag_method'],
                        node_naming=naming_strategy,
                        seed=dag_config['structural_seed']  # Same seed = same structure!
                    )
                    
                    # Create naming variant entry
                    variant = {
                        # Link back to same DAG structure
                        'dag_id': dag_config['dag_id'],
                        
                        # Unique identifier for this DAG+naming combination
                        'naming_variant_id': f'variant_{variant_counter:04d}',
                        
                        # Naming information
                        'naming_strategy': naming_strategy,
                        
                        # The actual DAG with applied naming
                        'dag': dag,
                        'achieved_treewidth': achieved_treewidth,
                        'generation_metadata': metadata,
                        
                        # Copy all original DAG configuration fields
                        **dag_config
                    }
                    
                    naming_variants.append(variant)
                    variant_counter += 1
                    
                except Exception as e:
                    # Still update progress bar even on failure
                    tqdm.write(f"✗ Failed to generate {naming_strategy} variant for {dag_config['dag_id']}: {e}")
                    continue
                finally:
                    pbar.update(1)
    
    return naming_variants
