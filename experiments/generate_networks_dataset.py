"""
Clean Network Dataset Generation for Probabilistic Reasoning Experiments

This module provides a structured approach to generating Bayesian network datasets
for LLM probabilistic reasoning experiments with proper separation of concerns:

1. DAG Structure Generation: Creates base DAG topologies with different structural parameters
2. Naming Variants: Applies different naming strategies to the same DAG structure  
3. CPT Variants: Generates different CPT configurations for each DAG+naming combination
4. Clean Export: Outputs organized DataFrame with proper dag_id and model_id for ablation studies

Key Design Principles:
- Same seed + same structural parameters + different naming = Same topology, different names
- Clear separation between DAG structure (dag_id) and model variants (model_id)
- Reproducible generation with proper seed management
- Easy ablation testing on naming strategies and CPT parameters

Author: Generated for LLM probabilistic reasoning research
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import itertools

from src.graph_generation import generate_dag_with_treewidth
from src.bn_generation import generate_variants_for_dag


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
    
    # Generate all combinations of structural parameters
    for n_nodes, treewidth, dag_method in itertools.product(n_nodes_list, treewidths, dag_methods):
        # Skip invalid combinations
        if treewidth >= n_nodes:
            continue
            
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
    
    return configs


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
        - All oginal dag_config fields (n_nodes, target_treewidth, etc.)
        
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
    
    print(f"Generating naming variants for {len(dag_configs)} base DAG configurations...")
    print(f"Naming strategies: {naming_strategies}")
    print()
    
    for dag_config in dag_configs:
        print(f"Processing {dag_config['dag_id']} ({dag_config['n_nodes']} nodes, treewidth {dag_config['target_treewidth']})...")
        
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
                
                print(f"  ✓ {naming_strategy}: {variant['naming_variant_id']} "
                      f"(nodes: {list(dag.nodes())[:3]}{'...' if len(dag.nodes()) > 3 else ''})")
                
            except Exception as e:
                print(f"  ✗ Failed to generate {naming_strategy} variant: {e}")
                continue
        
        print()
    
    print(f"Successfully generated {len(naming_variants)} naming variants total")
    print(f"Expected: {len(dag_configs)} DAGs × {len(naming_strategies)} naming strategies = {len(dag_configs) * len(naming_strategies)}")
    
    return naming_variants


def generate_cpt_variants(
    naming_variants: List[Dict[str, Any]],
    arity_specs: List[Dict[str, Any]] = [{"type": "range", "min": 2, "max": 3}],
    dirichlet_alphas: List[float] = [0.5, 1.0],
    determinism_fracs: List[float] = [0.0, 0.1],
    variants_per_combo: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate CPT variants for each naming variant with different probability distributions.
    
    For each DAG+naming combination, this function generates multiple Bayesian networks
    with different CPT parameters, storing both string format (for LLM experiments) 
    and array format (for numerical analysis).
    
    Args:
        naming_variants: List of naming variants from generate_naming_variants()
        arity_specs: List of arity strategy specifications
        dirichlet_alphas: List of Dirichlet alpha values for CPT sampling
        determinism_fracs: List of determinism fractions for CPT columns
        variants_per_combo: Number of variants per parameter combination
        
    Returns:
        List of dictionaries, each containing:
        - dag_id: Links back to same DAG structure
        - naming_variant_id: Links back to same DAG+naming combination
        - model_id: Unique identifier for this complete Bayesian network
        - cpds_as_string: Joined string of all CPDs (ready for LLM prompts)
        - cpd_arrays: Dictionary of CPD arrays (ready for numerical analysis)
        - All generation metadata and parameters
        
    Example:
        >>> naming_vars = generate_naming_variants([...])
        >>> cpt_vars = generate_cpt_variants(naming_vars)
        >>> print(f"Generated {len(cpt_vars)} complete Bayesian networks")
        >>> # Each model has both string and array formats
        >>> model = cpt_vars[0]
        >>> print("LLM format:")
        >>> print(model['cpds_as_string'][:200] + "...")
        >>> print("Array format:")
        >>> print(list(model['cpd_arrays'].keys()))
    """
    cpt_variants = []
    model_counter = 1
    
    print(f"Generating CPT variants for {len(naming_variants)} naming variants...")
    print(f"CPT parameter combinations: {len(arity_specs)} arity × {len(dirichlet_alphas)} alpha × {len(determinism_fracs)} determinism × {variants_per_combo} variants")
    total_expected = len(naming_variants) * len(arity_specs) * len(dirichlet_alphas) * len(determinism_fracs) * variants_per_combo
    print(f"Expected total models: {total_expected}")
    print()
    
    for naming_variant in naming_variants:
        print(f"Processing {naming_variant['naming_variant_id']} ({naming_variant['dag_id']}, {naming_variant['naming_strategy']})...")
        
        # Generate all combinations of CPT parameters
        for arity_spec in arity_specs:
            for alpha in dirichlet_alphas:
                for det_frac in determinism_fracs:
                    
                    # Create multiple variants for this parameter combination
                    cpt_configs = []
                    for i in range(variants_per_combo):
                        cpt_configs.append({
                            "arity_strategy": arity_spec,
                            "dirichlet_alpha": alpha,
                            "determinism_fraction": det_frac,
                        })
                    
                    # Generate BN variants using the existing function
                    try:
                        bn_variants = generate_variants_for_dag(
                            naming_variant['dag'], 
                            cpt_configs, 
                            base_seed=naming_variant['structural_seed'] + model_counter * 100
                        )
                        
                        for (bn, bn_meta) in bn_variants:
                            # Extract CPDs in both formats
                            cpd_strings = [str(cpd) for cpd in bn.get_cpds()]
                            cpds_as_string = "\n\n".join(cpd_strings)
                            cpd_arrays = {cpd.variable: cpd.values for cpd in bn.get_cpds()}
                            
                            # Create complete model entry
                            model = {
                                # IDs linking back to DAG structure and naming
                                'dag_id': naming_variant['dag_id'],
                                'naming_variant_id': naming_variant['naming_variant_id'],
                                'model_id': f'model_{model_counter:04d}',
                                
                                # CPT data in both formats
                                'cpds_as_string': cpds_as_string,
                                'cpd_arrays': cpd_arrays,
                                
                                # Generation parameters
                                'cpt_config': {
                                    'arity_strategy': arity_spec,
                                    'dirichlet_alpha': bn_meta['dirichlet_alpha'],
                                    'determinism_fraction': bn_meta['determinism_fraction'],
                                    'cpt_seed': bn_meta['seed'],
                                    'variant_index': bn_meta['variant_index']
                                },
                                
                                # Copy all naming variant fields
                                **{k: v for k, v in naming_variant.items() if k not in ['dag']}  # Exclude DAG object for now
                            }
                            
                            cpt_variants.append(model)
                            
                            # Show progress
                            if 'fixed' in arity_spec:
                                arity_str = f"arity:{arity_spec['fixed']}"
                            else:
                                arity_str = f"arity:{arity_spec.get('min')}-{arity_spec.get('max')}"
                            print(f"  ✓ {model['model_id']}: {arity_str}, α={alpha}, det={det_frac}, variant={bn_meta['variant_index']}")
                            
                            model_counter += 1
                            
                    except Exception as e:
                        print(f"  ✗ Failed CPT generation for α={alpha}, det={det_frac}: {e}")
                        continue
        
        print()
    
    print(f"Successfully generated {len(cpt_variants)} complete Bayesian network models")
    print(f"Expected: {total_expected}")
    
    return cpt_variants


def main():
    """Main execution function for step-by-step dataset generation."""
    print("=== Network Dataset Generation ===")
    print()
    
    # Step 1: Generate base DAG configurations
    print("Step 1: Generating base DAG configurations...")
    
    # Parameters (can be adjusted)
    n_nodes_list = [7, 11, 15]
    treewidths = [2, 3, 4]
    dag_methods = ['random']  # Could add 'topological' if needed
    samples_per_config = 2
    base_seed = 42
    
    dag_configs = generate_base_dag_configs(
        n_nodes_list=n_nodes_list,
        treewidths=treewidths,
        dag_methods=dag_methods,
        samples_per_config=samples_per_config,
        base_seed=base_seed
    )
    
    print(f"Generated {len(dag_configs)} base DAG configurations:")
    print("\nSample configurations:")
    for i, config in enumerate(dag_configs[:5]):  # Show first 5
        print(f"  {config['dag_id']}: {config['n_nodes']} nodes, treewidth {config['target_treewidth']}, "
              f"method {config['dag_method']}, seed {config['structural_seed']}")
    
    if len(dag_configs) > 5:
        print(f"  ... and {len(dag_configs) - 5} more")
    
    print()
    print("Step 1 completed successfully!")
    print()
    
    # Step 2: Generate naming variants for each DAG
    print("Step 2: Generating naming variants for each DAG...")
    print()
    
    naming_strategies = ['simple', 'confusing']
    naming_variants = generate_naming_variants(
        dag_configs=dag_configs,
        naming_strategies=naming_strategies
    )
    
    print()
    print("Step 2 completed successfully!")
    print(f"Generated {len(naming_variants)} total naming variants")
    print()
    
    # Show some examples to verify same structure, different names
    print("Verification - Same DAG structure with different naming:")
    first_dag_id = dag_configs[0]['dag_id']
    same_structure_variants = [v for v in naming_variants if v['dag_id'] == first_dag_id]
    
    for variant in same_structure_variants:
        nodes_preview = list(variant['dag'].nodes())[:4]
        edges_count = variant['dag'].number_of_edges()
        print(f"  {variant['naming_variant_id']} ({variant['naming_strategy']}): "
              f"nodes={nodes_preview}{'...' if len(variant['dag'].nodes()) > 4 else ''}, "
              f"edges={edges_count}")
    
    print()
    
    # Step 3: Generate CPT variants for each DAG+naming combination
    print("Step 3: Generating CPT variants for each DAG+naming combination...")
    print()
    
    # CPT parameters (can be adjusted)
    arity_specs = [{"type": "range", "min": 2, "max": 3}]
    dirichlet_alphas = [0.5, 1.0]
    determinism_fracs = [0.0, 0.1]
    variants_per_combo = 2
    
    cpt_variants = generate_cpt_variants(
        naming_variants=naming_variants,
        arity_specs=arity_specs,
        dirichlet_alphas=dirichlet_alphas,
        determinism_fracs=determinism_fracs,
        variants_per_combo=variants_per_combo
    )
    
    print()
    print("Step 3 completed successfully!")
    print(f"Generated {len(cpt_variants)} complete Bayesian network models")
    print()
    
    # Show some examples to verify the dual storage format
    print("Verification - Dual storage format examples:")
    sample_model = cpt_variants[0]
    print(f"Model {sample_model['model_id']} ({sample_model['dag_id']}, {sample_model['naming_strategy']}):")
    print(f"  CPDs as string (first 150 chars): {sample_model['cpds_as_string'][:150]}...")
    print(f"  CPD arrays keys: {list(sample_model['cpd_arrays'].keys())}")
    print(f"  First array shape: {list(sample_model['cpd_arrays'].values())[0].shape}")
    
    print()
    print("Next step will be:")
    print("4. Export clean DataFrame with dag_id and model_id")


if __name__ == "__main__":
    main()
