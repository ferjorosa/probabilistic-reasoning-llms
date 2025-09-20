"""
Dataset Export for Probabilistic Reasoning Experiments

This module handles the export of generated Bayesian networks to HuggingFace-compatible
datasets. It converts the internal data structures to a clean DataFrame format suitable
for LLM probabilistic inference evaluation.

The exported dataset contains all information needed for reproducibility and
LLM experiments, including network descriptions formatted for prompts.

Author: Generated for LLM probabilistic reasoning research
"""

import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


def export_networks_dataset(
    cpt_variants: List[Dict[str, Any]], 
    naming_variants: List[Dict[str, Any]],
    output_path: str = "experiments/networks.parquet"
) -> pd.DataFrame:
    """
    Export generated Bayesian networks to a HuggingFace-compatible dataset.
    
    Converts the internal data structures to a clean DataFrame with all information
    needed for reproducibility and LLM probabilistic inference evaluation.
    
    Args:
        cpt_variants: Complete Bayesian network models from generate_cpt_variants()
        naming_variants: Naming variants containing DAG objects
        output_path: Path where to save the parquet file
        
    Returns:
        DataFrame with the exported dataset
        
    Schema:
        - model_id: Unique identifier (primary key)
        - dag_id: For structural ablation studies  
        - naming_variant_id: For naming strategy ablation studies
        - n_nodes: Number of nodes
        - target_treewidth: Target treewidth
        - achieved_treewidth: Actual treewidth achieved
        - dag_method: Generation method
        - naming_strategy: Node naming strategy
        - dirichlet_alpha: Dirichlet parameter for CPT sampling
        - determinism_fraction: Fraction of deterministic columns
        - arity_min: Minimum variable arity
        - arity_max: Maximum variable arity
        - structural_seed: Seed for DAG structure generation
        - cpt_seed: Seed for CPT generation
        - sample_idx: Sample index within parameter combination
        - network_description: Full network as string (for LLM prompts)
        - cpd_arrays: CPD probability arrays as JSON dict {variable: array}
        - nodes: List of node names (JSON array)
        - edges: List of edges as [parent, child] pairs (JSON array)
        - edges_count: Number of edges
        - created_at: Timestamp when the network was generated (ISO format)
    """
    
    print(f"Exporting {len(cpt_variants)} networks to dataset with DAG structure info...")
    
    # Get current timestamp for all networks in this batch
    created_at = datetime.now().isoformat()
    
    # Create lookup for naming variants by naming_variant_id
    naming_lookup = {nv['naming_variant_id']: nv for nv in naming_variants}
    
    # Extract data for DataFrame
    records = []
    
    for model in cpt_variants:
        # Get corresponding naming variant with DAG object
        naming_variant = naming_lookup[model['naming_variant_id']]
        dag = naming_variant['dag']
        
        # Extract arity information
        arity_strategy = model['cpt_config']['arity_strategy']
        if arity_strategy['type'] == 'range':
            arity_min = arity_strategy['min']
            arity_max = arity_strategy['max']
        else:  # fixed
            arity_min = arity_strategy['fixed']
            arity_max = arity_strategy['fixed']
        
        # Extract DAG structure
        nodes = list(dag.nodes())
        edges = list(dag.edges())  # List of (parent, child) tuples
        edges_count = len(edges)
        
        record = {
            # Core identifiers
            'model_id': model['model_id'],
            'dag_id': model['dag_id'],
            'naming_variant_id': model['naming_variant_id'],
            
            # Network structure
            'n_nodes': model['n_nodes'],
            'target_treewidth': model['target_treewidth'],
            'achieved_treewidth': model['achieved_treewidth'],
            'dag_method': model['dag_method'],
            'naming_strategy': model['naming_strategy'],
            
            # CPT parameters
            'dirichlet_alpha': model['cpt_config']['dirichlet_alpha'],
            'determinism_fraction': model['cpt_config']['determinism_fraction'],
            'arity_min': arity_min,
            'arity_max': arity_max,
            
            # Seeds for reproducibility
            'structural_seed': model['structural_seed'],
            'cpt_seed': model['cpt_config']['cpt_seed'],
            'sample_idx': model['sample_idx'],
            
            # Network content for LLM experiments
            'network_description': model['cpds_as_string'],
            'cpd_arrays': json.dumps({var: cpd_array.tolist() for var, cpd_array in model['cpd_arrays'].items()}),
            
            # DAG structure
            'nodes': json.dumps(nodes),  # JSON array of node names
            'edges': json.dumps(edges),  # JSON array of [parent, child] pairs
            'edges_count': edges_count,
            
            # Metadata
            'created_at': created_at
        }
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to parquet
    output_path = Path(output_path)
    df.to_parquet(output_path, index=False)
    
    print(f"✓ Exported dataset to {output_path}")
    print(f"  • File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Columns: {list(df.columns)}")
    
    return df
