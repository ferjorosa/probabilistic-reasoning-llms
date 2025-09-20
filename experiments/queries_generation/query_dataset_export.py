"""
Query Dataset Export for Probabilistic Reasoning Experiments

This module handles the export of generated queries to HuggingFace-compatible
datasets. It converts the query data structures to a clean DataFrame format suitable
for LLM probabilistic inference evaluation.

Author: Generated for LLM probabilistic reasoning research
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


def export_queries_dataset(
    query_variants: List[Dict[str, Any]], 
    output_path: str = "experiments/queries_generation/queries.parquet"
) -> pd.DataFrame:
    """
    Export generated queries to a HuggingFace-compatible dataset.
    
    Converts the internal query data structures to a clean DataFrame with all information
    needed for LLM probabilistic inference evaluation.
    
    Args:
        query_variants: Complete query specifications from generate_query_variants()
        output_path: Path where to save the parquet file
        
    Returns:
        DataFrame with the exported dataset
        
    Schema:
        Network References:
        - model_id: Reference to network in networks.parquet (foreign key)
        - dag_id: Reference to DAG structure (for joining/filtering)
        
        Query Information:
        - query_id: Unique query identifier (primary key)
        - combination_index: Index of parameter combination (0, 1, 2, ...)
        - query_index_in_combination: Query index within the combination (0, 1, ...)
        - query_string: LLM-ready query string like "P(A=s0, B=s1 | C=s0)"
        - query_vars: List of query variable names (JSON array)
        - query_states: List of query variable states (JSON array)  
        - evidence: Evidence assignments as JSON dict {var: state}
        - exact_probability: Computed exact probability value
        
        Parameter Combination (what was requested):
        - num_query_nodes_param: Target number of query nodes for this combination
        - num_evidence_nodes_param: Target number of evidence nodes for this combination  
        - distance_bucket_param: Target distance bucket for this combination (JSON array)
        
        Actual Results (what was achieved - may differ due to network constraints):
        - num_query_nodes_actual: Actual number of query variables
        - num_evidence_nodes_actual: Actual number of evidence variables
        - min_target_evidence_distance: Minimum distance between query and evidence
        - distance_bucket_actual: Actual distance bucket achieved (JSON array)
        - evidence_distances: Detailed distance info between each query-evidence pair (JSON array)
        
        Metadata:
        - query_metadata: Full query generation metadata (JSON dict)
        - combo_seed: Seed used for this parameter combination
        - query_created_at: Timestamp when queries were generated
    """
    
    print(f"Exporting {len(query_variants)} queries to dataset...")
    
    # Get current timestamp for all queries in this batch
    query_created_at = datetime.now().isoformat()
    
    # Add timestamp to each query
    for query in query_variants:
        query['query_created_at'] = query_created_at
    
    # Create DataFrame directly from the query variants
    df = pd.DataFrame(query_variants)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"✓ Exported queries dataset to {output_path}")
    print(f"  • File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Total queries: {len(df)}")
    print(f"  • Unique networks: {df['model_id'].nunique()}")
    print(f"  • Average queries per network: {len(df) / df['model_id'].nunique():.1f}")
    
    # Show some statistics
    print(f"\nQuery Statistics:")
    print(f"  • Query nodes: {df['num_query_nodes_actual'].value_counts().to_dict()}")
    print(f"  • Evidence nodes: {df['num_evidence_nodes_actual'].value_counts().to_dict()}")
    print(f"  • Probability range: [{df['exact_probability'].min():.6f}, {df['exact_probability'].max():.6f}]")
    
    return df


def load_queries_dataset(dataset_path: str = "experiments/queries_generation/queries.parquet") -> pd.DataFrame:
    """
    Load queries dataset from parquet file.
    
    Args:
        dataset_path: Path to the queries parquet file
        
    Returns:
        DataFrame with the queries dataset
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Queries dataset not found at {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    print(f"Loaded queries dataset: {len(df)} queries from {df['model_id'].nunique()} networks")
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the queries dataset.
    
    Args:
        df: Queries dataset DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    import json
    
    summary = {
        'total_queries': len(df),
        'unique_networks': df['model_id'].nunique(),
        'queries_per_network': len(df) / df['model_id'].nunique(),
        'query_node_distribution': df['num_query_nodes'].value_counts().to_dict(),
        'evidence_node_distribution': df['num_evidence_nodes'].value_counts().to_dict(),
        'probability_stats': {
            'min': float(df['exact_probability'].min()),
            'max': float(df['exact_probability'].max()),
            'mean': float(df['exact_probability'].mean()),
            'std': float(df['exact_probability'].std())
        },
        'network_properties': {
            'node_counts': df['n_nodes'].value_counts().to_dict(),
            'treewidths': df['achieved_treewidth'].value_counts().to_dict(),
            'naming_strategies': df['naming_strategy'].value_counts().to_dict()
        }
    }
    
    return summary


__all__ = ["export_queries_dataset", "load_queries_dataset", "get_dataset_summary"]
