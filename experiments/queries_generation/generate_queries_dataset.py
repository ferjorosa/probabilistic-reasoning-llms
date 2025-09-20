"""
Clean Query Dataset Generation for Probabilistic Reasoning Experiments

This module provides a structured approach to generating probabilistic queries from
existing Bayesian network datasets for LLM probabilistic reasoning experiments.

Key Design Principles:
- Reuse existing query generation functionality from src/query_generation.py
- Reference networks by model_id rather than duplicating network data
- Generate multiple queries per network with exact inference results
- Clean separation between network data and query data

Author: Generated for LLM probabilistic reasoning research
"""

import pandas as pd
from pathlib import Path

from query_variant_generation import generate_query_variants_systematic
from query_dataset_export import export_queries_dataset


def main():
    """Main execution function for query dataset generation."""
    print("=== Query Dataset Generation ===")
    print()
    
    # Step 1: Load networks dataset
    print("Step 1: Loading networks dataset...")
    
    networks_path = Path("experiments/network_generation/networks.parquet")
    if not networks_path.exists():
        # Try alternative path
        networks_path = Path("experiments/networks.parquet")
        if not networks_path.exists():
            raise FileNotFoundError(f"Networks dataset not found. Please ensure networks.parquet exists.")
    
    networks_df = pd.read_parquet(networks_path)
    print(f"✓ Loaded {len(networks_df)} networks from {networks_path}")
    print(f"  • Unique DAGs: {networks_df['dag_id'].nunique()}")
    print(f"  • Networks per DAG: {len(networks_df) / networks_df['dag_id'].nunique():.1f}")
    print()
    
    # Step 2: Generate queries for each network
    print("Step 2: Generating queries for each network...")
    
    # Query generation parameters (systematic combinations)
    queries_per_combination = 1  # Number of queries per parameter combination
    query_node_counts = [1, 2]
    evidence_counts = [1, 2, 3]
    distance_buckets = [(1, 1), (2, 3), (4, 5), (6, 7)]
    base_seed = 1000
    
    query_variants = generate_query_variants_systematic(
        networks_df=networks_df,
        queries_per_combination=queries_per_combination,
        query_node_counts=query_node_counts,
        evidence_counts=evidence_counts,
        distance_buckets=distance_buckets,
        base_seed=base_seed
    )
    
    print(f"✓ Generated {len(query_variants)} queries")
    print()
    
    # Step 3: Export to parquet dataset
    print("Step 3: Exporting to queries.parquet dataset...")
    
    queries_df = export_queries_dataset(
        query_variants=query_variants,
        output_path="experiments/queries_generation/queries.parquet"
    )
    
    print()
    
    # Show final summary
    print("=== Generation Complete ===")
    print(f"• Source networks: {len(networks_df)}")
    print(f"• Generated queries: {len(query_variants)}")
    print(f"• Average queries per network: {len(query_variants) / len(networks_df):.1f}")
    print(f"• Dataset exported: queries.parquet ({queries_df.shape[0]} rows, {queries_df.shape[1]} columns)")
    print()
    print("✅ Ready for LLM probabilistic inference evaluation!")


if __name__ == "__main__":
    main()
