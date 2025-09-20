"""
Query Variant Generation for Probabilistic Reasoning Experiments

This module handles the generation of probabilistic queries for each Bayesian network
in the dataset. It takes the networks dataset and generates multiple queries per network
using the existing query generation functionality.

The module stores queries with their exact inference results for LLM evaluation.

Author: Generated for LLM probabilistic reasoning research
"""

import json
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

from src.query_generation import generate_queries, QuerySpec
from pgmpy.inference import VariableElimination


def _format_probability_query(query_vars, query_states, evidence=None):
    """
    Generate formatted query string like P(A=s0, B=s1 | C=s0, D=s1).
    
    Inspired by discrete_inference.py format_probability_query function.
    
    Args:
        query_vars: List of query variable names
        query_states: List of query variable states
        evidence: Dictionary of evidence assignments (or None)
        
    Returns:
        Formatted query string for LLM prompts
    """
    # Format query variables
    if len(query_vars) == 1:
        query_part = f"{query_vars[0]}={query_states[0]}"
    else:
        query_parts = [f"{var}={state}" for var, state in zip(query_vars, query_states)]
        query_part = ", ".join(query_parts)
    
    # Format evidence
    if evidence:
        evidence_parts = [f"{var}={state}" for var, state in evidence.items()]
        evidence_str = ", ".join(evidence_parts)
        return f"P({query_part} | {evidence_str})"
    else:
        return f"P({query_part})"


def reconstruct_bayesian_network(row):
    """
    Reconstruct a pgmpy DiscreteBayesianNetwork from a dataset row.
    
    Note: This is a simplified version for query generation.
    """
    import numpy as np
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    
    # Parse the stored data
    nodes = json.loads(row['nodes'])
    edges = json.loads(row['edges'])
    cpd_arrays = json.loads(row['cpd_arrays'])
    
    # Create the network structure
    model = DiscreteBayesianNetwork(edges)
    
    # Reconstruct CPDs
    cpds = []
    for node in nodes:
        # Get the CPD array for this node
        cpd_values = np.array(cpd_arrays[node])
        
        # Determine parents from the edges
        parents = [edge[0] for edge in edges if edge[1] == node]
        
        # Get cardinalities
        variable_card = cpd_values.shape[0]
        
        # Ensure 2D format
        if cpd_values.ndim != 2:
            raise ValueError(f"Expected 2D CPD values for {node}, got shape {cpd_values.shape}")
        
        if parents:
            # Calculate evidence cardinalities
            evidence_card = []
            for parent in parents:
                parent_cpd = np.array(cpd_arrays[parent])
                evidence_card.append(parent_cpd.shape[0])
            
            # Create state names
            state_names = {node: [f"s{i}" for i in range(variable_card)]}
            for parent in parents:
                parent_cpd = np.array(cpd_arrays[parent])
                state_names[parent] = [f"s{i}" for i in range(parent_cpd.shape[0])]
            
            cpd = TabularCPD(
                variable=node,
                variable_card=variable_card,
                values=cpd_values,
                evidence=parents,
                evidence_card=evidence_card,
                state_names=state_names
            )
        else:
            # Root node
            state_names = {node: [f"s{i}" for i in range(variable_card)]}
            
            cpd = TabularCPD(
                variable=node,
                variable_card=variable_card,
                values=cpd_values,
                state_names=state_names
            )
        
        cpds.append(cpd)
    
    # Add CPDs to the model
    model.add_cpds(*cpds)
    model.check_model()
    
    return model


def generate_query_variants_systematic(
    networks_df: pd.DataFrame,
    queries_per_combination: int = 2,
    query_node_counts: List[int] = [1, 2],
    evidence_counts: List[int] = [0, 1, 2],
    distance_buckets: List[tuple] = [(1, 1), (2, 3), (1, 3)],
    base_seed: int = 1000
) -> List[Dict[str, Any]]:
    """
    Generate query variants systematically for each parameter combination.
    
    Similar to network generation, this creates queries for every combination of:
    - query_node_counts × evidence_counts × distance_buckets
    
    Args:
        networks_df: DataFrame with network data (from networks.parquet)
        queries_per_combination: Number of queries per parameter combination
        query_node_counts: List of numbers of query nodes to test
        evidence_counts: List of numbers of evidence nodes to test
        distance_buckets: List of (min_dist, max_dist) tuples for evidence distance
        base_seed: Base seed for reproducible query generation
        
    Returns:
        List of dictionaries, each containing query data and exact probabilities
    """
    import itertools
    
    query_variants = []
    query_counter = 1
    
    # Generate parameter combinations, but skip distance buckets when evidence_count = 0
    param_combinations = []
    for query_count in query_node_counts:
        for evidence_count in evidence_counts:
            if evidence_count == 0:
                # When no evidence, distance doesn't matter - use a single dummy bucket
                param_combinations.append((query_count, evidence_count, (0, 0)))
            else:
                # When evidence exists, use all distance buckets
                for distance_bucket in distance_buckets:
                    param_combinations.append((query_count, evidence_count, distance_bucket))
    
    total_networks = len(networks_df)
    total_combinations = len(param_combinations)
    total_expected_queries = total_networks * total_combinations * queries_per_combination
    
    print(f"Systematic query generation:")
    print(f"  • Parameter combinations: {total_combinations}")
    print(f"    - Query node counts: {query_node_counts}")
    print(f"    - Evidence counts: {evidence_counts}")
    print(f"    - Distance buckets: {distance_buckets} (only used when evidence > 0)")
    print(f"  • Queries per combination: {queries_per_combination}")
    print(f"  • Total queries per network: {total_combinations * queries_per_combination}")
    print(f"  • Total expected queries: {total_expected_queries}")
    print()
    
    # Show the actual combinations for clarity
    print("Parameter combinations:")
    for i, (q_count, e_count, dist_bucket) in enumerate(param_combinations):
        if e_count == 0:
            print(f"  {i:2d}: {q_count} query, {e_count} evidence (distance: N/A)")
        else:
            print(f"  {i:2d}: {q_count} query, {e_count} evidence, distance {dist_bucket}")
    print()
    
    with tqdm(total=total_expected_queries, desc="Generating systematic queries") as pbar:
        for network_idx, (_, network_row) in enumerate(networks_df.iterrows()):
            try:
                # Reconstruct the Bayesian network
                bn = reconstruct_bayesian_network(network_row)
                
                # Set up inference engine
                inference = VariableElimination(bn)
                
                # Generate queries for each parameter combination
                for combo_idx, (num_query_nodes, num_evidence_nodes, distance_bucket) in enumerate(param_combinations):
                    # Generate multiple queries for this specific combination
                    combo_seed = base_seed + network_idx * 1000 + combo_idx * 100
                    
                    queries = generate_queries(
                        bn,
                        num_queries=queries_per_combination,
                        query_node_counts=[num_query_nodes],  # Fixed for this combination
                        evidence_counts=[num_evidence_nodes],  # Fixed for this combination
                        distance_buckets=[distance_bucket],    # Fixed for this combination
                        seed=combo_seed
                    )
                    
                    # Process each query in this combination
                    for query_idx, query in enumerate(queries):
                        try:
                            # Extract query components
                            query_vars = [var for var, _ in query.targets]
                            query_states = [state for _, state in query.targets]
                            evidence = query.evidence if query.evidence else {}
                            
                            # Compute detailed distance information
                            evidence_distances = []
                            if evidence:
                                # Import the distance function (same as in query_generation.py)
                                import networkx as nx
                                
                                # Convert BN to undirected graph for distance calculation
                                G = nx.Graph()
                                G.add_edges_from(bn.edges())
                                
                                # Calculate distance from each query var to each evidence var
                                for q_var in query_vars:
                                    for e_var in evidence.keys():
                                        try:
                                            distance = nx.shortest_path_length(G, q_var, e_var)
                                            evidence_distances.append({
                                                'query_var': q_var,
                                                'evidence_var': e_var,
                                                'distance': distance
                                            })
                                        except nx.NetworkXNoPath:
                                            # If no path exists, set a large distance
                                            evidence_distances.append({
                                                'query_var': q_var,
                                                'evidence_var': e_var,
                                                'distance': 999
                                            })
                            
                            # Compute exact probability
                            result = inference.query(
                                variables=query_vars, 
                                evidence=evidence if evidence else None,
                                show_progress=False
                            )
                            
                            # Get probability for the specific assignment
                            assignment = dict(zip(query_vars, query_states))
                            exact_probability = float(result.get_value(**assignment))
                            
                            # Create query entry
                            query_entry = {
                                # Network references (minimal - just for joining)
                                'model_id': network_row['model_id'],
                                'dag_id': network_row['dag_id'],
                                
                                # Query identifiers
                                'query_id': f'query_{query_counter:06d}',
                                'combination_index': combo_idx,
                                'query_index_in_combination': query_idx,
                                
                                # Parameter combination (explicit)
                                'num_query_nodes_param': num_query_nodes,
                                'num_evidence_nodes_param': num_evidence_nodes,
                                'distance_bucket_param': json.dumps(distance_bucket),
                                
                                # Query specification (for LLM prompts)
                                'query_string': _format_probability_query(query_vars, query_states, evidence),
                                'query_vars': json.dumps(query_vars),
                                'query_states': json.dumps(query_states),
                                'evidence': json.dumps(evidence),
                                
                                # Results
                                'exact_probability': exact_probability,
                                
                                # Query metadata (actual values - may differ from params due to network constraints)
                                'query_metadata': json.dumps(query.meta),
                                'num_query_nodes_actual': len(query_vars),
                                'num_evidence_nodes_actual': len(evidence),
                                'min_target_evidence_distance': query.meta.get('min_target_evidence_distance', 0),
                                'distance_bucket_actual': json.dumps(query.meta.get('distance_bucket', (0, 0))),
                                
                                # Detailed distance information
                                'evidence_distances': json.dumps(evidence_distances),
                                
                                # Generation parameters
                                'combo_seed': combo_seed
                            }
                            
                            query_variants.append(query_entry)
                            query_counter += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            tqdm.write(f"✗ Failed query {query_idx} for combination {combo_idx} in model {network_row['model_id']}: {e}")
                            pbar.update(1)
                            continue
                            
            except Exception as e:
                tqdm.write(f"✗ Failed query generation for model {network_row['model_id']}: {e}")
                # Update progress bar for failed queries too
                for _ in range(total_combinations * queries_per_combination):
                    pbar.update(1)
                continue
    
    return query_variants


def generate_query_variants(
    networks_df: pd.DataFrame,
    num_queries_per_network: int = 5,
    query_node_counts: List[int] = [1, 2],
    evidence_counts: List[int] = [0, 1, 2],
    distance_buckets: List[tuple] = [(1, 1), (2, 3), (1, 3)],
    base_seed: int = 1000
) -> List[Dict[str, Any]]:
    """
    Generate query variants for each network in the dataset.
    
    Args:
        networks_df: DataFrame with network data (from networks.parquet)
        num_queries_per_network: Number of queries to generate per network
        query_node_counts: List of possible numbers of query nodes
        evidence_counts: List of possible numbers of evidence nodes
        distance_buckets: List of (min_dist, max_dist) tuples for evidence distance
        base_seed: Base seed for reproducible query generation
        
    Returns:
        List of dictionaries, each containing:
        - All network metadata (dag_id, model_id, etc.)
        - query_id: Unique identifier for this query
        - query_index: Index of query within this network (0 to num_queries_per_network-1)
        - query_string: Human-readable query like "P(A=s0 | B=s1)"
        - query_vars: List of query variable names
        - query_states: List of query variable states
        - evidence: Dictionary of evidence assignments
        - exact_probability: Computed exact probability
        - query_metadata: Difficulty metrics and generation parameters
    """
    query_variants = []
    query_counter = 1
    
    total_expected = len(networks_df) * num_queries_per_network
    
    with tqdm(total=total_expected, desc="Generating queries") as pbar:
        for idx, network_row in networks_df.iterrows():
            try:
                # Reconstruct the Bayesian network
                bn = reconstruct_bayesian_network(network_row)
                
                # Generate queries for this network
                query_seed = base_seed + idx
                queries = generate_queries(
                    bn,
                    num_queries=num_queries_per_network,
                    query_node_counts=query_node_counts,
                    evidence_counts=evidence_counts,
                    distance_buckets=distance_buckets,
                    seed=query_seed
                )
                
                # Set up inference engine
                inference = VariableElimination(bn)
                
                # Process each query
                for query_idx, query in enumerate(queries):
                    try:
                        # Extract query components
                        query_vars = [var for var, _ in query.targets]
                        query_states = [state for _, state in query.targets]
                        evidence = query.evidence if query.evidence else {}
                        
                        # Compute detailed distance information
                        evidence_distances = []
                        if evidence:
                            # Import the distance function (same as in query_generation.py)
                            import networkx as nx
                            
                            # Convert BN to undirected graph for distance calculation
                            G = nx.Graph()
                            G.add_edges_from(bn.edges())
                            
                            # Calculate distance from each query var to each evidence var
                            for q_var in query_vars:
                                for e_var in evidence.keys():
                                    try:
                                        distance = nx.shortest_path_length(G, q_var, e_var)
                                        evidence_distances.append({
                                            'query_var': q_var,
                                            'evidence_var': e_var,
                                            'distance': distance
                                        })
                                    except nx.NetworkXNoPath:
                                        # If no path exists, set a large distance
                                        evidence_distances.append({
                                            'query_var': q_var,
                                            'evidence_var': e_var,
                                            'distance': 999
                                        })
                        
                        # Compute exact probability
                        result = inference.query(
                            variables=query_vars, 
                            evidence=evidence if evidence else None,
                            show_progress=False
                        )
                        
                        # Get probability for the specific assignment
                        assignment = dict(zip(query_vars, query_states))
                        exact_probability = float(result.get_value(**assignment))
                        
                        # Create query entry
                        query_entry = {
                            # Network references (minimal - just for joining)
                            'model_id': network_row['model_id'],
                            'dag_id': network_row['dag_id'],
                            
                            # Query identifiers
                            'query_id': f'query_{query_counter:06d}',
                            'query_index': query_idx,
                            
                            # Query specification (for LLM prompts)
                            'query_string': _format_probability_query(query_vars, query_states, evidence),
                            'query_vars': json.dumps(query_vars),
                            'query_states': json.dumps(query_states),
                            'evidence': json.dumps(evidence),
                            
                            # Results
                            'exact_probability': exact_probability,
                            
                            # Query metadata
                            'query_metadata': json.dumps(query.meta),
                            'num_query_nodes': query.meta.get('num_query_nodes', len(query_vars)),
                            'num_evidence_nodes': query.meta.get('num_evidence_nodes', len(evidence)),
                            'min_target_evidence_distance': query.meta.get('min_target_evidence_distance', 0),
                            'distance_bucket': json.dumps(query.meta.get('distance_bucket', (0, 0))),
                            
                            # Detailed distance information
                            'evidence_distances': json.dumps(evidence_distances),
                            
                            # Generation parameters
                            'query_seed': query_seed
                        }
                        
                        query_variants.append(query_entry)
                        query_counter += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        tqdm.write(f"✗ Failed to compute query {query_idx} for {network_row['model_id']}: {e}")
                        pbar.update(1)
                        continue
                        
            except Exception as e:
                # Skip this network but update progress bar
                for _ in range(num_queries_per_network):
                    pbar.update(1)
                tqdm.write(f"✗ Failed to process network {network_row['model_id']}: {e}")
                continue
    
    return query_variants


