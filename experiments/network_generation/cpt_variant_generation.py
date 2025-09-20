"""
CPT Variant Generation for Probabilistic Reasoning Experiments

This module handles the generation of CPT (Conditional Probability Table) variants
for DAG+naming combinations. It takes naming variants and generates multiple 
Bayesian networks with different probability distributions.

The module stores CPTs in both string format (for LLM experiments) and array 
format (for numerical analysis), enabling comprehensive evaluation of 
probabilistic reasoning capabilities.

Author: Generated for LLM probabilistic reasoning research
"""

from typing import List, Dict, Any
from tqdm import tqdm

from src.bn_generation import generate_variants_for_dag
from src.cpd_utils import cpd_to_ascii_table


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
    
    # Calculate total expected models
    total_expected = len(naming_variants) * len(arity_specs) * len(dirichlet_alphas) * len(determinism_fracs) * variants_per_combo
    
    # Generate CPT variants with progress bar
    with tqdm(total=total_expected, desc="Generating CPT variants") as pbar:
        for naming_variant in naming_variants:
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
                                # Extract CPDs in both formats (use custom formatter to avoid truncation)
                                cpd_strings = [cpd_to_ascii_table(cpd) for cpd in bn.get_cpds()]
                                cpds_as_string = "\n\n".join(cpd_strings)
                                
                                # Store CPDs in 2D constructor format for easier reconstruction
                                cpd_arrays = {}
                                for cpd in bn.get_cpds():
                                    # Use get_values() method which returns 2D format
                                    cpd_arrays[cpd.variable] = cpd.get_values()
                                
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
                                model_counter += 1
                                pbar.update(1)
                                
                        except Exception as e:
                            # Update progress bar for failed variants too
                            for _ in range(variants_per_combo):
                                pbar.update(1)
                            tqdm.write(f"✗ Failed CPT generation for {naming_variant['naming_variant_id']}, α={alpha}, det={det_frac}: {e}")
                            continue
    
    return cpt_variants
