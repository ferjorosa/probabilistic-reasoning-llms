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

from dag_config_generation import generate_base_dag_configs
from naming_variant_generation import generate_naming_variants
from cpt_variant_generation import generate_cpt_variants
from dataset_export import export_networks_dataset


def main():
    """Main execution function for step-by-step dataset generation."""
    print("=== Network Dataset Generation ===")
    print()
    
    # Step 1: Generate base DAG configurations
    print("Step 1: Generating base DAG configurations...")
    
    # Parameters (can be adjusted)
    n_nodes_list = [5, 10, 15, 20, 25]
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
    
    print(f"✓ Generated {len(dag_configs)} base DAG configurations")
    print()
    
    # Step 2: Generate naming variants for each DAG
    print("Step 2: Generating naming variants for each DAG...")
    
    naming_strategies = ['simple', 'confusing']
    naming_variants = generate_naming_variants(
        dag_configs=dag_configs,
        naming_strategies=naming_strategies
    )
    
    print(f"✓ Generated {len(naming_variants)} naming variants")
    print()
    
    # Step 3: Generate CPT variants for each DAG+naming combination
    print("Step 3: Generating CPT variants for each DAG+naming combination...")
    
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
    
    print(f"✓ Generated {len(cpt_variants)} complete Bayesian network models")
    print()
    
    # Step 4: Export to HuggingFace dataset format
    print("Step 4: Exporting to networks.parquet dataset...")
    
    dataset_df = export_networks_dataset(
        cpt_variants=cpt_variants,
        naming_variants=naming_variants,
        output_path="networks.parquet"
    )
    
    print()
    
    # Show final summary
    print("=== Generation Complete ===")
    print(f"• DAG configurations: {len(dag_configs)}")
    print(f"• Naming variants: {len(naming_variants)}")
    print(f"• Complete models: {len(cpt_variants)}")
    print(f"• Dataset exported: networks.parquet ({dataset_df.shape[0]} rows, {dataset_df.shape[1]} columns)")
    print()
    print("✅ Ready for LLM probabilistic inference evaluation!")


if __name__ == "__main__":
    main()
