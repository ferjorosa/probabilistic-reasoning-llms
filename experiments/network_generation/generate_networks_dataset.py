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
    n_nodes_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Larger range for better scaling analysis
    dag_methods = ['random']  # Could add 'topological' if needed
    samples_per_config = 5
    base_seed = 42
    treewidth_fractions = [0.1, 0.15, 0.2, 0.25, 0.3]  # 10%, 20%, 30%, 50% of nodes
    
    # Choose ONE of the following approaches:
    
    # Option 1: Proportional scaling (recommended for varying node counts)

    dag_configs = generate_base_dag_configs(
        n_nodes_list=n_nodes_list,
        treewidth_fractions=treewidth_fractions,
        dag_methods=dag_methods,
        samples_per_config=samples_per_config,
        base_seed=base_seed
    )
    
    # Option 2: Specific treewidth values (uncomment to use instead)
    # treewidths = [2, 4, 6, 8]  
    # dag_configs = generate_base_dag_configs(
    #     n_nodes_list=n_nodes_list,
    #     treewidths=treewidths,
    #     dag_methods=dag_methods,
    #     samples_per_config=samples_per_config,
    #     base_seed=base_seed
    # )
    
    print(f"✓ Generated {len(dag_configs)} base DAG configurations")
    
    # Show treewidth scaling for transparency
    if 'treewidth_fractions' in locals():
        print("Treewidth scaling (proportional):")
        from dag_config_generation import get_proportional_treewidths
        for n_nodes in n_nodes_list:
            tws = get_proportional_treewidths(n_nodes, treewidth_fractions)
            fractions_str = ", ".join([f"{f:.2f}" for f in treewidth_fractions])
            tws_str = ", ".join([str(tw) for tw in tws])
            print(f"  {n_nodes} nodes (fractions {fractions_str}): treewidths {tws_str}")
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
    arity_specs = [{"type": "range", "min": 2, "max": 4}]
    dirichlet_alphas = [0.5, 1.0]
    determinism_fracs = [0.0] # indicates percentage of rows with deterministic 0/1 entries
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
        output_path="experiments/networks.parquet"
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
