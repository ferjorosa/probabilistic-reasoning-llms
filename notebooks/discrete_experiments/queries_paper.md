# Bayesian Network Generation and Query Complexity Analysis: A Parameter Sweep Study

## Abstract

This paper presents a comprehensive framework for generating Bayesian networks (BNs) with controlled structural properties and evaluating the computational complexity of probabilistic queries within these networks. The study implements a systematic parameter sweep across multiple BN generation parameters including network size, treewidth, variable arity, conditional probability table (CPT) characteristics, and query complexity metrics. The framework enables controlled experimentation for understanding how different structural properties affect both exact inference complexity and potential large language model (LLM) performance on probabilistic reasoning tasks.

## 1. Introduction

Bayesian networks provide a powerful framework for representing and reasoning about uncertain knowledge. However, the computational complexity of exact inference in BNs grows exponentially with the network's treewidth, making it crucial to understand how different structural properties affect both theoretical complexity and practical reasoning performance. This work presents a systematic approach to generating BNs with controlled properties and analyzing query-specific complexity metrics.

## 2. Methodology

### 2.1 Bayesian Network Generation

The framework generates BNs through a multi-parameter sweep approach:

**Structural Parameters:**
- **Network size (n)**: Number of variables in the network
- **Target treewidth**: Controls the complexity of exact inference
- **Variable arity**: Cardinality of each variable (binary, ternary, quaternary)
- **Naming strategies**: Simple, confusing, or semantic variable names

**Probabilistic Parameters:**
- **Dirichlet alpha**: Controls CPT skewness (α=0.1 for skewed, α=1.0 for uniform)
- **Determinism fraction**: Proportion of deterministic (0/1) entries in CPTs
- **Variants per combination**: Multiple BN instances per parameter set

### 2.2 Query Generation and Analysis

For each generated BN, the framework:

1. **Generates diverse queries** with varying complexity:
   - Single and joint probability queries
   - Different evidence set sizes (0, 1, 2 variables)
   - Controlled distance between query and evidence variables

2. **Computes exact probabilities** using variable elimination

3. **Analyzes query-specific complexity**:
   - Induced width of elimination order
   - Total computational cost
   - Maximum intermediate factor size
   - Number of variables eliminated vs. kept

### 2.3 Complexity Metrics

The framework implements query-specific complexity analysis that considers:

- **Variables to eliminate**: All non-query variables
- **Variables to keep**: Query variables (preserved until final computation)
- **Evidence handling**: Evidence variables have effective cardinality of 1
- **Elimination order optimization**: Uses WeightedMinFill heuristic

Key metrics include:
- Induced width
- Total factor work (sum of intermediate factor sizes)
- Maximum intermediate factor size
- Logarithmic complexity measures

## 3. Implementation

### 3.1 Network Generation Pipeline

```python
# Parameter sweep configuration
ns = [5]  # Network sizes
treewidths = [3, 4]  # Target treewidths
arity_specs = [{"type": "range", "min": 2, "max": 3}]
dirichlet_alphas = [1.0, 0.5]
determinism_fracs = [0.0]
naming_strategies = ['confusing']
variants_per_combo = 4
```

### 3.2 Query Complexity Analysis

The framework implements a sophisticated complexity analysis that:

1. **Identifies query-specific elimination requirements**
2. **Computes optimal elimination order** for non-query variables
3. **Simulates variable elimination** to measure computational cost
4. **Accounts for evidence** by reducing effective cardinalities

### 3.3 Quality Control

The framework includes filtering mechanisms:
- **Structural filtering**: Removes BNs with insufficient connectivity (edges < 2×nodes)
- **Query filtering**: Focuses on queries with significant prior-posterior differences
- **Complexity filtering**: Selects queries with sufficient structural complexity

## 4. Results and Analysis

### 4.1 Network Generation Statistics

The framework successfully generates multiple BN variants with controlled properties:
- **Total networks generated**: 16 (before filtering)
- **Networks after filtering**: 8 (50% retention rate)
- **Queries per network**: 24
- **Total queries analyzed**: 192

### 4.2 Query Complexity Distribution

The analysis reveals:
- **Query-specific elimination**: 4-5 variables eliminated per query
- **Induced width range**: 3-4 for the tested networks
- **Computational cost**: Varies significantly based on query structure
- **Factor size progression**: Shows exponential growth during elimination

### 4.3 Prior-Posterior Analysis

The framework analyzes the relationship between prior and posterior probabilities:
- **Absolute differences**: Range from 0.1 to 0.8
- **Relative differences**: Up to 5x changes from prior to posterior
- **Evidence impact**: Significant updates when evidence is informative

## 5. Applications and Extensions

### 5.1 LLM Evaluation Framework

The generated BNs and queries serve as a controlled testbed for evaluating:
- **LLM probabilistic reasoning capabilities**
- **Impact of network structure on LLM performance**
- **Effect of query complexity on reasoning accuracy**

### 5.2 Complexity Prediction

The framework enables:
- **Theoretical complexity analysis** for different query types
- **Empirical validation** of complexity predictions
- **Optimization** of inference algorithms

### 5.3 Educational Applications

The systematic generation approach provides:
- **Controlled examples** for teaching probabilistic reasoning
- **Scalable difficulty** for progressive learning
- **Realistic scenarios** with known ground truth

## 6. Technical Implementation

### 6.1 Dependencies

The framework builds upon:
- **pgmpy**: Bayesian network representation and inference
- **NetworkX**: Graph structure manipulation
- **pandas**: Data management and analysis
- **matplotlib**: Visualization
- **OpenAI/OpenRouter**: LLM integration

### 6.2 Modular Design

Key components:
- **`generate_bayesian_networks_and_metadata()`**: BN generation
- **`compute_query_complexity()`**: Query-specific analysis
- **`inspect_row_and_call_llm()`**: LLM evaluation
- **`generate_queries()`**: Query generation

## 7. Future Work

### 7.1 Extended Parameter Sweeps

- **Larger networks**: Test with 10-50 variables
- **Higher treewidths**: Explore networks with treewidth 5-10
- **More arity options**: Include quaternary and higher cardinalities

### 7.2 Advanced Query Types

- **MAP queries**: Most probable explanation
- **MPE queries**: Most probable explanation
- **Marginal queries**: Complex marginalization patterns

### 7.3 LLM Integration

- **Systematic LLM evaluation** across all generated queries
- **Prompt engineering** optimization
- **Error analysis** and failure mode identification

## 8. Conclusion

This framework provides a comprehensive approach to generating controlled Bayesian networks and analyzing query-specific complexity. The systematic parameter sweep enables controlled experimentation for understanding the relationship between network structure, query complexity, and reasoning performance. The modular design facilitates extension to larger networks, more complex queries, and systematic LLM evaluation.

The framework's ability to generate networks with known structural properties and compute exact complexity metrics makes it valuable for both theoretical research and practical applications in probabilistic reasoning, machine learning, and artificial intelligence.

## References

- Pearl, J. (1988). Probabilistic reasoning in intelligent systems: networks of plausible inference.
- Koller, D., & Friedman, N. (2009). Probabilistic graphical models: principles and techniques.
- Darwiche, A. (2009). Modeling and reasoning with Bayesian networks.
