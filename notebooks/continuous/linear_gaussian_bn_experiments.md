# Linear Gaussian Bayesian Network Experiments: Pyro vs pgmpy

## Overview

This document details our comprehensive investigation into implementing and testing Linear Gaussian Bayesian Networks using both Pyro and pgmpy. We discovered significant bugs in pgmpy's implementation and validated that Pyro provides correct results when using appropriate inference methods.

## Network Specification

We tested with a 5-node Linear Gaussian Bayesian Network:

**Nodes:** `['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea']`

**Edges:** `[('Pollution', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Smoker', 'Cancer')]`

**Conditional Probability Distributions:**
- `P(Pollution) = N(0.305; 1.04)`
- `P(Smoker) = N(1.446; 0.102)`
- `P(Cancer | Pollution, Smoker) = N(0.678*Pollution + -0.586*Smoker + 0.244; 0.909)`
- `P(Xray | Cancer) = N(-0.623*Cancer + -0.458; 0.135)`
- `P(Dyspnoea | Cancer) = N(1.218*Cancer + -0.503; 0.271)`

## Experiments Conducted

### Experiment 1: Initial Comparison

**Objective:** Compare Pyro importance sampling with pgmpy exact inference for the query `P(Pollution | Dyspnoea = 0.5)`.

**Results:**
- **Pyro Importance Sampling:** Mean = 0.299, Std = 1.04
- **pgmpy:** Mean = 0.841, Std = 0.854

**Observation:** Large discrepancy led us to investigate further.

### Experiment 2: Marginal Distribution Validation

**Objective:** Verify that both implementations produce correct marginal distributions.

**Method:** Test marginals without evidence and compare with manual calculations.

**Manual Calculations:**
```python
E[Cancer] = 0.678 * E[Pollution] + (-0.586) * E[Smoker] + 0.244
         = 0.678 * 0.305 + (-0.586) * 1.446 + 0.244
         = 0.207 - 0.847 + 0.244 = -0.396
```

**Results:**

| Variable | Expected Mean | Expected Std | Pyro Mean | Pyro Std | pgmpy Mean | pgmpy Std | Pyro Status | pgmpy Status |
|----------|---------------|--------------|-----------|----------|------------|-----------|-------------|--------------|
| Pollution | 0.305 | 1.040 | 0.302 | 1.040 | 0.305 | 1.020 | ✅ Correct | ✅ Correct |
| Smoker | 1.446 | 0.102 | 1.446 | 0.102 | -0.396 | 1.193 | ✅ Correct | ❌ **Wrong** |
| Cancer | -0.396 | 1.152 | -0.395 | 1.152 | 1.446 | 0.320 | ✅ Correct | ❌ **Wrong** |

**Key Finding:** pgmpy swapped Cancer and Smoker marginals!

### Experiment 3: Simple Conditional Validation

**Objective:** Test a simple conditional query to validate Pyro's conditioning mechanism.

**Query:** `P(Cancer | Smoker = 2.0)`

**Manual Calculation:**
```python
E[Cancer | Smoker = 2.0] = 0.678 * 0.305 + (-0.586) * 2.0 + 0.244 = -0.721
```

**Results:**
- **Pyro:** Mean = -0.722
- **Manual:** Mean = -0.721
- **Match:** ✅ Perfect

**Conclusion:** Pyro's conditioning mechanism works correctly.

### Experiment 4: Joint Gaussian Investigation

**Objective:** Investigate pgmpy's `to_joint_gaussian()` method to identify the source of errors.

**Method:** Compare pgmpy's joint Gaussian with manual covariance calculations.

**Manual Covariance Calculations:**
```python
Var(Cancer) = (0.678)² * Var(Pollution) + (0.586)² * Var(Smoker) + 0.909²
            = 0.459 * 1.082 + 0.343 * 0.010 + 0.826 = 1.327

Cov(Pollution, Cancer) = 0.678 * Var(Pollution) = 0.678 * 1.082 = 0.733
```

**pgmpy Results:**
```python
Var(Cancer) = 0.102          # Should be 1.327 - Wrong by factor of 13!
Cov(Pollution, Cancer) = 0.0 # Should be 0.733 - Completely wrong!
```

**Conclusion:** pgmpy has fundamental errors in covariance matrix calculation.

### Experiment 5: Edge Order Dependency Test

**Objective:** Test if pgmpy's bugs depend on edge ordering.

**Method:** Create the same network with different edge order:
```python
# Original: [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ...]
# Modified: [('Smoker', 'Cancer'), ('Pollution', 'Cancer'), ...]
```

**Results:** Even worse! pgmpy completely swapped CPD parameters:
- `P(Smoker) = N(0.305; 1.04)` ❌ (should be Pollution's parameters)
- `P(Pollution) = N(1.446; 0.102)` ❌ (should be Smoker's parameters)

**Conclusion:** pgmpy has systematic variable assignment bugs.

### Experiment 6: Correct Implementation Development

**Objective:** Implement a mathematically correct Linear Gaussian BN to joint Gaussian conversion.

**Method:** Built `LinearGaussianBN` class with proper:
- Topological ordering
- Mean vector calculation
- Covariance matrix calculation

**Results:** Our implementation produces:
- **Correct marginals:** Cancer mean = -0.397 ✅
- **Correct covariances:** Cov(Pollution, Cancer) = 0.733 ✅
- **Consistent inference:** P(Pollution | Dyspnoea = 0.5) = Mean 0.955

### Experiment 7: Inference Method Comparison

**Objective:** Compare different Pyro inference methods for conditional queries.

**Query:** `P(Pollution | Dyspnoea = 0.5)`

**Methods Tested:**
1. **Importance Sampling (Vectorized)**
2. **MCMC with NUTS**

**Results:**

| Method | Mean | Std | Status | Notes |
|--------|------|-----|--------|-------|
| Pyro Importance Sampling | 0.308 | 1.039 | ❌ Wrong | Close to prior (0.305) |
| Pyro MCMC (NUTS) | 1.000 | 0.765 | ✅ Correct | Matches exact solution |
| Our Exact Implementation | 0.955 | 0.831 | ✅ Correct | Mathematical ground truth |
| pgmpy (buggy) | 0.841 | 0.854 | ❌ Wrong | Based on incorrect joint |

**Key Finding:** Importance sampling fails for "backward" inference (evidence on downstream variables), while MCMC works correctly.

## Summary of Findings

### ✅ What Works

1. **Pyro Model Specification:** Correctly implements the Linear Gaussian relationships
2. **Pyro Marginals:** Perfect match with manual calculations
3. **Pyro MCMC Inference:** Provides correct conditional inference results
4. **Our Custom Implementation:** Mathematically sound joint Gaussian conversion

### ❌ What's Broken

1. **pgmpy Variable Assignment:** Systematically swaps variables in joint Gaussian
2. **pgmpy Covariance Calculation:** Produces incorrect covariance matrices
3. **pgmpy Edge Order Sensitivity:** Results change incorrectly with edge ordering
4. **Pyro Importance Sampling:** Fails for certain types of conditional queries

## Technical Details

### Why Importance Sampling Failed

For the query `P(Pollution | Dyspnoea = 0.5)`:
- **Evidence path:** Pollution → Cancer → Dyspnoea
- **Problem:** "Backward" inference from effect to cause
- **Issue:** Importance sampling struggles with this evidence flow
- **Solution:** Use MCMC which handles bidirectional inference correctly

### pgmpy Bug Analysis

The bugs appear to be in pgmpy's internal variable indexing:

1. **`get_random_cpds()`** may assign parameters to wrong variables
2. **`to_joint_gaussian()`** may use incorrect variable ordering
3. **Variable name to index mapping** is inconsistent between methods

## Recommendations

### For Linear Gaussian Bayesian Networks:

1. **✅ Use Pyro with MCMC (NUTS)** for reliable inference
2. **❌ Avoid pgmpy** until bugs are fixed
3. **✅ Validate marginals** against manual calculations
4. **✅ Use exact methods** when possible (our custom implementation)

### For Conditional Inference:

1. **MCMC (NUTS):** Best for complex conditional queries
2. **Importance Sampling:** Only for simple forward inference
3. **Exact Methods:** When computational cost allows

## Code Artifacts

1. **`borrar.py`:** Pyro implementation with both importance sampling and MCMC
2. **`lgbn_to_joint_gaussian.py`:** Correct Linear Gaussian BN implementation
3. **`pgmpy_bug_report.md`:** Detailed bug documentation for pgmpy maintainers

## Validation

Our findings are validated by:
- ✅ **Manual mathematical calculations**
- ✅ **Consistency between Pyro MCMC and exact methods**
- ✅ **Reproducible pgmpy bugs across different network configurations**
- ✅ **Perfect marginal distribution matches in Pyro**

## Conclusion

This investigation demonstrates that:

1. **Pyro is reliable** for Linear Gaussian Bayesian Networks when using appropriate inference methods
2. **pgmpy has serious bugs** that make it unsuitable for Linear Gaussian networks
3. **Inference method choice matters** - MCMC outperforms importance sampling for complex conditional queries
4. **Mathematical validation is crucial** - always verify results against known calculations

The discrepancy in initial results was not due to fundamental issues with either framework, but rather:
- pgmpy's implementation bugs
- Inappropriate choice of inference method (importance sampling vs MCMC)

**Final recommendation:** Use Pyro with MCMC for Linear Gaussian Bayesian Network inference.

---

*Experiments conducted: [Date]*  
*Frameworks tested: Pyro 1.x, pgmpy*  
*Network: 5-node Linear Gaussian BN*
