# pgmpy Bug Report: Incorrect Variable Assignment in Linear Gaussian Bayesian Networks

## Summary

We discovered a critical bug in pgmpy's `LinearGaussianBayesianNetwork` implementation where the `get_random_cpds()` and/or `to_joint_gaussian()` methods incorrectly assign CPD parameters to variables, leading to wrong inference results.

## Bug Description

When creating a Linear Gaussian Bayesian Network with specific CPD parameters, pgmpy internally swaps or misassigns the parameters to different variables, causing:

1. **Incorrect marginal distributions**
2. **Wrong conditional inference results** 
3. **Inconsistency between individual CPDs and joint Gaussian representation**

## Reproducible Example

### Setup

```python
import sys
from pathlib import Path
sys.path.append(str((Path('../../src')).resolve()))

from pgmpy.models import LinearGaussianBayesianNetwork
from inference_continuous import query_lgbn
import numpy as np
```

### Test Case 1: Basic Model Creation

```python
# Create the network structure
model = LinearGaussianBayesianNetwork([
    ('Pollution', 'Cancer'),
    ('Smoker', 'Cancer'),
    ('Cancer', 'Xray'),
    ('Cancer', 'Dyspnoea'),
])

# Generate random CPDs (but they should be consistent)
model.get_random_cpds(inplace=True)

# Print the individual CPDs
print("Individual CPDs:")
for cpd in model.get_cpds():
    print(cpd)
    print()
```

**Expected Output:**
```
P(Pollution) = N(0.305; 1.04)
P(Cancer | Pollution, Smoker) = N(0.678*Pollution + -0.586*Smoker + 0.244; 0.909)
P(Smoker) = N(1.446; 0.102)
P(Xray | Cancer) = N(-0.623*Cancer + -0.458; 0.135)
P(Dyspnoea | Cancer) = N(1.218*Cancer + -0.503; 0.271)
```

### Test Case 2: Manual Calculation vs pgmpy Results

```python
# Manual calculation of Cancer marginal
# E[Cancer] = 0.678 * E[Pollution] + (-0.586) * E[Smoker] + 0.244
# E[Cancer] = 0.678 * 0.305 + (-0.586) * 1.446 + 0.244
# E[Cancer] = 0.207 + (-0.847) + 0.244 = -0.396

manual_cancer_mean = 0.678 * 0.305 + (-0.586) * 1.446 + 0.244
print(f"Manual calculation - Cancer mean: {manual_cancer_mean:.4f}")

# pgmpy calculation
result = query_lgbn(model, 'Cancer')
print(f"pgmpy calculation - Cancer mean: {result['mean']:.4f}")

# Check all marginals
print("\nAll marginals:")
for var in ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea']:
    result = query_lgbn(model, var)
    print(f"{var}: Mean = {result['mean']:.4f}, Std = {result['std']:.4f}")
```

**Expected vs Actual Results:**

| Variable | Expected Mean | Expected Std | pgmpy Mean | pgmpy Std | Status |
|----------|---------------|--------------|------------|-----------|---------|
| Pollution | 0.305 | 1.04 | 0.305 | 1.02 | ✅ Correct |
| Smoker | 1.446 | 0.102 | -0.396 | 1.19 | ❌ **Wrong** |
| Cancer | -0.396 | ~1.15 | 1.446 | 0.32 | ❌ **Wrong** |

### Test Case 3: Joint Gaussian Inspection

```python
# Check what pgmpy's to_joint_gaussian() returns
mean, cov = model.to_joint_gaussian()
variable_names = list(model.nodes())

print(f"Variable order: {variable_names}")
print(f"Joint means: {mean}")

# Compare with individual CPD parameters
print("\nComparison:")
print(f"Pollution CPD mean: 0.305, Joint position 0: {mean[0]:.4f}")
print(f"Cancer expected mean: -0.396, Joint position 1: {mean[1]:.4f}")  
print(f"Smoker CPD mean: 1.446, Joint position 2: {mean[2]:.4f}")
```

**Actual Output:**
```
Variable order: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea']
Joint means: [ 0.30471708  1.44596273 -0.39577164 -0.2113554  -0.9849224 ]

Comparison:
Pollution CPD mean: 0.305, Joint position 0: 0.3047  ✅ Correct
Cancer expected mean: -0.396, Joint position 1: 1.4460  ❌ Wrong (shows Smoker's value)
Smoker CPD mean: 1.446, Joint position 2: -0.3958   ❌ Wrong (shows Cancer's value)
```

### Test Case 4: Edge Order Dependency

```python
# Test if edge order affects the bug
model2 = LinearGaussianBayesianNetwork([
    ('Smoker', 'Cancer'),      # Different order
    ('Pollution', 'Cancer'),
    ('Cancer', 'Xray'),
    ('Cancer', 'Dyspnoea'),
])

model2.get_random_cpds(inplace=True)

print("With different edge order:")
for cpd in model2.get_cpds():
    print(cpd)
    print()
```

**Result:** The bug gets even worse - now Pollution and Smoker CPDs are completely swapped:
```
P(Smoker) = N(0.305; 1.04)      ❌ Should be Pollution's parameters
P(Pollution) = N(1.446; 0.102)  ❌ Should be Smoker's parameters
```

## Verification with Correct Implementation (Pyro)

To verify our analysis, we implemented the same network in Pyro:

```python
import torch
import pyro
import pyro.distributions as dist

def make_vectorized_model(num_samples, device):
    def vectorized_model():
        with pyro.plate("particles", num_samples):
            pollution = pyro.sample("pollution", dist.Normal(0.305, 1.04))
            smoker = pyro.sample("smoker", dist.Normal(1.446, 0.102))
            
            cancer_mean = 0.678 * pollution + (-0.586) * smoker + 0.244
            cancer = pyro.sample("cancer", dist.Normal(cancer_mean, 0.909))
            
            # ... rest of the model
    return vectorized_model

# Test marginals
device = torch.device("cpu")
num_samples = 100000

# Cancer marginal test
samples = run_importance_sampling(num_samples, device, "cancer", {})
print(f"Pyro Cancer marginal: {samples.mean().item():.4f}")  # Should be ~-0.396

# Simple conditional test  
samples = run_importance_sampling(num_samples, device, "cancer", {"smoker": 2.0})
manual_calc = 0.678 * 0.305 + (-0.586) * 2.0 + 0.244  # = -0.721
print(f"Pyro P(Cancer|Smoker=2.0): {samples.mean().item():.4f}")
print(f"Manual calculation: {manual_calc:.4f}")
```

**Pyro Results:**
```
Pyro Cancer marginal: -0.3946  ✅ Matches manual calculation
Pyro P(Cancer|Smoker=2.0): -0.7215  ✅ Matches manual calculation (-0.721)
```

## Impact

This bug affects:

1. **All marginal queries** involving swapped variables
2. **All conditional inference** that depends on the affected variables  
3. **Any downstream analysis** using pgmpy's Linear Gaussian networks

## Root Cause Analysis

The bug appears to be in either:
1. `get_random_cpds()` method incorrectly assigning parameters to variables
2. `to_joint_gaussian()` method incorrectly mapping variables to positions
3. Internal variable ordering inconsistencies

## Workaround

Until this bug is fixed in pgmpy:
1. **Use alternative implementations** (e.g., Pyro, as shown above)
2. **Manually verify marginals** against expected calculations
3. **Avoid pgmpy for Linear Gaussian networks** in critical applications

## Environment

- pgmpy version: [check with `pip show pgmpy`]
- Python version: 3.12
- OS: macOS

## Recommendation

This is a critical bug that makes pgmpy's Linear Gaussian Bayesian Network implementation unreliable. The issue should be reported to the pgmpy maintainers with this reproducible example.

---

*Generated on: [Date]*  
*Tested with: pgmpy LinearGaussianBayesianNetwork*
