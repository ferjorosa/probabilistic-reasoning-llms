# What We Learned About MCMC in Discrete Bayesian Networks

This documents our deep dive into why MCMC fails with discrete variables and what alternatives work better.

## **The Original Problem**

**Goal**: Make MCMC work for the Asia Bayesian network  
**Challenge**: All variables are discrete (Categorical distributions)  
**Initial Attempt**: NUTS kernel with conditioned model  
**Result**: Complete failure - no samples returned ❌

---

## **Discovery #1: NUTS Doesn't Work with Discrete Variables**

**The Bug**: NUTS (No-U-Turn Sampler) requires continuous gradients:

```python
# FAILED APPROACH
nuts_kernel = NUTS(conditioned_model, step_size=0.1)
mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=20)
mcmc.run()
samples_dict = mcmc.get_samples()
print(samples_dict)  # Output: {} (empty!)
```

**Why It Fails**:
- NUTS uses Hamiltonian dynamics (requires gradients)
- Categorical distributions are discrete → no meaningful gradients
- The sampler runs but returns no samples

**Lesson**: Wrong tool for the job - like using a screwdriver as a hammer.

---

## **Discovery #2: Vectorized Models Confuse MCMC Further**

**The Problem**: Original code used `pyro.plate` for vectorization:

```python
# PROBLEMATIC APPROACH
def vectorized_model():
    with pyro.plate("particles", num_samples):  # ← Problem!
        asia = pyro.sample("asia", dist.Categorical(asia_probs))
        # ... rest of model
```

**Why Vectorization Hurts MCMC**:
- `pyro.plate` is for independent batch sampling
- MCMC needs to sample single instances sequentially
- Creates shape/indexing conflicts in discrete space

**Fix**: Use non-vectorized model for MCMC:
```python
# CORRECT APPROACH FOR MCMC
def model():  # No pyro.plate!
    asia = pyro.sample("asia", dist.Categorical(asia_probs))
    # ... rest of model
```

---

## **Discovery #3: Available MCMC Kernels Are Limited**

**What We Tried**:
1. **NUTS** → Failed (continuous only)
2. **DiscreteHMCGibbs** → Not available in this Pyro version
3. **MH (Metropolis-Hastings)** → Import error
4. **RandomWalkKernel** → Runs but returns empty samples

**Available Kernels in Our Pyro Version**:
```python
['HMC', 'NUTS', 'RandomWalkKernel', 'MCMC']
```

**The Reality**: Modern Pyro focuses on continuous variables for MCMC.

---

## **Discovery #4: RandomWalkKernel Runs But Fails Silently**

**The Attempt**:
```python
rw_kernel = RandomWalkKernel(conditioned_model)
mcmc = MCMC(rw_kernel, num_samples=20, warmup_steps=5)
mcmc.run()  # ✓ Runs without error
samples_dict = mcmc.get_samples()
print(list(samples_dict.keys()))  # Output: [] (empty!)
```

**What Happened**:
- No crash or error message
- Progress bar shows "100% complete"
- But no actual samples are collected
- Silent failure is worse than obvious failure!

**Lesson**: Always validate MCMC outputs - "no error" ≠ "working correctly"

---

## **Discovery #5: Rejection Sampling as MCMC Alternative**

**The Solution**: When MCMC fails, fall back to rejection sampling:

```python
def rejection_sampling_mcmc_replacement(num_samples, device, query, evidence):
    """Use rejection sampling when MCMC fails with discrete variables"""
    
    accepted_samples = []
    attempts = 0
    max_attempts = num_samples * 1000  # Safety limit
    
    while len(accepted_samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Generate a complete sample
        trace = poutine.trace(model).get_trace()
        sample_values = {name: node["value"] for name, node in trace.nodes.items() 
                        if node["type"] == "sample"}
        
        # Check if sample satisfies evidence
        accept = all(sample_values.get(var) == val for var, val in evidence.items())
        
        if accept:
            accepted_samples.append(sample_values[query].item())
    
    return torch.tensor(accepted_samples)
```

**Performance Results**:
```
Rejection sampling: 1000 samples from 2006 attempts
Time: ~0.5 seconds
Accuracy: Perfect match with importance sampling ✓
```

---

## **Discovery #6: Why Discrete Variables Break MCMC**

**Mathematical Reason**:
- **MCMC relies on**: Small perturbations → small probability changes
- **Discrete space**: Any change is a "jump" to completely different value
- **No gradients**: Can't use efficient proposal mechanisms
- **Poor mixing**: Hard to explore the space effectively

**Analogy**: 
- **Continuous MCMC**: Like walking smoothly across a landscape
- **Discrete MCMC**: Like teleporting randomly between cities

---

## **Discovery #7: Hybrid Networks (Continuous + Discrete)**

**The Question**: Would MCMC work on mixed continuous/discrete networks?

**Approaches That Could Work**:

### **1. Marginalization** (Best)
```python
# Integrate out discrete variables analytically
# Sample only continuous variables with NUTS
# Works when discrete variables have few states
```

### **2. Gibbs Sampling**
```python
# Alternate between:
# - NUTS for continuous variables | discrete fixed
# - Discrete sampling for discrete | continuous fixed
```

### **3. Continuous Relaxation**
```python
# Replace discrete with Gumbel-Softmax
# Sample continuous relaxations with NUTS
# Discretize results post-hoc
```

**Reality**: Even hybrid approaches are complex and problem-specific.

---

## **Key Takeaways**

### **What We Learned About MCMC:**

1. **MCMC ≠ Silver Bullet**: Great for continuous, poor for discrete
2. **Silent Failures**: Always validate sample collection
3. **Vectorization Conflicts**: Don't mix `pyro.plate` with MCMC
4. **Tool Selection Matters**: Choose the right algorithm for variable types

### **When to Use Each Method:**

| Method | Best For | Avoid When |
|--------|----------|------------|
| **MCMC (NUTS)** | Continuous variables, complex posteriors | Discrete variables |
| **Importance Sampling** | Discrete networks, exact inference | Very high-dimensional |
| **Rejection Sampling** | Simple discrete problems, debugging | Low-probability evidence |
| **SVI** | Large-scale problems, approximate OK | Critical accuracy needed |

### **The Hierarchy for Discrete Bayesian Networks:**

1. **Importance Sampling** ← Use this! ✅
2. **Rejection Sampling** ← Good for debugging
3. **Exact Inference** ← When computationally feasible  
4. **SVI** ← When approximate is acceptable
5. **MCMC** ← Avoid for pure discrete networks ❌

### **Red Flags to Watch For:**

- ⚠️ MCMC returns empty samples dictionary
- ⚠️ "100% complete" but no actual samples
- ⚠️ Using NUTS/HMC with Categorical distributions
- ⚠️ Mixing vectorized models with MCMC

---

## **Final Verdict**

**For the Asia Bayesian Network**: Stick with importance sampling! It's:
- Fast (`"importance", # really fast` in your comments)
- Accurate (matches exact inference)
- Designed for discrete networks
- Already working perfectly in `borrar.py`

**MCMC served its purpose**: The debugging process taught us why tool selection matters in probabilistic programming.
