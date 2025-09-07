# What We Learned About SVI in Discrete Bayesian Networks

This documents our journey debugging why SVI was giving wrong results for the Asia network, and the key insights we discovered.

## **The Original Problem**

**Query**: P(Dysp | Smoke = no)  
**Expected**: [0.3191, 0.6809] (dysp=yes, dysp=no)  
**SVI Result**: [0.463, 0.537] → Later became [0.685, 0.315] ❌ **FLIPPED!**  
**Importance Sampling**: [0.319, 0.681] ✅ (correct baseline)

---

## **Discovery #1: Evidence Wasn't Being Passed to the Guide**

**The Bug**: Original SVI guide was sampling ALL variables, including observed ones:

```python
# WRONG: Guide always samples smoke, ignoring evidence
def guide():
    smoke = pyro.sample("smoke", dist.Categorical(smoke_param))  # Always samples!
    # ... other variables
```

**The Fix**: Guide must respect evidence:
```python
# CORRECT: Only sample unobserved variables  
def guide():
    if "smoke" not in evidence:
        smoke = pyro.sample("smoke", dist.Categorical(smoke_param))
    # Only sample what we don't observe
```

**Impact**: This was the PRIMARY cause of wrong results. Once fixed, SVI became much faster (no enumeration warnings) but still gave flipped results.

---

## **Discovery #2: Mean-Field Approximation Breaks Bayesian Networks**

**The Core Issue**: SVI with mean-field treats all variables as independent, destroying the conditional dependencies that make Bayesian networks work.

```python
# TRUE MODEL: dysp depends on bronc AND either
dysp = pyro.sample("dysp", dist.Categorical(dysp_probs[bronc, either]))

# MEAN-FIELD GUIDE: dysp treated as independent
dysp = pyro.sample("dysp", dist.Categorical(torch.softmax(dysp_param, -1)))
```

**Result**: 
- **True**: P(dysp=yes|smoke=no) = 0.319
- **SVI**: P(dysp=yes|smoke=no) = 0.685 ← **FLIPPED!**

**Why This Happens**:
1. Mean-field assumes posterior factorizes: q(x₁,x₂,...) = q(x₁)q(x₂)...
2. But in Bayesian networks, variables are strongly dependent
3. The approximation error manifests as completely wrong probabilities

---

## **Discovery #3: The "Flip" is Dangerous, Not Just Wrong**

**Medical Analogy**: 
- **True result**: 32% chance of disease → "Low risk, monitor"
- **SVI result**: 68% chance of disease → "High risk, immediate treatment"

This isn't just "slightly off" - it's the **opposite medical conclusion**!

**Learned Parameter Analysis**:
```python
# SVI learned: dysp_param: [0.686, 0.314]
# Interpretation: 68.6% yes, 31.4% no
# But truth is: 31.9% yes, 68.1% no
```

---

## **Discovery #4: Structured Guides Help But Don't Solve It**

**Attempted Fix**: Create guide that respects some dependencies:
```python
# Better guide with conditional structure
tub = pyro.sample("tub", dist.Categorical(tub_param[asia]))  # Depends on asia
dysp = pyro.sample("dysp", dist.Categorical(dysp_param[bronc, either]))  # Depends on parents
```

**Result**: [0.460, 0.540] - better than [0.685, 0.315] but still wrong!

**Lesson**: Even structured guides struggle with complex posterior dependencies.

---

## **Discovery #5: When SVI Becomes Extremely Slow**

**The Problem**: Using `AutoDiscreteParallel` caused exponential slowdown after step 1600.

**Why**: 
- Enumeration over all discrete variable combinations
- 8 binary variables = 2⁸ = 256 combinations to track
- Memory and computation explode as training progresses

**Solution**: Avoid auto-guides for discrete variables; use manual mean-field or structured guides.

---

## **Discovery #6: Index Confusion Led Us Astray**

**Initially Thought**: Importance sampling was wrong because we tested `smoke=0` vs `smoke=1`
**Reality**: 
- `smoke=0` means "smoke=yes" 
- `smoke=1` means "smoke=no"
- Once corrected, importance sampling was perfect: [0.319, 0.681] ✅

**Lesson**: Always validate tensor indexing when debugging probabilistic models!

---

## **Key Takeaways**

### **What We Learned About SVI:**
1. **Evidence handling is critical** - guides must respect observations
2. **Mean-field breaks Bayesian networks** - independence assumption is too strong
3. **Discrete auto-guides can explode** - manual guides often better
4. **Validation is essential** - always check against exact methods
5. **"Wrong" can mean "opposite"** - not just imprecise, but dangerous

### **When to Use SVI vs Exact Methods:**
- **Use SVI**: Large-scale problems where exact inference impossible
- **Use Exact**: Critical applications, when model size allows
- **Always**: Cross-validate SVI against exact methods on smaller problems

### **SVI is Not "Bad" - It Has Limitations:**
SVI works well when:
- Variables are weakly correlated
- Posterior is unimodal and simple
- Approximate answers are acceptable

SVI fails when:
- Strong conditional dependencies (like Bayesian networks)
- Multimodal posteriors
- Exact answers required for safety/correctness 