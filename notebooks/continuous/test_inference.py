import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, MCMC, NUTS
from pyro import poutine


def make_lgn_params(device):
    """Linear Gaussian Network parameters"""
    return {
        'pollution': {'mean': torch.tensor(0.305, device=device), 'std': torch.tensor(1.04, device=device)},
        'smoker': {'mean': torch.tensor(1.446, device=device), 'std': torch.tensor(0.102, device=device)},
        'cancer': {
            'coeff_pollution': torch.tensor(0.678, device=device),
            'coeff_smoker': torch.tensor(-0.586, device=device),
            'intercept': torch.tensor(0.244, device=device),
            'std': torch.tensor(0.909, device=device)
        },
        'xray': {
            'coeff_cancer': torch.tensor(-0.623, device=device),
            'intercept': torch.tensor(-0.458, device=device),
            'std': torch.tensor(0.135, device=device)
        },
        'dyspnoea': {
            'coeff_cancer': torch.tensor(1.218, device=device),
            'intercept': torch.tensor(-0.503, device=device),
            'std': torch.tensor(0.271, device=device)
        }
    }


def make_vectorized_model(num_samples, device):
    """Create vectorized Pyro model using pyro.plate"""
    params = make_lgn_params(device)
    
    def vectorized_model():
        with pyro.plate("particles", num_samples):
            pollution = pyro.sample("pollution", dist.Normal(params['pollution']['mean'], params['pollution']['std']))
            smoker = pyro.sample("smoker", dist.Normal(params['smoker']['mean'], params['smoker']['std']))
            
            cancer_mean = (params['cancer']['coeff_pollution'] * pollution + 
                          params['cancer']['coeff_smoker'] * smoker + 
                          params['cancer']['intercept'])
            cancer = pyro.sample("cancer", dist.Normal(cancer_mean, params['cancer']['std']))
            
            xray_mean = params['xray']['coeff_cancer'] * cancer + params['xray']['intercept']
            xray = pyro.sample("xray", dist.Normal(xray_mean, params['xray']['std']))
            
            dyspnoea_mean = params['dyspnoea']['coeff_cancer'] * cancer + params['dyspnoea']['intercept']
            dyspnoea = pyro.sample("dyspnoea", dist.Normal(dyspnoea_mean, params['dyspnoea']['std']))
            
            return dyspnoea
    
    return vectorized_model


def run_importance_sampling(num_samples, device, query, evidence):
    """Vectorized Importance Sampling"""
    vectorized_model = make_vectorized_model(num_samples, device)
    
    if evidence:
        evidence_tensors = {k: torch.tensor(v, device=device, dtype=torch.float32).expand(num_samples) 
                           for k, v in evidence.items()}
        conditioned = poutine.condition(vectorized_model, data=evidence_tensors)
        
        # Debug: print evidence to make sure it's being applied
        print(f"  Debug: Evidence applied: {evidence}")
        
        importance = Importance(conditioned, num_samples=1)
        posterior = importance.run()
        traces = list(posterior.exec_traces)
        samples = traces[0].nodes[query]["value"]
    else:
        # No evidence case
        importance = Importance(vectorized_model, num_samples=1)
        posterior = importance.run()
        traces = list(posterior.exec_traces)
        samples = traces[0].nodes[query]["value"]
    
    return samples


def run_mcmc_inference(num_samples, device, query, evidence):
    """MCMC inference using NUTS"""
    # For MCMC, use a non-vectorized model
    params = make_lgn_params(device)
    
    def model():
        pollution = pyro.sample("pollution", dist.Normal(params['pollution']['mean'], params['pollution']['std']))
        smoker = pyro.sample("smoker", dist.Normal(params['smoker']['mean'], params['smoker']['std']))
        
        cancer_mean = (params['cancer']['coeff_pollution'] * pollution + 
                      params['cancer']['coeff_smoker'] * smoker + 
                      params['cancer']['intercept'])
        cancer = pyro.sample("cancer", dist.Normal(cancer_mean, params['cancer']['std']))
        
        xray_mean = params['xray']['coeff_cancer'] * cancer + params['xray']['intercept']
        xray = pyro.sample("xray", dist.Normal(xray_mean, params['xray']['std']))
        
        dyspnoea_mean = params['dyspnoea']['coeff_cancer'] * cancer + params['dyspnoea']['intercept']
        dyspnoea = pyro.sample("dyspnoea", dist.Normal(dyspnoea_mean, params['dyspnoea']['std']))
        
        return dyspnoea
    
    if evidence:
        evidence_tensors = {k: torch.tensor(v, device=device, dtype=torch.float32) 
                           for k, v in evidence.items()}
        conditioned = poutine.condition(model, data=evidence_tensors)
        print(f"  Debug: MCMC Evidence applied: {evidence}")
    else:
        conditioned = model
    
    # Run MCMC
    nuts_kernel = NUTS(conditioned, adapt_step_size=True, max_tree_depth=10)
    mcmc = MCMC(nuts_kernel, num_samples=min(1000, num_samples), warmup_steps=200, num_chains=1)
    mcmc.run()
    
    # Get samples
    samples_dict = mcmc.get_samples()
    samples = samples_dict[query]
    
    # If we need more samples, repeat
    if len(samples) < num_samples:
        repeat_factor = num_samples // len(samples) + 1
        samples = samples.repeat(repeat_factor)[:num_samples]
    
    return samples


# Test the query: P(Pollution | Dyspnoea = 0.5)
if __name__ == "__main__":
    device = torch.device("cpu")
    pyro.clear_param_store()
    
    num_samples = 100000
    
    # First test marginals (no evidence) to check if model is correct
    print("=== MARGINAL TESTS (No Evidence) ===")
    
    # Test Pollution marginal: should be N(0.305, 1.04)
    samples = run_importance_sampling(num_samples, device, "pollution", {})
    print(f"Pollution marginal - Mean: {samples.mean().item():.4f} (expected: 0.305), Std: {samples.std().item():.4f} (expected: 1.04)")
    
    # Test Smoker marginal: should be N(1.446, 0.102)  
    samples = run_importance_sampling(num_samples, device, "smoker", {})
    print(f"Smoker marginal - Mean: {samples.mean().item():.4f} (expected: 1.446), Std: {samples.std().item():.4f} (expected: 0.102)")
    
    # Test Cancer marginal: should be -0.3966
    samples = run_importance_sampling(num_samples, device, "cancer", {})
    print(f"Cancer marginal - Mean: {samples.mean().item():.4f} (expected: -0.397), Std: {samples.std().item():.4f} (expected: 1.152)")
    
    print("\n=== CONDITIONAL TEST ===")
    
    # Test a simple conditional: P(Cancer | Smoker = 2.0)
    # Manual calculation: E[Cancer | Smoker=2.0] = 0.678*E[Pollution] + (-0.586)*2.0 + 0.244
    #                                            = 0.678*0.305 + (-0.586)*2.0 + 0.244
    #                                            = 0.207 - 1.172 + 0.244 = -0.721
    print("Simple test: P(Cancer | Smoker = 2.0)")
    samples = run_importance_sampling(num_samples, device, "cancer", {"smoker": 2.0})
    print(f"Pyro - Mean: {samples.mean().item():.4f} (manual calc: -0.721)")
    
    # Now test the original conditional query  
    evidence = {"dyspnoea": 0.5}
    query = "pollution"
    
    print("\nOriginal query: P(Pollution | Dyspnoea = 0.5)")
    
    # Test with Importance Sampling
    samples_is = run_importance_sampling(num_samples, device, query, evidence)
    mean_is = samples_is.mean().item()
    std_is = samples_is.std().item()
    
    print(f"Pyro Importance Sampling - Mean: {mean_is:.4f}, Std: {std_is:.4f}")
    
    # Test with MCMC
    samples_mcmc = run_mcmc_inference(10000, device, query, evidence)  # Fewer samples for MCMC
    mean_mcmc = samples_mcmc.mean().item()
    std_mcmc = samples_mcmc.std().item()
    
    print(f"Pyro MCMC - Mean: {mean_mcmc:.4f}, Std: {std_mcmc:.4f}")
    print(f"Our exact implementation - Mean: 0.9550, Std: 0.8312")
    print(f"pgmpy (buggy) - Mean: 0.8406, Std: 0.8544")