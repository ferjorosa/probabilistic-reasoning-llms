import time
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, EmpiricalMarginal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
from pyro.optim import Adam
from pyro import poutine


# --- CPTs wrapped into a helper ---
def make_cpts(device):
    asia_probs = torch.tensor([0.01, 0.99], device=device)
    smoke_probs = torch.tensor([0.5, 0.5], device=device)

    tub_probs = torch.tensor([
        [0.05, 0.95],
        [0.01, 0.99],
    ], device=device)

    lung_probs = torch.tensor([
        [0.1, 0.9],
        [0.01, 0.99],
    ], device=device)

    bronc_probs = torch.tensor([
        [0.6, 0.4],
        [0.3, 0.7],
    ], device=device)

    either_probs = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]],  # lung=yes
        [[1.0, 0.0], [0.0, 1.0]],  # lung=no
    ], device=device)

    xray_probs = torch.tensor([
        [0.98, 0.02],
        [0.05, 0.95],
    ], device=device)

    dysp_probs = torch.tensor([
        [[0.9, 0.1], [0.8, 0.2]],
        [[0.7, 0.3], [0.1, 0.9]],
    ], device=device)

    return asia_probs, smoke_probs, tub_probs, lung_probs, bronc_probs, either_probs, xray_probs, dysp_probs

# --- Model factory (closes over CPTs) ---
def make_model(device):
    asia_probs, smoke_probs, tub_probs, lung_probs, bronc_probs, either_probs, xray_probs, dysp_probs = make_cpts(device)

    def model():
        asia = pyro.sample("asia", dist.Categorical(asia_probs))
        smoke = pyro.sample("smoke", dist.Categorical(smoke_probs))
        tub = pyro.sample("tub", dist.Categorical(tub_probs[asia]))
        lung = pyro.sample("lung", dist.Categorical(lung_probs[smoke]))
        bronc = pyro.sample("bronc", dist.Categorical(bronc_probs[smoke]))
        either = pyro.sample("either", dist.Categorical(either_probs[lung, tub]))
        xray = pyro.sample("xray", dist.Categorical(xray_probs[either]))
        dysp = pyro.sample("dysp", dist.Categorical(dysp_probs[bronc, either]))
        return
    return model

def run_sequential_inference(num_samples=5000, device="cpu", query="dysp", evidence=None):
    if evidence is None:
        evidence = {}

    device = torch.device(device)
    pyro.clear_param_store()

    # Build model with CPTs on device
    model = make_model(device)

    # Convert evidence to tensors on the same device
    evidence_tensors = {k: torch.tensor(v, device=device) for k, v in evidence.items()}

    # Conditioned model
    conditioned = poutine.condition(model, data=evidence_tensors)

    # Importance sampling
    start = time.time()
    importance = Importance(conditioned, num_samples=num_samples)
    posterior = importance.run()
    elapsed = time.time() - start

    # Extract samples
    marginal = EmpiricalMarginal(posterior, sites=query)
    samples = [marginal().item() for _ in range(num_samples)]
    samples = torch.tensor(samples)

    counts = torch.bincount(samples, minlength=2).float()
    probs = counts / counts.sum()

    return probs, elapsed

# ===== Base Model Setup =====
def make_vectorized_model(num_samples, device):
    """Create a vectorized Pyro model using pyro.plate"""
    asia_probs, smoke_probs, tub_probs, lung_probs, bronc_probs, either_probs, xray_probs, dysp_probs = make_cpts(device)
    
    def vectorized_model():
        with pyro.plate("particles", num_samples):
            asia = pyro.sample("asia", dist.Categorical(asia_probs))
            smoke = pyro.sample("smoke", dist.Categorical(smoke_probs))
            tub = pyro.sample("tub", dist.Categorical(tub_probs[asia]))
            lung = pyro.sample("lung", dist.Categorical(lung_probs[smoke]))
            bronc = pyro.sample("bronc", dist.Categorical(bronc_probs[smoke]))
            either = pyro.sample("either", dist.Categorical(either_probs[lung, tub]))
            xray = pyro.sample("xray", dist.Categorical(xray_probs[either]))
            dysp = pyro.sample("dysp", dist.Categorical(dysp_probs[bronc, either]))
            return dysp
    
    return vectorized_model


# ===== Importance Sampling =====
def run_importance_sampling(num_samples, device, query, evidence):
    """Vectorized Importance Sampling"""
    
    vectorized_model = make_vectorized_model(num_samples, device)
    evidence_tensors = {k: torch.tensor(v, device=device).expand(num_samples) for k, v in evidence.items()}
    conditioned = poutine.condition(vectorized_model, data=evidence_tensors)
    
    start = time.time()
    importance = Importance(conditioned, num_samples=1)
    posterior = importance.run()
    traces = list(posterior.exec_traces)
    samples = traces[0].nodes[query]["value"]
    elapsed = time.time() - start
    
    return samples, elapsed


# ===== Stochastic Variational Inference =====
def run_svi_inference(num_samples, device, query, evidence):
    """Stochastic Variational Inference"""
    
    vectorized_model = make_vectorized_model(num_samples, device)
    evidence_tensors = {k: torch.tensor(v, device=device).expand(num_samples) for k, v in evidence.items()}
    conditioned = poutine.condition(vectorized_model, data=evidence_tensors)
    
    # Simple mean-field variational family that respects evidence
    def guide():
        with pyro.plate("particles", num_samples):
            # Only sample variables that are NOT observed
            if "asia" not in evidence:
                asia_param = pyro.param("asia_param", torch.ones(2, device=device))
                asia = pyro.sample("asia", dist.Categorical(torch.softmax(asia_param, -1)))
            
            if "smoke" not in evidence:
                smoke_param = pyro.param("smoke_param", torch.ones(2, device=device))
                smoke = pyro.sample("smoke", dist.Categorical(torch.softmax(smoke_param, -1)))
            
            if "tub" not in evidence:
                tub_param = pyro.param("tub_param", torch.ones(2, device=device))
                tub = pyro.sample("tub", dist.Categorical(torch.softmax(tub_param, -1)))
            
            if "lung" not in evidence:
                lung_param = pyro.param("lung_param", torch.ones(2, device=device))
                lung = pyro.sample("lung", dist.Categorical(torch.softmax(lung_param, -1)))
            
            if "bronc" not in evidence:
                bronc_param = pyro.param("bronc_param", torch.ones(2, device=device))
                bronc = pyro.sample("bronc", dist.Categorical(torch.softmax(bronc_param, -1)))
            
            if "either" not in evidence:
                either_param = pyro.param("either_param", torch.ones(2, device=device))
                either = pyro.sample("either", dist.Categorical(torch.softmax(either_param, -1)))
            
            if "xray" not in evidence:
                xray_param = pyro.param("xray_param", torch.ones(2, device=device))
                xray = pyro.sample("xray", dist.Categorical(torch.softmax(xray_param, -1)))
            
            if "dysp" not in evidence:
                dysp_param = pyro.param("dysp_param", torch.ones(2, device=device))
                dysp = pyro.sample("dysp", dist.Categorical(torch.softmax(dysp_param, -1)))
    
    start = time.time()
    # Run SVI optimization
    svi = SVI(conditioned, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
    
    # Optimize for a few steps
    for step in range(100):
        svi.step()
    
    # Sample from the learned guide
    guide_trace = poutine.trace(guide).get_trace()
    samples = guide_trace.nodes[query]["value"]
    elapsed = time.time() - start
    
    return samples, elapsed


# ===== MCMC with NUTS =====
# def run_mcmc_inference(num_samples, device, query, evidence):
#     """MCMC with NUTS - Optimized for faster execution"""
    
#     vectorized_model = make_vectorized_model(num_samples, device)
#     evidence_tensors = {k: torch.tensor(v, device=device).expand(num_samples) for k, v in evidence.items()}
#     conditioned = poutine.condition(vectorized_model, data=evidence_tensors)
    
#     start = time.time()
#     # Use fewer samples for faster testing - MCMC is slow for discrete models
#     mcmc_samples = min(50, num_samples // 200)  # Much fewer samples
#     warmup = min(10, mcmc_samples // 5)  # Short warmup
    
#     nuts_kernel = NUTS(conditioned, step_size=0.1, adapt_step_size=True, max_tree_depth=3)
#     mcmc = MCMC(nuts_kernel, num_samples=mcmc_samples, warmup_steps=warmup, num_chains=2)
#     mcmc.run()
    
#     # Get samples from all chains and repeat to match requested sample size
#     samples_dict = mcmc.get_samples()
#     mcmc_samples_tensor = samples_dict[query].flatten()  # Flatten all chains
    
#     # Repeat samples to match requested sample size (approximate)
#     repeat_factor = num_samples // len(mcmc_samples_tensor) + 1
#     samples = mcmc_samples_tensor.repeat(repeat_factor)[:num_samples]
#     elapsed = time.time() - start
    
#     return samples, elapsed

# ===== Unified Interface =====
def run_vectorized_inference_multi(num_samples=5000, device="cpu", query="dysp", evidence=None, algorithm="importance"):
    """
    Vectorized inference with multiple algorithm support
    
    Args:
        algorithm: "importance", "svi", "mcmc"
    """
    if evidence is None:
        evidence = {}
    
    device = torch.device(device)
    pyro.clear_param_store()
    
    # Route to appropriate algorithm
    if algorithm == "importance":
        samples, elapsed = run_importance_sampling(num_samples, device, query, evidence)
    elif algorithm == "svi":
        samples, elapsed = run_svi_inference(num_samples, device, query, evidence)
    # elif algorithm == "mcmc":
    #     samples, elapsed = run_mcmc_inference(num_samples, device, query, evidence)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Compute probabilities
    counts = torch.bincount(samples, minlength=2).float()
    probs = counts / counts.sum()
    
    return probs, elapsed


# Test multiple vectorized inference algorithms
print("=== Vectorized Inference Algorithm Comparison ===\n")

algorithms = [
    "importance", # really fast
    "svi", # slow
    # "mcmc"
]

devices = ["cpu", "cuda"]
n_samples = 100000
results = {}

for device in devices:
    print(f"=== {device.upper()} Results ===")
    results[device] = {}

    for algorithm in algorithms:
        try:
            print(f"Running {algorithm} on {device}...")
            probs, elapsed = run_vectorized_inference_multi(
                num_samples=n_samples,
                device=device,
                query="dysp",
                evidence={"smoke": 1},
                algorithm=algorithm
            )

            # Handle GPU tensor conversion for display
            if device == "cuda":
                probs_np = probs.cpu().numpy()
            else:
                probs_np = probs.numpy()

            results[device][algorithm] = {"probs": probs_np, "time": elapsed}
            print(f" {algorithm.upper()}: {probs_np} (time: {elapsed:.3f}s)")

        except Exception as e:
            print(f" {algorithm.upper()}: Failed - {e}")
            results[device][algorithm] = {"probs": None, "time": None}

    print()

# Summary comparison
print("=== Performance Summary ===")
for algorithm in algorithms:
    cpu_time = results["cpu"][algorithm]["time"]
    gpu_time = results["cuda"][algorithm]["time"]

    if cpu_time is not None and gpu_time is not None:
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{algorithm.upper()}: GPU {speedup:.2f}x faster than CPU ({gpu_time:.3f}s vs {cpu_time:.3f}s)")
    else:
        print(f"{algorithm.upper()}: Could not compare (one failed)")