import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine
import matplotlib.pyplot as plt

# Simple model to test ELBO vs flip detection
def simple_model():
    # Simple Bayesian network: X -> Y
    x = pyro.sample("x", dist.Categorical(torch.tensor([0.3, 0.7])))  # P(x=0)=0.3
    
    # Y depends on X
    y_probs = torch.tensor([[0.8, 0.2],   # P(y=0|x=0)=0.8, P(y=1|x=0)=0.2  
                           [0.1, 0.9]])   # P(y=0|x=1)=0.1, P(y=1|x=1)=0.9
    y = pyro.sample("y", dist.Categorical(y_probs[x]))
    return y

def test_svi_with_elbo_tracking():
    # Evidence: observe x=0
    evidence = {"x": torch.tensor(0)}
    conditioned = poutine.condition(simple_model, data=evidence)
    
    print("=== Testing ELBO vs Flip Detection ===")
    print("True: P(y=0|x=0) = 0.8, P(y=1|x=0) = 0.2")
    print()
    
    # Test multiple random initializations
    results = []
    
    for trial in range(5):
        print(f"--- Trial {trial + 1} ---")
        pyro.clear_param_store()
        
        # Mean-field guide
        def guide():
            y_param = pyro.param(f"y_param_{trial}", torch.randn(2))  # Random init
            pyro.sample("y", dist.Categorical(logits=y_param))
        
        # Track ELBO during training
        svi = SVI(conditioned, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
        
        losses = []
        for step in range(200):
            loss = svi.step()
            losses.append(loss)
        
        # Get final result
        final_loss = losses[-1]
        y_param = pyro.param(f"y_param_{trial}")
        final_probs = torch.softmax(y_param, -1).detach().numpy()
        
        # Check if flipped
        is_flipped = final_probs[0] < 0.5  # Should be ~0.8
        
        results.append({
            'trial': trial + 1,
            'final_loss': final_loss,
            'probs': final_probs,
            'is_flipped': is_flipped,
            'losses': losses
        })
        
        print(f"Final ELBO: {final_loss:.3f}")
        print(f"Learned P(y=0|x=0): {final_probs[0]:.3f} (should be 0.8)")
        print(f"Flipped: {'YES' if is_flipped else 'NO'}")
        print()
    
    # Analyze results
    print("=== Analysis ===")
    flipped_trials = [r for r in results if r['is_flipped']]
    correct_trials = [r for r in results if not r['is_flipped']]
    
    if flipped_trials and correct_trials:
        avg_loss_flipped = sum(r['final_loss'] for r in flipped_trials) / len(flipped_trials)
        avg_loss_correct = sum(r['final_loss'] for r in correct_trials) / len(correct_trials)
        
        print(f"Average ELBO for FLIPPED trials: {avg_loss_flipped:.3f}")
        print(f"Average ELBO for CORRECT trials: {avg_loss_correct:.3f}")
        print(f"ELBO difference: {abs(avg_loss_flipped - avg_loss_correct):.3f}")
        
        if abs(avg_loss_flipped - avg_loss_correct) < 0.1:
            print("🚨 ELBO cannot distinguish flipped from correct!")
        else:
            print("✅ ELBO can detect the difference")
    else:
        print("All trials converged to the same mode")
    
    return results

if __name__ == "__main__":
    results = test_svi_with_elbo_tracking()
