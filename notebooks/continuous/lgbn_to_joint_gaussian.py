"""
Correct implementation to transform a Linear Gaussian Bayesian Network 
into a joint Gaussian distribution.

This implementation fixes the bugs in pgmpy's to_joint_gaussian() method.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LinearGaussianCPD:
    """Represents a Linear Gaussian Conditional Probability Distribution"""
    variable: str
    parents: List[str]
    intercept: float
    coefficients: Dict[str, float]  # parent -> coefficient
    variance: float


class LinearGaussianBN:
    """
    A correct implementation of Linear Gaussian Bayesian Network
    that can be converted to joint Gaussian distribution.
    """
    
    def __init__(self):
        self.variables = []
        self.cpds = {}
        self.parents = {}
        self.children = {}
    
    def add_variable(self, variable: str):
        """Add a variable to the network"""
        if variable not in self.variables:
            self.variables.append(variable)
            self.parents[variable] = []
            self.children[variable] = []
    
    def add_edge(self, parent: str, child: str):
        """Add a directed edge from parent to child"""
        self.add_variable(parent)
        self.add_variable(child)
        
        if parent not in self.parents[child]:
            self.parents[child].append(parent)
        if child not in self.children[parent]:
            self.children[parent].append(child)
    
    def add_cpd(self, cpd: LinearGaussianCPD):
        """Add a Conditional Probability Distribution"""
        self.add_variable(cpd.variable)
        self.cpds[cpd.variable] = cpd
        
        # Add edges for parents
        for parent in cpd.parents:
            self.add_edge(parent, cpd.variable)
    
    def get_topological_order(self) -> List[str]:
        """Get variables in topological order (parents before children)"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Cycle detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            for parent in self.parents[node]:
                visit(parent)
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        for variable in self.variables:
            if variable not in visited:
                visit(variable)
        
        return order
    
    def to_joint_gaussian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the Linear Gaussian BN to a joint Gaussian distribution.
        
        Returns:
            mean: Mean vector of the joint distribution
            cov: Covariance matrix of the joint distribution
        """
        n = len(self.variables)
        topo_order = self.get_topological_order()
        
        # Create mapping from variable name to index
        var_to_idx = {var: i for i, var in enumerate(self.variables)}
        
        # Initialize mean vector and covariance matrix
        mean = np.zeros(n)
        cov = np.zeros((n, n))
        
        # Process variables in topological order
        for var in topo_order:
            idx = var_to_idx[var]
            cpd = self.cpds[var]
            
            if not cpd.parents:
                # Root node: X ~ N(intercept, variance)
                mean[idx] = cpd.intercept
                cov[idx, idx] = cpd.variance
            else:
                # Child node: X = intercept + sum(coeff_i * parent_i) + noise
                # where noise ~ N(0, variance)
                
                # Mean: E[X] = intercept + sum(coeff_i * E[parent_i])
                mean[idx] = cpd.intercept
                for parent in cpd.parents:
                    parent_idx = var_to_idx[parent]
                    coeff = cpd.coefficients[parent]
                    mean[idx] += coeff * mean[parent_idx]
                
                # Variance: Var[X] = sum(coeff_i^2 * Var[parent_i]) + 
                #                   2 * sum_i sum_j coeff_i * coeff_j * Cov[parent_i, parent_j] +
                #                   noise_variance
                var_x = cpd.variance  # noise variance
                
                for i, parent_i in enumerate(cpd.parents):
                    parent_i_idx = var_to_idx[parent_i]
                    coeff_i = cpd.coefficients[parent_i]
                    
                    # Add coeff_i^2 * Var[parent_i]
                    var_x += coeff_i**2 * cov[parent_i_idx, parent_i_idx]
                    
                    # Add cross terms: 2 * coeff_i * coeff_j * Cov[parent_i, parent_j]
                    for j, parent_j in enumerate(cpd.parents):
                        if i < j:  # avoid double counting
                            parent_j_idx = var_to_idx[parent_j]
                            coeff_j = cpd.coefficients[parent_j]
                            var_x += 2 * coeff_i * coeff_j * cov[parent_i_idx, parent_j_idx]
                
                cov[idx, idx] = var_x
                
                # Covariances: Cov[X, Y] for all other variables Y
                for other_var in self.variables:
                    other_idx = var_to_idx[other_var]
                    if other_idx != idx:
                        # Cov[X, Y] = sum(coeff_i * Cov[parent_i, Y])
                        cov_xy = 0
                        for parent in cpd.parents:
                            parent_idx = var_to_idx[parent]
                            coeff = cpd.coefficients[parent]
                            cov_xy += coeff * cov[parent_idx, other_idx]
                        
                        cov[idx, other_idx] = cov_xy
                        cov[other_idx, idx] = cov_xy  # symmetric
        
        return mean, cov
    
    def print_network(self):
        """Print the network structure and CPDs"""
        print("Network Structure:")
        for var in self.variables:
            parents = self.parents[var]
            if parents:
                print(f"  {var} <- {parents}")
            else:
                print(f"  {var} (root)")
        
        print("\nCPDs:")
        for var in self.variables:
            cpd = self.cpds[var]
            if not cpd.parents:
                print(f"  P({var}) = N({cpd.intercept:.3f}, {np.sqrt(cpd.variance):.3f})")
            else:
                terms = [f"{cpd.intercept:.3f}"]
                for parent in cpd.parents:
                    coeff = cpd.coefficients[parent]
                    if coeff >= 0:
                        terms.append(f"{coeff:.3f}*{parent}")
                    else:
                        terms.append(f"{coeff:.3f}*{parent}")
                equation = " + ".join(terms).replace("+ -", "- ")
                print(f"  P({var} | {', '.join(cpd.parents)}) = N({equation}, {np.sqrt(cpd.variance):.3f})")


def conditional_gaussian_inference(mean: np.ndarray, cov: np.ndarray, 
                                 variable_names: List[str],
                                 evidence_vars: List[str], evidence_values: List[float],
                                 query_vars: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform conditional inference on a multivariate Gaussian distribution.
    
    Returns:
        conditional_mean: Mean of query variables given evidence
        conditional_cov: Covariance matrix of query variables given evidence
    """
    # Get indices for evidence and query variables
    evidence_indices = [variable_names.index(var) for var in evidence_vars]
    query_indices = [variable_names.index(var) for var in query_vars]
    
    # Split mean vector
    mu_q = mean[query_indices]  # query variables mean
    mu_e = mean[evidence_indices]  # evidence variables mean
    
    # Split covariance matrix
    cov_qq = cov[np.ix_(query_indices, query_indices)]  # query-query covariance
    cov_ee = cov[np.ix_(evidence_indices, evidence_indices)]  # evidence-evidence covariance
    cov_qe = cov[np.ix_(query_indices, evidence_indices)]  # query-evidence covariance
    
    # Convert evidence values to numpy array
    evidence_values = np.array(evidence_values)
    
    # Conditional mean: μ_q + Σ_qe * Σ_ee^(-1) * (x_e - μ_e)
    conditional_mean = mu_q + cov_qe @ np.linalg.inv(cov_ee) @ (evidence_values - mu_e)
    
    # Conditional covariance: Σ_qq - Σ_qe * Σ_ee^(-1) * Σ_eq
    conditional_cov = cov_qq - cov_qe @ np.linalg.inv(cov_ee) @ cov_qe.T
    
    return conditional_mean, conditional_cov


def create_cancer_network() -> LinearGaussianBN:
    """Create the cancer network from the example"""
    bn = LinearGaussianBN()
    
    # Add CPDs
    bn.add_cpd(LinearGaussianCPD(
        variable="Pollution",
        parents=[],
        intercept=0.305,
        coefficients={},
        variance=1.04**2
    ))
    
    bn.add_cpd(LinearGaussianCPD(
        variable="Smoker", 
        parents=[],
        intercept=1.446,
        coefficients={},
        variance=0.102**2
    ))
    
    bn.add_cpd(LinearGaussianCPD(
        variable="Cancer",
        parents=["Pollution", "Smoker"],
        intercept=0.244,
        coefficients={"Pollution": 0.678, "Smoker": -0.586},
        variance=0.909**2
    ))
    
    bn.add_cpd(LinearGaussianCPD(
        variable="Xray",
        parents=["Cancer"],
        intercept=-0.458,
        coefficients={"Cancer": -0.623},
        variance=0.135**2
    ))
    
    bn.add_cpd(LinearGaussianCPD(
        variable="Dyspnoea",
        parents=["Cancer"],
        intercept=-0.503,
        coefficients={"Cancer": 1.218},
        variance=0.271**2
    ))
    
    return bn


if __name__ == "__main__":
    # Create the cancer network
    bn = create_cancer_network()
    
    print("=== Linear Gaussian Bayesian Network ===")
    bn.print_network()
    
    # Convert to joint Gaussian
    mean, cov = bn.to_joint_gaussian()
    
    print(f"\n=== Joint Gaussian Distribution ===")
    print(f"Variables: {bn.variables}")
    print(f"Mean vector: {mean}")
    print(f"Covariance matrix:")
    print(cov)
    
    print(f"\n=== Marginal Distributions ===")
    for i, var in enumerate(bn.variables):
        print(f"{var}: Mean = {mean[i]:.4f}, Std = {np.sqrt(cov[i,i]):.4f}")
    
    print(f"\n=== Conditional Inference Test ===")
    # Test P(Pollution | Dyspnoea = 0.5)
    evidence_vars = ["Dyspnoea"]
    evidence_values = [0.5]
    query_vars = ["Pollution"]
    
    conditional_mean, conditional_cov = conditional_gaussian_inference(
        mean, cov, bn.variables, evidence_vars, evidence_values, query_vars
    )
    
    print(f"P(Pollution | Dyspnoea = 0.5):")
    print(f"  Mean: {conditional_mean[0]:.4f}")
    print(f"  Std: {np.sqrt(conditional_cov[0,0]):.4f}")
    print(f"  Variance: {conditional_cov[0,0]:.4f}")
    
    # Compare with Pyro results
    print(f"\n=== Comparison with Pyro ===")
    print(f"Our implementation: Mean = {conditional_mean[0]:.4f}, Std = {np.sqrt(conditional_cov[0,0]):.4f}")
    print(f"Pyro (from earlier): Mean ≈ 0.307, Std ≈ 1.04")
    print(f"pgmpy (buggy): Mean = 0.8406, Std = 0.8544")
