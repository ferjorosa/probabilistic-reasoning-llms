"""
Simple translation from pgmpy Linear Gaussian Bayesian Network to joint Gaussian distribution.

This fixes the variable ordering issues in pgmpy's built-in to_joint_gaussian() method.
"""

import numpy as np
from typing import Tuple, List
import warnings

try:
    from pgmpy.models import LinearGaussianBayesianNetwork
    import networkx as nx
except ImportError:
    warnings.warn("pgmpy not found. Install with: pip install pgmpy")
    LinearGaussianBayesianNetwork = None
    nx = None


def pgmpy_to_joint_gaussian(model) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert a pgmpy Linear Gaussian Bayesian Network to a joint Gaussian distribution.
    
    This function correctly handles variable ordering issues that exist in pgmpy's
    built-in to_joint_gaussian() method.
    
    Args:
        model: pgmpy LinearGaussianBayesianNetwork instance
        
    Returns:
        mean: Mean vector of the joint distribution
        cov: Covariance matrix of the joint distribution  
        variables: List of variable names in the order they appear in mean/cov
    """
    if LinearGaussianBayesianNetwork is None or nx is None:
        raise ImportError("pgmpy and networkx are required. Install with: pip install pgmpy networkx")
    
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError("Input must be a pgmpy LinearGaussianBayesianNetwork")
    
    # Get variables in topological order using NetworkX
    # Since LinearGaussianBayesianNetwork inherits from NetworkX DiGraph,
    # we can use NetworkX's topological_sort function
    variables = list(nx.topological_sort(model))
    n = len(variables)
    
    # Create mapping from variable name to index
    var_to_idx = {var: i for i, var in enumerate(variables)}
    
    # Initialize mean vector and covariance matrix
    mean = np.zeros(n)
    cov = np.zeros((n, n))
    
    # Extract CPDs and organize by variable
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}
    
    # Process variables in topological order
    for var in variables:
        idx = var_to_idx[var]
        cpd = cpds[var]
        
        # Get parents from the CPD
        parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
        
        if not parents:
            # Root node: X ~ N(mean, variance)
            mean[idx] = cpd.beta[0]
            cov[idx, idx] = cpd.std ** 2  # Convert std to variance
        else:
            # Child node: X = intercept + sum(coeff_i * parent_i) + noise
            
            # Extract parameters from CPD
            intercept = cpd.beta[0]  # The constant term
            coefficients = {}
            
            # Get coefficients for each parent
            for i, parent in enumerate(parents):
                # The coefficient is in the beta array at position i+1
                coefficients[parent] = cpd.beta[i + 1]
            
            noise_variance = cpd.std ** 2  # Convert std to variance
            
            # Mean: E[X] = intercept + sum(coeff_i * E[parent_i])
            mean[idx] = intercept
            for parent in parents:
                parent_idx = var_to_idx[parent]
                coeff = coefficients[parent]
                mean[idx] += coeff * mean[parent_idx]
            
            # Variance: Var[X] = sum(coeff_i^2 * Var[parent_i]) + 
            #                   2 * sum_i sum_j coeff_i * coeff_j * Cov[parent_i, parent_j] +
            #                   noise_variance
            var_x = noise_variance
            
            for i, parent_i in enumerate(parents):
                parent_i_idx = var_to_idx[parent_i]
                coeff_i = coefficients[parent_i]
                
                # Add coeff_i^2 * Var[parent_i]
                var_x += coeff_i**2 * cov[parent_i_idx, parent_i_idx]
                
                # Add cross terms: 2 * coeff_i * coeff_j * Cov[parent_i, parent_j]
                for j, parent_j in enumerate(parents):
                    if i < j:  # avoid double counting
                        parent_j_idx = var_to_idx[parent_j]
                        coeff_j = coefficients[parent_j]
                        var_x += 2 * coeff_i * coeff_j * cov[parent_i_idx, parent_j_idx]
            
            cov[idx, idx] = var_x
            
            # Covariances: Cov[X, Y] for all other variables Y
            for other_var in variables:
                other_idx = var_to_idx[other_var]
                if other_idx != idx:
                    # Cov[X, Y] = sum(coeff_i * Cov[parent_i, Y])
                    cov_xy = 0
                    for parent in parents:
                        parent_idx = var_to_idx[parent]
                        coeff = coefficients[parent]
                        cov_xy += coeff * cov[parent_idx, other_idx]
                    
                    cov[idx, other_idx] = cov_xy
                    cov[other_idx, idx] = cov_xy  # symmetric
    
    return mean, cov, variables