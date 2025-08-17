"""
Inference functions for continuous Bayesian Networks (Linear Gaussian models).

This module provides utilities for performing exact inference on Linear Gaussian
Bayesian Networks by converting them to multivariate Gaussian distributions.
"""

import numpy as np
from scipy.stats import multivariate_normal


def format_continuous_query(variable, evidence=None, prob_range=None):
    """
    Generate formatted query string for continuous variables.
    
    Parameters:
    - variable: string, name of the variable to query
    - evidence: dict, evidence variables and their values (e.g., {'Xray': -1.0, 'Smoker': 2.0})
    - prob_range: tuple, (lower, upper) bounds for probability calculation
                 Use None for unbounded (e.g., (None, 0) for P(X < 0))
    
    Returns:
    - string: formatted query
    
    Examples:
    >>> format_continuous_query('Cancer', {'Xray': -1.0, 'Smoker': 2.0})
    'P(Cancer | Xray = -1.0, Smoker = 2.0)'
    
    >>> format_continuous_query('Pollution', {'Dyspnoea': 0.5}, prob_range=(0, 1))
    'P(0 < Pollution < 1 | Dyspnoea = 0.5)'
    
    >>> format_continuous_query('Cancer', {'Smoker': 2.0, 'Pollution': 0.8}, prob_range=(None, 0))
    'P(Cancer < 0 | Smoker = 2.0, Pollution = 0.8)'
    """
    if prob_range is not None:
        # Probability calculation query
        lower, upper = prob_range
        
        if lower is None:
            condition = f"{variable} < {upper}"
        elif upper is None:
            condition = f"{variable} > {lower}"
        else:
            condition = f"{lower} < {variable} < {upper}"
        
        if evidence:
            evidence_str = ', '.join([f"{k} = {v}" for k, v in evidence.items()])
            return f"P({condition} | {evidence_str})"
        else:
            return f"P({condition})"
    else:
        # Posterior estimation query
        if evidence:
            evidence_str = ', '.join([f"{k} = {v}" for k, v in evidence.items()])
            return f"P({variable} | {evidence_str})"
        else:
            return f"P({variable})"


def conditional_gaussian_inference(mean, cov, variable_names, evidence_vars, evidence_values, query_vars):
    """
    Perform conditional inference on a multivariate Gaussian distribution.
    
    Parameters:
    - mean: mean vector of the joint distribution
    - cov: covariance matrix of the joint distribution  
    - variable_names: list of variable names in order
    - evidence_vars: list of evidence variable names
    - evidence_values: list of evidence values (same order as evidence_vars)
    - query_vars: list of query variable names
    
    Returns:
    - conditional_mean: mean of query variables given evidence
    - conditional_cov: covariance matrix of query variables given evidence
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


def query_lgbn(model, query_var, evidence=None, prob_range=None):
    """
    Convenient function for querying a Linear Gaussian Bayesian Network.
    
    Parameters:
    - model: LinearGaussianBayesianNetwork object
    - query_var: string, name of the variable to query
    - evidence: dict, evidence variables and their values (e.g., {'Dyspnoea': 0.5})
    - prob_range: tuple, (lower, upper) bounds for probability calculation
                 Use None for unbounded (e.g., (None, 0) for P(X < 0))
    
    Returns:
    - dict with mean, std, variance, and probability (if range specified)
    """
    mean, cov = model.to_joint_gaussian()
    variable_names = list(model.nodes())
    
    if evidence is None:
        evidence = {}
    
    evidence_vars = list(evidence.keys())
    evidence_values = list(evidence.values())
    query_vars = [query_var]
    
    if len(evidence_vars) > 0:
        conditional_mean, conditional_cov = conditional_gaussian_inference(
            mean, cov, variable_names, evidence_vars, evidence_values, query_vars
        )
        result_mean = conditional_mean[0]
        result_var = conditional_cov[0, 0]
    else:
        # No evidence, just return marginal
        query_idx = variable_names.index(query_var)
        result_mean = mean[query_idx]
        result_var = cov[query_idx, query_idx]
    
    result = {
        'mean': result_mean,
        'variance': result_var,
        'std': np.sqrt(result_var)
    }
    
    # Calculate probability if range is specified
    if prob_range is not None:
        lower, upper = prob_range
        dist = multivariate_normal(mean=result_mean, cov=result_var)
        
        if lower is None:
            prob = dist.cdf(upper)
        elif upper is None:
            prob = 1 - dist.cdf(lower)
        else:
            prob = dist.cdf(upper) - dist.cdf(lower)
        
        result['probability'] = prob
        result['range'] = prob_range
    
    return result