"""
Unified inference for joint Gaussian distributions.

This module performs exact inference on multivariate Gaussian distributions,
supporting both marginal and conditional queries for single or multiple variables.

The main entry point is the `inference()` function, which automatically handles:
- Marginal inference: P(query_vars) when no evidence is provided
- Conditional inference: P(query_vars | evidence) when evidence is provided
- Flexible output formatting based on query type

Key mathematical operations:
- Marginal: Extract relevant dimensions from the joint distribution
- Conditional: Apply Gaussian conditioning formulas to compute P(Q|E=e)
  where Q are query variables, E are evidence variables, e are evidence values
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union


def inference(mean: np.ndarray, cov: np.ndarray, 
              variable_names: List[str],
              query_vars: Union[str, List[str]],
              evidence: Optional[Dict[str, float]] = None,
              return_format: str = 'auto') -> Union[Dict[str, float], Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Perform inference on multivariate Gaussian distributions.
    
    This function automatically detects the type of query and applies the appropriate
    mathematical operations for exact inference.
    
    Supported query types:
    1. Marginal inference: P(query_vars) when no evidence is provided
    2. Conditional inference: P(query_vars | evidence) when evidence is provided
    3. Single or multiple variable queries
    
    Mathematical formulations:
    - Marginal: Extract relevant dimensions from the joint distribution
    - Conditional: Apply standard Gaussian conditioning formulas:
      * μ_q|e = μ_q + Σ_qe @ Σ_ee^(-1) @ (e - μ_e)
      * Σ_q|e = Σ_qq - Σ_qe @ Σ_ee^(-1) @ Σ_eq
    
    Args:
        mean: Mean vector of the joint Gaussian distribution (shape: [n_vars])
        cov: Covariance matrix of the joint distribution (shape: [n_vars, n_vars])
        variable_names: Variable names corresponding to mean/cov indices.
                       Must match the length of the mean vector.
        query_vars: Variable(s) to query. Can be a single variable name (str)
                   or list of variable names.
        evidence: Evidence as {variable_name: observed_value}.
                 If None or empty, performs marginal inference.
                 Example: {"A": 1.5, "B": -0.3}
        return_format: Controls output format:
                      - 'auto': Dict for single variables, tuple for multiple
                      - 'dict': Always return dict with statistics
                      - 'arrays': Always return (mean, cov) tuple
    
    Returns:
        Format depends on return_format and number of query variables:
        
        Single variable with 'auto' or 'dict':
            Dict[str, float] with 'mean' and 'std' keys
            
        Multiple variables with 'auto' or 'arrays':
            Tuple[np.ndarray, np.ndarray]: (posterior_mean, posterior_cov)
            
        Multiple variables with 'dict':
            Dict[str, np.ndarray] mapping variable names to [mean, std] arrays
    
    Raises:
        ValueError: If variables are not found, overlap between query and evidence,
                   or covariance matrix is singular
        np.linalg.LinAlgError: If matrix operations fail
    
    Examples:
        # Marginal query for single variable
        >>> result = inference(mean, cov, ['A', 'B', 'C'], 'A')
        >>> # Returns: {'mean': 1.2, 'std': 0.707}
        
        # Conditional query for single variable  
        >>> result = inference(mean, cov, ['A', 'B', 'C'], 'A', evidence={'B': 2.0})
        >>> # Returns: {'mean': 1.8, 'std': 0.548}
        
        # Multi-variable marginal query
        >>> post_mean, post_cov = inference(mean, cov, ['A', 'B', 'C'], ['A', 'C'])
        >>> # Returns: (array([1.2, 0.8]), array([[0.5, 0.1], [0.1, 0.4]]))
        
        # Multi-variable conditional query
        >>> post_mean, post_cov = inference(mean, cov, ['A', 'B', 'C'], 
        ...                                ['A', 'C'], evidence={'B': 2.0})
        
        # Force dict format for multi-variable
        >>> result = inference(mean, cov, ['A', 'B', 'C'], ['A', 'C'], 
        ...                   return_format='dict')
        >>> # Returns: {'A': array([mean, std]), 'C': array([mean, std])}
    """
    
    # Input validation and normalization
    if evidence is None:
        evidence = {}
    
    # Normalize query_vars to list
    if isinstance(query_vars, str):
        query_vars_list = [query_vars]
        single_var_query = True
    else:
        query_vars_list = list(query_vars)
        single_var_query = len(query_vars_list) == 1
    
    # Validate all variables exist
    all_vars_to_check = query_vars_list + list(evidence.keys())
    for var in all_vars_to_check:
        if var not in variable_names:
            raise ValueError(f"Variable '{var}' not found in variable_names: {variable_names}")
    
    # Check for overlap between query and evidence variables
    evidence_vars = list(evidence.keys())
    overlap = set(query_vars_list) & set(evidence_vars)
    if overlap:
        raise ValueError(f"Variables cannot be both queried and evidence: {overlap}")
    
    # Determine indices for query and evidence variables
    query_indices = [variable_names.index(var) for var in query_vars_list]
    evidence_indices = [variable_names.index(var) for var in evidence_vars]
    
    # CASE 1: Marginal inference (no evidence)
    if len(evidence) == 0:
        # Simply extract the marginal distribution
        posterior_mean = mean[query_indices]
        posterior_cov = cov[np.ix_(query_indices, query_indices)]
        
    # CASE 2: Conditional inference (with evidence)  
    else:
        # Apply Gaussian conditioning formulas
        evidence_values = np.array([evidence[var] for var in evidence_vars])
        
        # Extract submatrices and subvectors
        mu_q = mean[query_indices]          # Query variables mean
        mu_e = mean[evidence_indices]       # Evidence variables mean
        cov_qq = cov[np.ix_(query_indices, query_indices)]      # Query-query covariance
        cov_ee = cov[np.ix_(evidence_indices, evidence_indices)] # Evidence-evidence covariance  
        cov_qe = cov[np.ix_(query_indices, evidence_indices)]   # Query-evidence covariance
        
        # Invert evidence covariance (with error handling)
        try:
            cov_ee_inv = np.linalg.inv(cov_ee)
        except np.linalg.LinAlgError:
            raise ValueError(f"Evidence covariance matrix is singular. Evidence variables: {evidence_vars}")
        
        # Conditional mean: μ_q + Σ_qe @ Σ_ee^(-1) @ (evidence_values - μ_e)
        posterior_mean = mu_q + cov_qe @ cov_ee_inv @ (evidence_values - mu_e)
        
        # Conditional covariance: Σ_qq - Σ_qe @ Σ_ee^(-1) @ Σ_eq  
        posterior_cov = cov_qq - cov_qe @ cov_ee_inv @ cov_qe.T
    
    # Format output based on return_format and number of variables
    if return_format == 'arrays':
        return posterior_mean, posterior_cov
    
    elif return_format == 'dict' or (return_format == 'auto' and single_var_query):
        if single_var_query:
            # Single variable: return simple dict
            return {
                'mean': float(posterior_mean[0]),
                'std': float(np.sqrt(posterior_cov[0, 0]))
            }
        else:
            # Multiple variables: return dict with per-variable stats
            result = {}
            for i, var in enumerate(query_vars_list):
                var_mean = float(posterior_mean[i])
                var_std = float(np.sqrt(posterior_cov[i, i]))
                result[var] = np.array([var_mean, var_std])
            return result
    
    else: # return_format == 'auto' and multi-variable
        return posterior_mean, posterior_cov
