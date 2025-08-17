"""
Inference functions for discrete Bayesian Networks.

This module provides utilities for performing exact inference on discrete Bayesian Networks.
"""

def format_probability_query(variable, value, evidence=None):
    """Generate formatted query string like P(dysp=no | smoke=yes, asia=no)"""
    if evidence:
        evidence_str = ', '.join([f"{k}={v}" for k, v in evidence.items()])
        return f"P({variable}={value} | {evidence_str})"
    return f"P({variable}={value})"

def query_probability(inference_engine, variable, value, evidence=None):
    """Run inference and return specific probability value using state index lookup"""
    query = inference_engine.query(variables=[variable], evidence=evidence)
    return query.values[query.state_names[variable].index(value)]