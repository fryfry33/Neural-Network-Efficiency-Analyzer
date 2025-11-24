# -*- coding: utf-8 -*-
"""
Metrics calculations for neural network efficiency analysis
"""

import numpy as np
from typing import Dict


def compute_entropy_metrics(importance_matrix: np.ndarray) -> Dict:
    """
    Compute entropy-based metrics for importance distribution
    
    Args:
        importance_matrix: Matrix of importance values
        
    Returns:
        Dictionary containing entropy metrics
    """
    flat_importance = importance_matrix.flatten()
    total_sum = np.sum(flat_importance)
    
    if total_sum > 0:
        normalized = flat_importance / total_sum
    else:
        normalized = flat_importance
    
    # Entropy-based metrics
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    max_entropy = np.log(len(normalized))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'effective_params': np.exp(entropy),
        'utilization_ratio': np.exp(entropy) / len(flat_importance),
    }


def compute_sparsity_metrics(importance_matrix: np.ndarray) -> Dict:
    """
    Compute sparsity metrics for importance distribution
    
    Args:
        importance_matrix: Matrix of importance values
        
    Returns:
        Dictionary containing sparsity metrics
    """
    flat_importance = importance_matrix.flatten()
    
    # Sparsity metrics at different thresholds
    threshold_1e2 = np.sum(flat_importance < 1e-2) / len(flat_importance)
    threshold_1e3 = np.sum(flat_importance < 1e-3) / len(flat_importance)
    
    return {
        'sparsity_1e-2': threshold_1e2,
        'sparsity_1e-3': threshold_1e3,
    }


def compute_concentration_metrics(importance_matrix: np.ndarray) -> Dict:
    """
    Compute concentration metrics (Gini coefficient and top-k coverage)
    
    Args:
        importance_matrix: Matrix of importance values
        
    Returns:
        Dictionary containing concentration metrics
    """
    flat_importance = importance_matrix.flatten()
    
    # Gini coefficient
    sorted_imp = np.sort(flat_importance)
    n = len(sorted_imp)
    cumsum = np.cumsum(sorted_imp)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n
    
    # Top-k coverage
    sorted_desc = np.sort(flat_importance)[::-1]
    cumsum_desc = np.cumsum(sorted_desc)
    total = cumsum_desc[-1]
    
    if total > 0:
        top10_coverage = np.searchsorted(cumsum_desc, 0.5 * total) / len(flat_importance)
        top20_coverage = np.searchsorted(cumsum_desc, 0.8 * total) / len(flat_importance)
        top50_coverage = np.searchsorted(cumsum_desc, 0.9 * total) / len(flat_importance)
    else:
        top10_coverage = top20_coverage = top50_coverage = 0
    
    return {
        'gini_coefficient': gini,
        'top10_coverage': top10_coverage,
        'top20_coverage': top20_coverage,
        'top50_coverage': top50_coverage,
    }


def compute_statistical_metrics(importance_matrix: np.ndarray) -> Dict:
    """
    Compute statistical metrics for importance distribution
    
    Args:
        importance_matrix: Matrix of importance values
        
    Returns:
        Dictionary containing statistical metrics
    """
    flat_importance = importance_matrix.flatten()
    
    return {
        'mean_importance': np.mean(flat_importance),
        'std_importance': np.std(flat_importance),
        'max_importance': np.max(flat_importance),
    }


def compute_all_metrics(importance_matrix: np.ndarray) -> Dict:
    """
    Compute all metrics for a given importance matrix
    
    Args:
        importance_matrix: Matrix of importance values
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'total_params': importance_matrix.size,
    }
    
    # Update with all metric types
    metrics.update(compute_entropy_metrics(importance_matrix))
    metrics.update(compute_sparsity_metrics(importance_matrix))
    metrics.update(compute_concentration_metrics(importance_matrix))
    metrics.update(compute_statistical_metrics(importance_matrix))
    
    # Compute redundancy score
    metrics['redundancy_score'] = 1 - metrics['normalized_entropy']
    
    return metrics
