# -*- coding: utf-8 -*-
"""
Metrics calculations for neural network efficiency analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional


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


def compute_flops_dense(input_size: int, output_size: int, batch_size: int = 1, 
                        has_bias: bool = True) -> Dict:
    """
    Compute FLOPs for a Dense/Linear layer
    
    Args:
        input_size: Number of input features
        output_size: Number of output neurons
        batch_size: Batch size (default: 1 for single sample inference)
        has_bias: Whether the layer has bias terms
        
    Returns:
        Dictionary containing FLOPs for inference and training
        
    Note:
        Training FLOPs are approximated using a simplified model:
        - Forward pass = inference FLOPs
        - Backward pass ≈ 2× forward pass (gradient w.r.t. input and weights)
        - Weight update is included but optimizer-specific operations (e.g., Adam momentum)
          are not accounted for. Actual training costs may be higher with complex optimizers.
    """
    # Inference FLOPs: matrix multiplication + bias addition
    # For each output neuron: input_size multiplications + input_size-1 additions
    # Plus bias addition if present
    inference_multiply = batch_size * output_size * input_size
    inference_add = batch_size * output_size * (input_size - 1)
    if has_bias:
        inference_add += batch_size * output_size
    inference_flops = inference_multiply + inference_add
    
    # Training FLOPs (approximate): forward pass + backward pass + weight update
    # Forward: same as inference
    # Backward: gradient computation (similar to forward pass)
    # Weight update: parameter updates
    forward_flops = inference_flops
    backward_flops = 2 * forward_flops  # Approximate: gradient w.r.t input and weights
    update_flops = batch_size * (input_size * output_size + (output_size if has_bias else 0))
    training_flops = forward_flops + backward_flops + update_flops
    
    return {
        'inference_flops': inference_flops,
        'training_flops': training_flops,
        'inference_flops_per_sample': inference_flops / batch_size,
        'training_flops_per_sample': training_flops / batch_size,
    }


def compute_flops_conv2d(input_shape: Tuple[int, int, int], kernel_size: Tuple[int, int],
                          in_channels: int, out_channels: int, stride: int = 1,
                          padding: int = 0, batch_size: int = 1, has_bias: bool = True) -> Dict:
    """
    Compute FLOPs for a Conv2D layer
    
    Args:
        input_shape: Input spatial dimensions (height, width, channels)
        kernel_size: Kernel dimensions (height, width)
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride value
        padding: Padding value
        batch_size: Batch size
        has_bias: Whether the layer has bias terms
        
    Returns:
        Dictionary containing FLOPs for inference and training
        
    Note:
        Training FLOPs are approximated using a 3× multiplier for backward pass
        complexity relative to forward pass. The actual computational cost depends on
        implementation details and whether gradients are computed for weights, inputs,
        or both. This is a reasonable approximation for most scenarios.
    """
    input_h, input_w = input_shape[0], input_shape[1]
    kernel_h, kernel_w = kernel_size
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding - kernel_h) // stride + 1
    output_w = (input_w + 2 * padding - kernel_w) // stride + 1
    
    # Inference FLOPs
    # For each output position: kernel_h * kernel_w * in_channels multiplications + additions
    ops_per_output = kernel_h * kernel_w * in_channels
    num_outputs = batch_size * output_h * output_w * out_channels
    
    inference_multiply = num_outputs * ops_per_output
    inference_add = num_outputs * (ops_per_output - 1)
    if has_bias:
        inference_add += num_outputs
    inference_flops = inference_multiply + inference_add
    
    # Training FLOPs (approximate)
    forward_flops = inference_flops
    backward_flops = 3 * forward_flops  # Approximate: more complex due to convolution
    num_params = kernel_h * kernel_w * in_channels * out_channels + (out_channels if has_bias else 0)
    update_flops = batch_size * num_params
    training_flops = forward_flops + backward_flops + update_flops
    
    return {
        'inference_flops': inference_flops,
        'training_flops': training_flops,
        'inference_flops_per_sample': inference_flops / batch_size,
        'training_flops_per_sample': training_flops / batch_size,
        'output_shape': (output_h, output_w, out_channels),
    }


def compute_all_metrics(importance_matrix: np.ndarray, flops_info: Optional[Dict] = None) -> Dict:
    """
    Compute all metrics for a given importance matrix
    
    Args:
        importance_matrix: Matrix of importance values
        flops_info: Optional dictionary containing FLOPs information
        
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
    
    # Add FLOPs information if provided
    if flops_info:
        metrics.update(flops_info)
    
    return metrics
