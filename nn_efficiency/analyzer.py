# -*- coding: utf-8 -*-
"""
Neural Network Efficiency Analyzer - Core Analysis Class
"""

import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .metrics import compute_all_metrics, compute_flops_dense, compute_flops_conv2d

# Framework detection
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LayerAnalysis:
    """Container for layer-specific analysis results"""
    def __init__(self, name: str, layer_type: str):
        self.name = name
        self.layer_type = layer_type
        self.importance_matrix = None
        self.weights = None
        self.metrics = {}
        self.flops_info = None
        
    def compute_metrics(self):
        """Compute comprehensive metrics for this layer"""
        if self.importance_matrix is None:
            return
        
        self.metrics = compute_all_metrics(self.importance_matrix, self.flops_info)


class NNEfficiencyAnalyzer:
    """Main analyzer class for neural network efficiency analysis"""
    
    def __init__(self, model, framework: str = 'auto'):
        self.model = model
        self.framework = self._detect_framework(framework)
        self.layer_analyses: List[LayerAnalysis] = []
        self.global_metrics = {}
        self.neural_pathways = None
        
    def _detect_framework(self, framework: str) -> str:
        """Detect the deep learning framework"""
        if framework != 'auto':
            return framework
        
        if TF_AVAILABLE and isinstance(self.model, (tf.keras.Model, tf.keras.Sequential)):
            return 'tensorflow'
        elif TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            return 'pytorch'
        else:
            raise ValueError("Could not detect framework. Please specify 'tensorflow' or 'pytorch'")
    
    def analyze(self, sample_data: np.ndarray, compute_activations: bool = True, verbose: bool = True) -> Dict:
        """
        Perform comprehensive analysis of the neural network
        
        Args:
            sample_data: Representative input data for activation analysis
            compute_activations: Whether to compute activation-based importance
            verbose: Whether to print progress messages
        
        Returns:
            Dictionary containing analysis results
        """
        if verbose:
            print("🔍 Analyzing neural network efficiency...")
        
        if self.framework == 'tensorflow':
            self._analyze_tensorflow(sample_data, compute_activations)
        elif self.framework == 'pytorch':
            self._analyze_pytorch(sample_data, compute_activations)
        
        self._compute_global_metrics()
        
        if verbose:
            print("✅ Analysis complete!")
        return self.get_summary()
    
    def _analyze_tensorflow(self, sample_data: np.ndarray, compute_activations: bool):
        """Analyze TensorFlow/Keras model"""
        current_input = sample_data.copy()
        batch_size = sample_data.shape[0]
        
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                analysis = LayerAnalysis(layer.name, 'Dense')
                weights, biases = layer.get_weights()
                
                # Compute FLOPs for Dense layer
                input_size = weights.shape[0]
                output_size = weights.shape[1]
                analysis.flops_info = compute_flops_dense(
                    input_size=input_size,
                    output_size=output_size,
                    batch_size=batch_size,
                    has_bias=(biases is not None)
                )
                
                if compute_activations:
                    z = np.dot(current_input, weights) + biases
                    activations = layer.activation(z)
                    avg_activation = np.mean(np.abs(activations.numpy() if isinstance(activations, tf.Tensor) else activations), axis=0)
                    importance = np.abs(weights) * avg_activation
                    current_input = activations.numpy() if isinstance(activations, tf.Tensor) else activations
                else:
                    importance = np.abs(weights)
                
                analysis.weights = weights
                analysis.importance_matrix = importance
                analysis.compute_metrics()
                self.layer_analyses.append(analysis)
            
            elif isinstance(layer, tf.keras.layers.Conv2D):
                analysis = LayerAnalysis(layer.name, 'Conv2D')
                weights = layer.get_weights()[0]
                
                # Compute FLOPs for Conv2D layer
                kernel_h, kernel_w, in_channels, out_channels = weights.shape
                # Estimate input shape from current_input
                if len(current_input.shape) == 4:  # (batch, height, width, channels)
                    input_shape = (current_input.shape[1], current_input.shape[2], current_input.shape[3])
                else:
                    input_shape = (28, 28, in_channels)  # Default assumption
                
                config = layer.get_config()
                stride = config.get('strides', (1, 1))[0] if isinstance(config.get('strides', (1, 1)), tuple) else config.get('strides', 1)
                padding_val = 0 if config.get('padding', 'valid') == 'valid' else 1
                
                analysis.flops_info = compute_flops_conv2d(
                    input_shape=input_shape,
                    kernel_size=(kernel_h, kernel_w),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding_val,
                    batch_size=batch_size,
                    has_bias=(len(layer.get_weights()) > 1)
                )
                
                # For conv layers, flatten and use magnitude
                importance = np.abs(weights).reshape(-1)
                analysis.weights = weights
                analysis.importance_matrix = importance
                analysis.compute_metrics()
                self.layer_analyses.append(analysis)
    
    def _analyze_pytorch(self, sample_data: np.ndarray, compute_activations: bool):
        """Analyze PyTorch model"""
        self.model.eval()
        
        if isinstance(sample_data, np.ndarray):
            current_input = torch.tensor(sample_data, dtype=torch.float32)
            batch_size = sample_data.shape[0]
        else:
            current_input = sample_data.clone().detach().float()
            batch_size = sample_data.shape[0]
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                analysis = LayerAnalysis(name if name else 'linear', 'Linear')
                
                # Compute FLOPs for Linear layer
                input_size = module.in_features
                output_size = module.out_features
                analysis.flops_info = compute_flops_dense(
                    input_size=input_size,
                    output_size=output_size,
                    batch_size=batch_size,
                    has_bias=(module.bias is not None)
                )
                
                with torch.no_grad():
                    if compute_activations:
                        z = module(current_input)
                        activations = torch.relu(z)
                        avg_activation = torch.mean(torch.abs(activations), dim=0)
                        importance = torch.abs(module.weight) * avg_activation.unsqueeze(1)
                        current_input = activations
                    else:
                        importance = torch.abs(module.weight)
                
                analysis.weights = module.weight.detach().numpy()
                analysis.importance_matrix = importance.numpy().T
                analysis.compute_metrics()
                self.layer_analyses.append(analysis)
            
            elif isinstance(module, nn.Conv2d):
                analysis = LayerAnalysis(name if name else 'conv2d', 'Conv2d')
                
                # Compute FLOPs for Conv2d layer
                out_channels, in_channels, kernel_h, kernel_w = module.weight.shape
                
                # Estimate input shape from current_input
                if len(current_input.shape) == 4:  # (batch, channels, height, width)
                    input_shape = (current_input.shape[2], current_input.shape[3], current_input.shape[1])
                else:
                    input_shape = (28, 28, in_channels)  # Default assumption
                
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                
                analysis.flops_info = compute_flops_conv2d(
                    input_shape=input_shape,
                    kernel_size=(kernel_h, kernel_w),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    batch_size=batch_size,
                    has_bias=(module.bias is not None)
                )
                
                importance = torch.abs(module.weight).detach().numpy()
                analysis.weights = module.weight.detach().numpy()
                analysis.importance_matrix = importance.reshape(-1)
                analysis.compute_metrics()
                self.layer_analyses.append(analysis)
    
    def _compute_global_metrics(self):
        """Compute global model-level metrics"""
        total_params = sum(la.metrics['total_params'] for la in self.layer_analyses)
        effective_params = sum(la.metrics['effective_params'] for la in self.layer_analyses)
        
        all_importances = np.concatenate([la.importance_matrix.flatten() for la in self.layer_analyses])
        
        # Compute total FLOPs
        total_inference_flops = sum(
            la.metrics.get('inference_flops', 0) for la in self.layer_analyses
        )
        total_training_flops = sum(
            la.metrics.get('training_flops', 0) for la in self.layer_analyses
        )
        
        self.global_metrics = {
            'total_parameters': total_params,
            'effective_parameters': effective_params,
            'global_utilization': effective_params / total_params if total_params > 0 else 0,
            'num_layers': len(self.layer_analyses),
            'avg_layer_redundancy': np.mean([la.metrics['redundancy_score'] for la in self.layer_analyses]),
            'global_sparsity_1e-2': np.sum(all_importances < 1e-2) / len(all_importances),
            'compression_potential': 1 - (effective_params / total_params) if total_params > 0 else 0,
            'total_inference_flops': total_inference_flops,
            'total_training_flops': total_training_flops,
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive analysis summary"""
        return {
            'global_metrics': self.global_metrics,
            'layer_analyses': [
                {
                    'name': la.name,
                    'type': la.layer_type,
                    'metrics': la.metrics
                }
                for la in self.layer_analyses
            ]
        }
    
    def print_report(self):
        """Print detailed analysis report"""
        print("\n" + "="*70)
        print("📊 NEURAL NETWORK EFFICIENCY REPORT")
        print("="*70)
        
        print("\n🌐 GLOBAL METRICS:")
        print(f"  Total Parameters: {self.global_metrics['total_parameters']:,}")
        print(f"  Effective Parameters: {self.global_metrics['effective_parameters']:.2f}")
        print(f"  Global Utilization: {self.global_metrics['global_utilization']*100:.2f}%")
        print(f"  Compression Potential: {self.global_metrics['compression_potential']*100:.2f}%")
        print(f"  Average Layer Redundancy: {self.global_metrics['avg_layer_redundancy']*100:.2f}%")
        
        # Display FLOPs information
        if self.global_metrics.get('total_inference_flops', 0) > 0:
            print(f"\n💰 COMPUTATIONAL COST (FLOPs):")
            print(f"  Inference FLOPs: {self._format_flops(self.global_metrics['total_inference_flops'])}")
            print(f"  Training FLOPs: {self._format_flops(self.global_metrics['total_training_flops'])}")
        
        print("\n📋 LAYER-BY-LAYER ANALYSIS:")
        for la in self.layer_analyses:
            print(f"\n  🔸 {la.name} ({la.layer_type})")
            print(f"     Parameters: {la.metrics['total_params']:,}")
            print(f"     Utilization: {la.metrics['utilization_ratio']*100:.2f}%")
            print(f"     Redundancy: {la.metrics['redundancy_score']*100:.2f}%")
            print(f"     Sparsity (<1e-2): {la.metrics['sparsity_1e-2']*100:.2f}%")
            print(f"     Gini Coefficient: {la.metrics['gini_coefficient']:.3f}")
            if 'inference_flops' in la.metrics:
                print(f"     Inference FLOPs: {self._format_flops(la.metrics['inference_flops'])}")
                print(f"     Training FLOPs: {self._format_flops(la.metrics['training_flops'])}")
        
        print("\n" + "="*70)
        self._print_recommendations()
    
    def _format_flops(self, flops: float) -> str:
        """Format FLOPs in human-readable form"""
        if flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"
    
    def _print_recommendations(self):
        """Print optimization recommendations"""
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        # Find most redundant layers
        redundant_layers = sorted(
            [(la.name, la.metrics['redundancy_score']) for la in self.layer_analyses],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if redundant_layers[0][1] > 0.7:
            print(f"\n  ⚠️  High Redundancy Detected:")
            for name, score in redundant_layers:
                if score > 0.6:
                    print(f"     - Layer '{name}': {score*100:.1f}% redundancy")
                    print(f"       → Consider reducing neurons or applying pruning")
        
        # Check global utilization
        if self.global_metrics['global_utilization'] < 0.3:
            print(f"\n  ⚠️  Low Global Utilization ({self.global_metrics['global_utilization']*100:.1f}%)")
            print(f"     → Model may be oversized. Consider:")
            print(f"       - Reducing layer widths by 30-50%")
            print(f"       - Applying structured pruning")
        
        # Check sparsity
        if self.global_metrics['global_sparsity_1e-2'] > 0.5:
            print(f"\n  ✅ High Sparsity Detected ({self.global_metrics['global_sparsity_1e-2']*100:.1f}%)")
            print(f"     → Excellent candidate for weight pruning")
            print(f"     → Potential {self.global_metrics['compression_potential']*100:.1f}% compression")
        
        # FLOPs-based recommendations
        if self.global_metrics.get('total_inference_flops', 0) > 0:
            total_flops = self.global_metrics['total_inference_flops']
            
            # Find most expensive layers
            expensive_layers = sorted(
                [(la.name, la.metrics.get('inference_flops', 0)) for la in self.layer_analyses],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if expensive_layers[0][1] > 0:
                print(f"\n  💰 Computational Cost Analysis:")
                print(f"     Most expensive layers by FLOPs:")
                for name, flops in expensive_layers:
                    if flops > 0:
                        pct = (flops / total_flops) * 100
                        print(f"     - '{name}': {self._format_flops(flops)} ({pct:.1f}% of total)")
                
                # Estimate potential savings
                potential_savings = self.global_metrics['compression_potential'] * total_flops
                if potential_savings > 0:
                    print(f"\n     → Potential FLOPs reduction: {self._format_flops(potential_savings)}")
                    print(f"     → Expected speedup: {1/(1-self.global_metrics['compression_potential']):.2f}x")
    
    def compute_neural_pathways(self, top_k: int = 10) -> Dict:
        """
        Compute most important neural pathways through the network
        
        Args:
            top_k: Number of top pathways to identify per layer transition
            
        Returns:
            Dictionary containing pathway information
        """
        if len(self.layer_analyses) < 2:
            return {'pathways': [], 'message': 'Need at least 2 layers for pathway analysis'}
        
        pathways = []
        
        # For dense/linear layers, trace connections based on importance
        for i in range(len(self.layer_analyses) - 1):
            layer1 = self.layer_analyses[i]
            layer2 = self.layer_analyses[i + 1]
            
            # Only works for dense/linear layers with 2D importance matrices
            if layer1.layer_type in ['Dense', 'Linear'] and layer2.layer_type in ['Dense', 'Linear']:
                # Get importance matrices
                imp1 = layer1.importance_matrix
                imp2 = layer2.importance_matrix
                
                if len(imp1.shape) == 2 and len(imp2.shape) == 2:
                    # Calculate pathway importance as product of consecutive layer importances
                    # For each output of layer1 that connects to inputs of layer2
                    output_importance1 = np.sum(imp1, axis=0)  # Importance of each output neuron
                    input_importance2 = np.sum(imp2, axis=1)   # Importance of each input connection
                    
                    # Combine importances (element-wise for neurons that connect)
                    if len(output_importance1) == len(input_importance2):
                        pathway_importance = output_importance1 * input_importance2
                        
                        # Get top-k pathways
                        top_indices = np.argsort(pathway_importance)[-top_k:][::-1]
                        
                        pathways.append({
                            'from_layer': layer1.name,
                            'to_layer': layer2.name,
                            'pathway_indices': top_indices.tolist(),
                            'pathway_importances': pathway_importance[top_indices].tolist(),
                            'relative_importance': (pathway_importance[top_indices] / np.sum(pathway_importance)).tolist()
                        })
        
        self.neural_pathways = {
            'pathways': pathways,
            'num_pathway_segments': len(pathways)
        }
        
        return self.neural_pathways
