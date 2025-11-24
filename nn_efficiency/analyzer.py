# -*- coding: utf-8 -*-
"""
Neural Network Efficiency Analyzer - Core Analysis Class
"""

import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .metrics import compute_all_metrics

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
        
    def compute_metrics(self):
        """Compute comprehensive metrics for this layer"""
        if self.importance_matrix is None:
            return
        
        self.metrics = compute_all_metrics(self.importance_matrix)


class NNEfficiencyAnalyzer:
    """Main analyzer class for neural network efficiency analysis"""
    
    def __init__(self, model, framework: str = 'auto'):
        self.model = model
        self.framework = self._detect_framework(framework)
        self.layer_analyses: List[LayerAnalysis] = []
        self.global_metrics = {}
        
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
    
    def analyze(self, sample_data: np.ndarray, compute_activations: bool = True) -> Dict:
        """
        Perform comprehensive analysis of the neural network
        
        Args:
            sample_data: Representative input data for activation analysis
            compute_activations: Whether to compute activation-based importance
        
        Returns:
            Dictionary containing analysis results
        """
        print("🔍 Analyzing neural network efficiency...")
        
        if self.framework == 'tensorflow':
            self._analyze_tensorflow(sample_data, compute_activations)
        elif self.framework == 'pytorch':
            self._analyze_pytorch(sample_data, compute_activations)
        
        self._compute_global_metrics()
        
        print("✅ Analysis complete!")
        return self.get_summary()
    
    def _analyze_tensorflow(self, sample_data: np.ndarray, compute_activations: bool):
        """Analyze TensorFlow/Keras model"""
        current_input = sample_data.copy()
        
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                analysis = LayerAnalysis(layer.name, 'Dense')
                weights, biases = layer.get_weights()
                
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
        else:
            current_input = sample_data.clone().detach().float()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                analysis = LayerAnalysis(name if name else 'linear', 'Linear')
                
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
        
        self.global_metrics = {
            'total_parameters': total_params,
            'effective_parameters': effective_params,
            'global_utilization': effective_params / total_params if total_params > 0 else 0,
            'num_layers': len(self.layer_analyses),
            'avg_layer_redundancy': np.mean([la.metrics['redundancy_score'] for la in self.layer_analyses]),
            'global_sparsity_1e-2': np.sum(all_importances < 1e-2) / len(all_importances),
            'compression_potential': 1 - (effective_params / total_params) if total_params > 0 else 0,
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
        
        print("\n📋 LAYER-BY-LAYER ANALYSIS:")
        for la in self.layer_analyses:
            print(f"\n  🔸 {la.name} ({la.layer_type})")
            print(f"     Parameters: {la.metrics['total_params']:,}")
            print(f"     Utilization: {la.metrics['utilization_ratio']*100:.2f}%")
            print(f"     Redundancy: {la.metrics['redundancy_score']*100:.2f}%")
            print(f"     Sparsity (<1e-2): {la.metrics['sparsity_1e-2']*100:.2f}%")
            print(f"     Gini Coefficient: {la.metrics['gini_coefficient']:.3f}")
        
        print("\n" + "="*70)
        self._print_recommendations()
    
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
