#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo: FLOPs Calculation and Neural Pathway Visualization

This example demonstrates the new features:
1. Automatic FLOPs calculation for model layers
2. Neural pathway visualization showing important connections
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_efficiency import NNEfficiencyAnalyzer, Visualizer

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, skipping TensorFlow example")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping PyTorch example")


def tensorflow_example():
    """Example using TensorFlow/Keras"""
    if not TF_AVAILABLE:
        return
    
    print("\n" + "="*70)
    print("TensorFlow Example: FLOPs and Neural Pathways")
    print("="*70)
    
    # Create a simple model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(20,), name='dense_1'),
        keras.layers.Dense(32, activation='relu', name='dense_2'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Generate sample data
    X_train = np.random.randn(100, 20)
    
    # Create analyzer
    analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
    
    # Perform analysis
    print("\n🔍 Analyzing model...")
    analyzer.analyze(X_train, compute_activations=True, verbose=False)
    
    # Print report with FLOPs
    analyzer.print_report()
    
    # Compute neural pathways
    print("\n🔍 Computing neural pathways...")
    pathways = analyzer.compute_neural_pathways(top_k=5)
    print(f"   Found {len(pathways['pathways'])} pathway segments")
    
    for i, segment in enumerate(pathways['pathways']):
        print(f"\n   Pathway {i+1}: {segment['from_layer']} → {segment['to_layer']}")
        print(f"   Top neuron: #{segment['pathway_indices'][0]} " +
              f"(importance: {segment['relative_importance'][0]*100:.1f}%)")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    viz = Visualizer()
    
    # Traditional visualizations
    viz.plot_layer_importance_distribution(analyzer)
    viz.plot_layer_comparison(analyzer)
    
    # New visualizations
    viz.plot_neural_pathways(analyzer)
    viz.plot_pathway_flow(analyzer)
    
    print("\n✅ TensorFlow example completed!")


def pytorch_example():
    """Example using PyTorch"""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*70)
    print("PyTorch Example: FLOPs and Neural Pathways")
    print("="*70)
    
    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNet()
    
    # Generate sample data
    X_train = np.random.randn(100, 20)
    
    # Create analyzer
    analyzer = NNEfficiencyAnalyzer(model, framework='pytorch')
    
    # Perform analysis
    print("\n🔍 Analyzing model...")
    analyzer.analyze(X_train, compute_activations=True, verbose=False)
    
    # Print report with FLOPs
    analyzer.print_report()
    
    # Compute neural pathways
    print("\n🔍 Computing neural pathways...")
    pathways = analyzer.compute_neural_pathways(top_k=5)
    print(f"   Found {len(pathways['pathways'])} pathway segments")
    
    for i, segment in enumerate(pathways['pathways']):
        print(f"\n   Pathway {i+1}: {segment['from_layer']} → {segment['to_layer']}")
        print(f"   Top neuron: #{segment['pathway_indices'][0]} " +
              f"(importance: {segment['relative_importance'][0]*100:.1f}%)")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    viz = Visualizer()
    
    # Traditional visualizations
    viz.plot_layer_importance_distribution(analyzer)
    viz.plot_layer_comparison(analyzer)
    
    # New visualizations
    viz.plot_neural_pathways(analyzer)
    viz.plot_pathway_flow(analyzer)
    
    print("\n✅ PyTorch example completed!")


def flops_comparison():
    """Compare FLOPs for different model sizes"""
    if not TF_AVAILABLE:
        return
    
    print("\n" + "="*70)
    print("FLOPs Comparison: Different Model Sizes")
    print("="*70)
    
    model_configs = [
        ("Small", [32, 16, 10]),
        ("Medium", [64, 32, 10]),
        ("Large", [128, 64, 10]),
    ]
    
    X_train = np.random.randn(100, 20)
    
    print("\n| Model  | Parameters | Inference FLOPs | Training FLOPs |")
    print("|--------|-----------|----------------|----------------|")
    
    for name, layers in model_configs:
        model = keras.Sequential([
            keras.layers.Dense(layers[0], activation='relu', input_shape=(20,)),
            keras.layers.Dense(layers[1], activation='relu'),
            keras.layers.Dense(layers[2], activation='softmax')
        ])
        
        analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
        analyzer.analyze(X_train, verbose=False)
        
        params = analyzer.global_metrics['total_parameters']
        inf_flops = analyzer.global_metrics['total_inference_flops']
        train_flops = analyzer.global_metrics['total_training_flops']
        
        print(f"| {name:6} | {params:9,} | {analyzer._format_flops(inf_flops):14} | {analyzer._format_flops(train_flops):14} |")
    
    print("\n✅ Comparison completed!")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║       Neural Network Efficiency Analyzer - New Features Demo    ║
    ║                                                                  ║
    ║  This demo showcases:                                           ║
    ║  💰 Automatic FLOPs calculation                                 ║
    ║  🔍 Neural pathway visualization                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    if TF_AVAILABLE:
        tensorflow_example()
        flops_comparison()
    
    if TORCH_AVAILABLE:
        pytorch_example()
    
    if not TF_AVAILABLE and not TORCH_AVAILABLE:
        print("\n⚠️  Neither TensorFlow nor PyTorch is available.")
        print("   Please install at least one framework to run the demo:")
        print("   pip install tensorflow")
        print("   pip install torch")
