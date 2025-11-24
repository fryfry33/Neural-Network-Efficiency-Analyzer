#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow Example - Neural Network Efficiency Analysis
Demonstrates how to use the analyzer with TensorFlow/Keras models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from nn_efficiency import NNEfficiencyAnalyzer, Visualizer


def create_sample_model():
    """Create a simple TensorFlow model for demonstration"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,), name='hidden_1'),
        keras.layers.Dense(64, activation='relu', name='hidden_2'),
        keras.layers.Dense(32, activation='relu', name='hidden_3'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def main():
    print("=" * 70)
    print("TensorFlow/Keras Model Efficiency Analysis Example")
    print("=" * 70)
    
    # Create sample data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 10, 1000)
    
    # Create and train model
    print("\n📦 Creating and training model...")
    model = create_sample_model()
    
    # Quick training to initialize weights meaningfully
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    print("✅ Model trained!")
    
    # Analyze the model
    print("\n🔍 Analyzing model efficiency...")
    analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
    results = analyzer.analyze(X_train[:100], compute_activations=True)
    
    # Print detailed report
    analyzer.print_report()
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    viz = Visualizer()
    
    print("  - Layer importance distributions")
    viz.plot_layer_importance_distribution(analyzer)
    
    print("  - Layer comparison")
    viz.plot_layer_comparison(analyzer)
    
    print("  - Pruning sensitivity analysis")
    viz.plot_pruning_sensitivity(analyzer)
    
    print("  - Efficiency radar for first layer")
    viz.plot_efficiency_radar(analyzer, layer_idx=0)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
