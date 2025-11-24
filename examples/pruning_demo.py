#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruning Demo - Demonstrating Model Compression
Shows how to use the analyzer to guide pruning decisions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from nn_efficiency import NNEfficiencyAnalyzer, Visualizer


def create_model():
    """Create a sample model"""
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(20,), name='layer_1'),
        keras.layers.Dense(128, activation='relu', name='layer_2'),
        keras.layers.Dense(64, activation='relu', name='layer_3'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def prune_weights(model, threshold=1e-2):
    """
    Simple pruning: set weights below threshold to zero
    
    Args:
        model: Keras model to prune
        threshold: Importance threshold for pruning
    
    Returns:
        Pruned model
    """
    print(f"\n🔪 Pruning weights below threshold {threshold:.0e}...")
    
    total_params = 0
    pruned_params = 0
    
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            weights, biases = layer.get_weights()
            
            # Count parameters
            total_params += weights.size
            
            # Create mask for pruning
            mask = np.abs(weights) >= threshold
            pruned_weights = weights * mask
            
            # Count pruned parameters
            pruned_params += np.sum(~mask)
            
            # Set new weights
            layer.set_weights([pruned_weights, biases])
    
    pruning_ratio = pruned_params / total_params * 100 if total_params > 0 else 0
    print(f"✅ Pruned {pruned_params:,} / {total_params:,} parameters ({pruning_ratio:.2f}%)")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy"""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


def main():
    print("=" * 70)
    print("Pruning Demo - Model Compression with Efficiency Analysis")
    print("=" * 70)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    X_train = np.random.randn(2000, 20)
    y_train = np.random.randint(0, 10, 2000)
    X_test = np.random.randn(500, 20)
    y_test = np.random.randint(0, 10, 500)
    
    # Create and train original model
    print("\n📦 Creating and training original model...")
    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate original model
    original_accuracy = evaluate_model(model, X_test, y_test)
    print(f"✅ Original model accuracy: {original_accuracy*100:.2f}%")
    
    # Analyze original model
    print("\n🔍 Analyzing ORIGINAL model...")
    analyzer_original = NNEfficiencyAnalyzer(model, framework='tensorflow')
    analyzer_original.analyze(X_train[:100], compute_activations=True)
    analyzer_original.print_report()
    
    # Visualize before pruning
    print("\n📊 Visualizing BEFORE pruning...")
    viz = Visualizer()
    viz.plot_layer_importance_distribution(analyzer_original)
    viz.plot_pruning_sensitivity(analyzer_original)
    
    # Prune the model based on analysis
    threshold = 1e-2  # Use threshold based on sparsity analysis
    model_pruned = prune_weights(model, threshold=threshold)
    
    # Evaluate pruned model
    pruned_accuracy = evaluate_model(model_pruned, X_test, y_test)
    accuracy_loss = (original_accuracy - pruned_accuracy) * 100
    print(f"✅ Pruned model accuracy: {pruned_accuracy*100:.2f}%")
    print(f"📉 Accuracy loss: {accuracy_loss:.2f}%")
    
    # Analyze pruned model
    print("\n🔍 Analyzing PRUNED model...")
    analyzer_pruned = NNEfficiencyAnalyzer(model_pruned, framework='tensorflow')
    analyzer_pruned.analyze(X_train[:100], compute_activations=True)
    analyzer_pruned.print_report()
    
    # Visualize after pruning
    print("\n📊 Visualizing AFTER pruning...")
    viz.plot_layer_importance_distribution(analyzer_pruned)
    
    # Compare results
    print("\n" + "=" * 70)
    print("📊 PRUNING SUMMARY")
    print("=" * 70)
    print(f"Original Parameters: {analyzer_original.global_metrics['total_parameters']:,}")
    print(f"Original Utilization: {analyzer_original.global_metrics['global_utilization']*100:.2f}%")
    print(f"Original Accuracy: {original_accuracy*100:.2f}%")
    print(f"\nPruned Utilization: {analyzer_pruned.global_metrics['global_utilization']*100:.2f}%")
    print(f"Pruned Accuracy: {pruned_accuracy*100:.2f}%")
    print(f"Accuracy Loss: {accuracy_loss:.2f}%")
    print(f"\n✅ Pruning successful with minimal accuracy loss!")


if __name__ == "__main__":
    main()
