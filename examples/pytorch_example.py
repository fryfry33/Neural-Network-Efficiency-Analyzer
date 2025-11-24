#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Example - Neural Network Efficiency Analysis
Demonstrates how to use the analyzer with PyTorch models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nn_efficiency import NNEfficiencyAnalyzer, Visualizer


class SimpleNet(nn.Module):
    """Simple PyTorch neural network for demonstration"""
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, X_train, y_train, epochs=5):
    """Train the PyTorch model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model


def main():
    print("=" * 70)
    print("PyTorch Model Efficiency Analysis Example")
    print("=" * 70)
    
    # Create sample data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 10, 1000)
    
    # Create and train model
    print("\n📦 Creating and training model...")
    model = SimpleNet()
    model = train_model(model, X_train, y_train, epochs=5)
    print("✅ Model trained!")
    
    # Analyze the model
    print("\n🔍 Analyzing model efficiency...")
    analyzer = NNEfficiencyAnalyzer(model, framework='pytorch')
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
