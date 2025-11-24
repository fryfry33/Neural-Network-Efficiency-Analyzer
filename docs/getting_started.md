# 🚀 Getting Started with Neural Network Efficiency Analyzer

## Installation

### From Source

```bash
git clone https://github.com/fryfry33/-Neural-Network-Efficiency-Analyzer.git
cd -Neural-Network-Efficiency-Analyzer
pip install -r requirements.txt
pip install -e .
```

### Using pip (when published)

```bash
pip install nn-efficiency-analyzer
```

## Quick Start

### 1. Basic Usage (3 Lines of Code)

```python
from nn_efficiency import quick_analyze
import numpy as np

# Your model and data
model = ...  # TensorFlow or PyTorch model
X_train = np.random.randn(1000, 20)  # Sample data

# Analyze and visualize
analyzer = quick_analyze(model, X_train)
```

This will:
- Automatically detect your framework (TensorFlow or PyTorch)
- Analyze weight importance and utilization
- Print a detailed efficiency report
- Generate 4 visualization plots

### 2. TensorFlow/Keras Example

```python
import tensorflow as tf
from tensorflow import keras
from nn_efficiency import NNEfficiencyAnalyzer

# Create your model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Train your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)

# Analyze efficiency
analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
results = analyzer.analyze(X_train[:100], compute_activations=True)
analyzer.print_report()
```

### 3. PyTorch Example

```python
import torch
import torch.nn as nn
from nn_efficiency import NNEfficiencyAnalyzer

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MyModel()

# Train your model...

# Analyze efficiency
analyzer = NNEfficiencyAnalyzer(model, framework='pytorch')
results = analyzer.analyze(X_train[:100], compute_activations=True)
analyzer.print_report()
```

## Understanding the Output

### Global Metrics

- **Total Parameters**: Total number of weights in your model
- **Effective Parameters**: Number of "useful" parameters (based on entropy)
- **Global Utilization**: Percentage of parameters being effectively used
- **Compression Potential**: How much the model could be compressed
- **Average Layer Redundancy**: Average redundancy across all layers

### Layer Metrics

For each layer, you'll see:
- **Parameters**: Number of weights in the layer
- **Utilization**: How efficiently the layer uses its parameters
- **Redundancy**: Percentage of redundant parameters
- **Sparsity**: Percentage of low-importance weights
- **Gini Coefficient**: Inequality in weight importance distribution

### Optimization Recommendations

The analyzer automatically provides recommendations:
- 🔴 **High redundancy detected**: Consider pruning or reducing layer size
- 🟡 **Low utilization**: Model may be oversized
- 🟢 **High sparsity**: Good candidate for weight pruning

## Visualization Guide

### 1. Layer Importance Distribution
Shows histogram of weight importance for each layer. Helps identify:
- Layers with many low-importance weights (good pruning candidates)
- Layers with uniform importance (well-utilized)

### 2. Layer Comparison
Compares key metrics across layers:
- Utilization ratio
- Redundancy score
- Sparsity percentage

### 3. Efficiency Radar
Shows multiple metrics for a single layer in a radar chart:
- Utilization
- Sparsity
- Gini coefficient (inequality)
- Coverage metrics
- Redundancy

### 4. Pruning Sensitivity
Shows how many weights would be pruned at different thresholds:
- Left plot: Per-layer pruning ratios
- Right plot: Cumulative importance curve

## Common Use Cases

### Case 1: Check if Model is Oversized

```python
analyzer = NNEfficiencyAnalyzer(model)
analyzer.analyze(X_train[:100])

if analyzer.global_metrics['global_utilization'] < 0.3:
    print("⚠️ Model may be oversized!")
    print(f"Compression potential: {analyzer.global_metrics['compression_potential']*100:.1f}%")
```

### Case 2: Find Layers to Prune

```python
analyzer.analyze(X_train[:100])

for layer in analyzer.layer_analyses:
    if layer.metrics['sparsity_1e-2'] > 0.5:
        print(f"Layer {layer.name} is a good pruning candidate")
        print(f"  - {layer.metrics['sparsity_1e-2']*100:.1f}% of weights are sparse")
```

### Case 3: Compare Different Architectures

```python
models = [model_small, model_medium, model_large]
analyzers = []

for model in models:
    analyzer = NNEfficiencyAnalyzer(model)
    analyzer.analyze(X_train[:100])
    analyzers.append(analyzer)

# Compare global utilization
for i, analyzer in enumerate(analyzers):
    util = analyzer.global_metrics['global_utilization']
    print(f"Model {i+1} utilization: {util*100:.2f}%")
```

### Case 4: Monitor Training Progress

```python
utilizations = []

for epoch in range(num_epochs):
    train_model(model, epoch)
    
    if epoch % 5 == 0:
        analyzer = NNEfficiencyAnalyzer(model)
        analyzer.analyze(X_train[:100])
        util = analyzer.global_metrics['global_utilization']
        utilizations.append(util)
        print(f"Epoch {epoch}: Utilization = {util*100:.2f}%")

# Plot utilization over time
import matplotlib.pyplot as plt
plt.plot(range(0, num_epochs, 5), utilizations)
plt.xlabel('Epoch')
plt.ylabel('Utilization')
plt.title('Model Utilization During Training')
plt.show()
```

## Tips and Best Practices

1. **Use Representative Data**: Pass a representative sample of your training data (100-1000 samples)
2. **Enable Activations**: Set `compute_activations=True` for more accurate importance estimates
3. **Analyze After Training**: Run analysis on trained models for meaningful results
4. **Compare Thresholds**: Use pruning sensitivity plot to choose optimal pruning threshold
5. **Iterative Pruning**: Prune, fine-tune, then analyze again

## Next Steps

- Read the [API Reference](api_reference.md) for detailed documentation
- Learn about the [Methodology](methodology.md) behind the metrics
- Try the examples in the `examples/` directory
- Experiment with the Jupyter notebook: `examples/demo_notebook.ipynb`

## Troubleshooting

### Framework Not Detected
```python
# Explicitly specify framework
analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')  # or 'pytorch'
```

### Memory Issues with Large Models
```python
# Use smaller sample size
analyzer.analyze(X_train[:50], compute_activations=False)
```

### Visualizations Not Showing
```python
import matplotlib.pyplot as plt
# Add this after creating visualizations
plt.show()
```
