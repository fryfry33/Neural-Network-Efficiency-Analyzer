# 📚 API Reference

## Core Classes

### `NNEfficiencyAnalyzer`

Main class for analyzing neural network efficiency.

```python
from nn_efficiency import NNEfficiencyAnalyzer

analyzer = NNEfficiencyAnalyzer(model, framework='auto')
```

#### Constructor Parameters

- **model**: Neural network model (TensorFlow `keras.Model` or PyTorch `nn.Module`)
- **framework** (str, optional): Framework to use. Options:
  - `'auto'` (default): Automatically detect framework
  - `'tensorflow'`: Use TensorFlow/Keras
  - `'pytorch'`: Use PyTorch

#### Methods

##### `analyze(sample_data, compute_activations=True)`

Perform comprehensive efficiency analysis of the neural network.

**Parameters:**
- **sample_data** (np.ndarray): Representative input data for analysis (typically 100-1000 samples)
- **compute_activations** (bool, optional): Whether to compute activation-based importance. Default: `True`

**Returns:**
- dict: Dictionary containing analysis results with keys:
  - `'global_metrics'`: Model-level metrics
  - `'layer_analyses'`: List of per-layer analysis results

**Example:**
```python
results = analyzer.analyze(X_train[:100], compute_activations=True)
```

##### `print_report()`

Print a detailed, human-readable analysis report to console.

**Example:**
```python
analyzer.print_report()
```

##### `get_summary()`

Get comprehensive analysis summary as a dictionary.

**Returns:**
- dict: Complete summary with global and per-layer metrics

**Example:**
```python
summary = analyzer.get_summary()
print(summary['global_metrics']['global_utilization'])
```

#### Attributes

##### `layer_analyses`

List of `LayerAnalysis` objects, one for each analyzed layer.

**Example:**
```python
for layer in analyzer.layer_analyses:
    print(f"{layer.name}: {layer.metrics['utilization_ratio']:.2%}")
```

##### `global_metrics`

Dictionary of model-level metrics:

- **total_parameters** (int): Total number of parameters
- **effective_parameters** (float): Number of effective parameters (based on entropy)
- **global_utilization** (float): Overall utilization ratio (0-1)
- **num_layers** (int): Number of analyzed layers
- **avg_layer_redundancy** (float): Average redundancy across layers (0-1)
- **global_sparsity_1e-2** (float): Fraction of weights below 1e-2 threshold (0-1)
- **compression_potential** (float): Potential compression ratio (0-1)

**Example:**
```python
print(f"Utilization: {analyzer.global_metrics['global_utilization']:.2%}")
print(f"Compression potential: {analyzer.global_metrics['compression_potential']:.2%}")
```

##### `framework`

String indicating the detected or specified framework ('tensorflow' or 'pytorch').

---

### `LayerAnalysis`

Container for layer-specific analysis results. Usually accessed via `analyzer.layer_analyses`.

#### Attributes

- **name** (str): Layer name
- **layer_type** (str): Layer type (e.g., 'Dense', 'Linear', 'Conv2D')
- **importance_matrix** (np.ndarray): Matrix of importance values for each weight
- **weights** (np.ndarray): Original weight values
- **metrics** (dict): Dictionary of computed metrics

#### Layer Metrics Dictionary

- **total_params** (int): Number of parameters in the layer
- **entropy** (float): Shannon entropy of importance distribution
- **normalized_entropy** (float): Entropy normalized by maximum possible (0-1)
- **effective_params** (float): e^entropy - effective number of parameters
- **utilization_ratio** (float): Ratio of effective to total parameters (0-1)
- **sparsity_1e-2** (float): Fraction of weights with importance < 0.01
- **sparsity_1e-3** (float): Fraction of weights with importance < 0.001
- **gini_coefficient** (float): Gini coefficient of importance distribution (0-1)
- **top10_coverage** (float): Fraction of weights needed to reach 50% importance
- **top20_coverage** (float): Fraction of weights needed to reach 80% importance
- **top50_coverage** (float): Fraction of weights needed to reach 90% importance
- **mean_importance** (float): Mean importance value
- **std_importance** (float): Standard deviation of importance
- **max_importance** (float): Maximum importance value
- **redundancy_score** (float): 1 - normalized_entropy (0-1, higher = more redundant)

**Example:**
```python
layer = analyzer.layer_analyses[0]
print(f"Layer: {layer.name}")
print(f"Parameters: {layer.metrics['total_params']}")
print(f"Utilization: {layer.metrics['utilization_ratio']:.2%}")
print(f"Redundancy: {layer.metrics['redundancy_score']:.2%}")
```

---

### `Visualizer`

Visualization utilities for network analysis results.

```python
from nn_efficiency import Visualizer

viz = Visualizer()
```

#### Methods

All methods are static and can be called directly on the class or an instance.

##### `plot_layer_importance_distribution(analyzer, figsize=(15, 10))`

Plot histograms of weight importance for each layer.

**Parameters:**
- **analyzer** (NNEfficiencyAnalyzer): Analyzer instance with results
- **figsize** (tuple, optional): Figure size (width, height)

**Example:**
```python
viz.plot_layer_importance_distribution(analyzer, figsize=(15, 10))
```

##### `plot_layer_comparison(analyzer, figsize=(12, 6))`

Create bar charts comparing metrics across all layers.

**Parameters:**
- **analyzer** (NNEfficiencyAnalyzer): Analyzer instance with results
- **figsize** (tuple, optional): Figure size (width, height)

**Example:**
```python
viz.plot_layer_comparison(analyzer)
```

##### `plot_efficiency_radar(analyzer, layer_idx=0, figsize=(8, 8))`

Create a radar chart showing multiple efficiency metrics for a single layer.

**Parameters:**
- **analyzer** (NNEfficiencyAnalyzer): Analyzer instance with results
- **layer_idx** (int, optional): Index of layer to visualize. Default: 0
- **figsize** (tuple, optional): Figure size (width, height)

**Example:**
```python
viz.plot_efficiency_radar(analyzer, layer_idx=1)
```

##### `plot_pruning_sensitivity(analyzer, figsize=(12, 5))`

Visualize the impact of pruning at different importance thresholds.

**Parameters:**
- **analyzer** (NNEfficiencyAnalyzer): Analyzer instance with results
- **figsize** (tuple, optional): Figure size (width, height)

**Example:**
```python
viz.plot_pruning_sensitivity(analyzer)
```

---

## Utility Functions

### `quick_analyze(model, sample_data, framework='auto', visualize=True)`

Convenience function for quick model analysis with one function call.

**Parameters:**
- **model**: Neural network model (TensorFlow or PyTorch)
- **sample_data** (np.ndarray): Representative input data
- **framework** (str, optional): Framework to use ('auto', 'tensorflow', or 'pytorch')
- **visualize** (bool, optional): Whether to generate visualizations. Default: `True`

**Returns:**
- NNEfficiencyAnalyzer: Analyzer instance with completed analysis

**Example:**
```python
from nn_efficiency import quick_analyze

analyzer = quick_analyze(model, X_train[:100], visualize=True)
```

---

## Metrics Module

The `metrics` module contains functions for computing various efficiency metrics. These are used internally by `LayerAnalysis` but can also be used independently.

### `compute_all_metrics(importance_matrix)`

Compute all metrics for a given importance matrix.

**Parameters:**
- **importance_matrix** (np.ndarray): Matrix of importance values

**Returns:**
- dict: Dictionary containing all computed metrics

**Example:**
```python
from nn_efficiency.metrics import compute_all_metrics
import numpy as np

importance = np.random.rand(10, 20)
metrics = compute_all_metrics(importance)
print(metrics)
```

### Individual Metric Functions

- `compute_entropy_metrics(importance_matrix)`: Entropy-based metrics
- `compute_sparsity_metrics(importance_matrix)`: Sparsity metrics
- `compute_concentration_metrics(importance_matrix)`: Concentration metrics (Gini, top-k)
- `compute_statistical_metrics(importance_matrix)`: Statistical metrics (mean, std, max)

---

## Type Annotations

The library uses type hints for better IDE support:

```python
from typing import Dict, List, Optional
import numpy as np

def analyze(
    self,
    sample_data: np.ndarray,
    compute_activations: bool = True
) -> Dict:
    ...
```

---

## Error Handling

### Framework Detection Errors

If the framework cannot be automatically detected:
```python
ValueError: Could not detect framework. Please specify 'tensorflow' or 'pytorch'
```

**Solution:** Explicitly specify the framework:
```python
analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
```

---

## Complete Example

```python
import numpy as np
from tensorflow import keras
from nn_efficiency import NNEfficiencyAnalyzer, Visualizer

# Create model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Generate data and train
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 10, 1000)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, verbose=0)

# Analyze
analyzer = NNEfficiencyAnalyzer(model, framework='tensorflow')
results = analyzer.analyze(X_train[:100], compute_activations=True)

# Print report
analyzer.print_report()

# Access metrics
print(f"\nGlobal utilization: {analyzer.global_metrics['global_utilization']:.2%}")

for layer in analyzer.layer_analyses:
    print(f"{layer.name}: {layer.metrics['utilization_ratio']:.2%} utilization")

# Visualize
viz = Visualizer()
viz.plot_layer_importance_distribution(analyzer)
viz.plot_layer_comparison(analyzer)
viz.plot_pruning_sensitivity(analyzer)
```
