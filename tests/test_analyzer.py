# -*- coding: utf-8 -*-
"""
Unit tests for analyzer module
"""

import unittest
import numpy as np
from nn_efficiency.analyzer import LayerAnalysis, NNEfficiencyAnalyzer

# Try to import frameworks for testing
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestLayerAnalysis(unittest.TestCase):
    """Test cases for LayerAnalysis class"""
    
    def test_layer_analysis_initialization(self):
        """Test LayerAnalysis initialization"""
        layer = LayerAnalysis("test_layer", "Dense")
        
        self.assertEqual(layer.name, "test_layer")
        self.assertEqual(layer.layer_type, "Dense")
        self.assertIsNone(layer.importance_matrix)
        self.assertIsNone(layer.weights)
        self.assertEqual(layer.metrics, {})
    
    def test_compute_metrics(self):
        """Test metric computation for a layer"""
        layer = LayerAnalysis("test_layer", "Dense")
        layer.importance_matrix = np.random.rand(10, 20)
        
        layer.compute_metrics()
        
        # Check that metrics were computed
        self.assertIsInstance(layer.metrics, dict)
        self.assertGreater(len(layer.metrics), 0)
        self.assertIn('total_params', layer.metrics)
        self.assertIn('entropy', layer.metrics)
        self.assertIn('utilization_ratio', layer.metrics)


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
class TestNNEfficiencyAnalyzerTensorFlow(unittest.TestCase):
    """Test cases for NNEfficiencyAnalyzer with TensorFlow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple model
        self.model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(5, activation='softmax')
        ])
        
        self.sample_data = np.random.randn(50, 10)
    
    def test_framework_detection(self):
        """Test automatic framework detection"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='auto')
        self.assertEqual(analyzer.framework, 'tensorflow')
    
    def test_explicit_framework(self):
        """Test explicit framework specification"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='tensorflow')
        self.assertEqual(analyzer.framework, 'tensorflow')
    
    def test_analysis(self):
        """Test complete analysis workflow"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='tensorflow')
        results = analyzer.analyze(self.sample_data, compute_activations=True)
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('global_metrics', results)
        self.assertIn('layer_analyses', results)
        
        # Check layer analyses
        self.assertEqual(len(analyzer.layer_analyses), 3)  # 3 Dense layers
        
        # Check global metrics
        self.assertIn('total_parameters', analyzer.global_metrics)
        self.assertIn('effective_parameters', analyzer.global_metrics)
        self.assertIn('global_utilization', analyzer.global_metrics)
    
    def test_analysis_without_activations(self):
        """Test analysis without computing activations"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='tensorflow')
        results = analyzer.analyze(self.sample_data, compute_activations=False)
        
        # Should still work
        self.assertIsInstance(results, dict)
        self.assertEqual(len(analyzer.layer_analyses), 3)
    
    def test_get_summary(self):
        """Test summary generation"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='tensorflow')
        analyzer.analyze(self.sample_data)
        
        summary = analyzer.get_summary()
        
        self.assertIn('global_metrics', summary)
        self.assertIn('layer_analyses', summary)
        
        # Check layer analysis structure
        for layer_analysis in summary['layer_analyses']:
            self.assertIn('name', layer_analysis)
            self.assertIn('type', layer_analysis)
            self.assertIn('metrics', layer_analysis)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestNNEfficiencyAnalyzerPyTorch(unittest.TestCase):
    """Test cases for NNEfficiencyAnalyzer with PyTorch"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 5)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        self.model = SimpleModel()
        self.sample_data = np.random.randn(50, 10)
    
    def test_framework_detection(self):
        """Test automatic framework detection"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='auto')
        self.assertEqual(analyzer.framework, 'pytorch')
    
    def test_explicit_framework(self):
        """Test explicit framework specification"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='pytorch')
        self.assertEqual(analyzer.framework, 'pytorch')
    
    def test_analysis(self):
        """Test complete analysis workflow"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='pytorch')
        results = analyzer.analyze(self.sample_data, compute_activations=True)
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('global_metrics', results)
        self.assertIn('layer_analyses', results)
        
        # Check layer analyses
        self.assertEqual(len(analyzer.layer_analyses), 3)  # 3 Linear layers
        
        # Check global metrics
        self.assertIn('total_parameters', analyzer.global_metrics)
        self.assertIn('effective_parameters', analyzer.global_metrics)
        self.assertIn('global_utilization', analyzer.global_metrics)
    
    def test_analysis_with_torch_tensor(self):
        """Test analysis with PyTorch tensor input"""
        analyzer = NNEfficiencyAnalyzer(self.model, framework='pytorch')
        tensor_data = torch.FloatTensor(self.sample_data)
        results = analyzer.analyze(tensor_data, compute_activations=True)
        
        # Should work with tensor input
        self.assertIsInstance(results, dict)
        self.assertEqual(len(analyzer.layer_analyses), 3)


if __name__ == '__main__':
    unittest.main()
