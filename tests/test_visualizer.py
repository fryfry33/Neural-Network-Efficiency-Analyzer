# -*- coding: utf-8 -*-
"""
Unit tests for visualizer module
"""

import unittest
import numpy as np
from nn_efficiency.visualizer import Visualizer
from nn_efficiency.analyzer import NNEfficiencyAnalyzer, LayerAnalysis

# Try to import frameworks for testing
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Mock matplotlib to avoid display issues in tests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


@unittest.skipIf(not TF_AVAILABLE, "TensorFlow not available")
class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple model
        self.model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(5, activation='softmax')
        ])
        
        self.sample_data = np.random.randn(50, 10)
        
        # Create and run analyzer
        self.analyzer = NNEfficiencyAnalyzer(self.model, framework='tensorflow')
        self.analyzer.analyze(self.sample_data, compute_activations=False)
        
        self.viz = Visualizer()
    
    def tearDown(self):
        """Clean up after tests"""
        plt.close('all')
    
    def test_plot_layer_importance_distribution(self):
        """Test layer importance distribution plotting"""
        try:
            self.viz.plot_layer_importance_distribution(self.analyzer, figsize=(12, 8))
            # If no exception raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_layer_importance_distribution raised exception: {e}")
    
    def test_plot_layer_comparison(self):
        """Test layer comparison plotting"""
        try:
            self.viz.plot_layer_comparison(self.analyzer, figsize=(12, 6))
            # If no exception raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_layer_comparison raised exception: {e}")
    
    def test_plot_efficiency_radar(self):
        """Test efficiency radar plotting"""
        try:
            self.viz.plot_efficiency_radar(self.analyzer, layer_idx=0, figsize=(8, 8))
            # If no exception raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_efficiency_radar raised exception: {e}")
    
    def test_plot_pruning_sensitivity(self):
        """Test pruning sensitivity plotting"""
        try:
            self.viz.plot_pruning_sensitivity(self.analyzer, figsize=(12, 5))
            # If no exception raised, test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_pruning_sensitivity raised exception: {e}")
    
    def test_visualizer_with_single_layer(self):
        """Test visualizer with a single layer model"""
        # Create a single layer model
        single_layer_model = keras.Sequential([
            keras.layers.Dense(10, activation='softmax', input_shape=(5,))
        ])
        
        analyzer = NNEfficiencyAnalyzer(single_layer_model, framework='tensorflow')
        analyzer.analyze(np.random.randn(20, 5), compute_activations=False)
        
        try:
            self.viz.plot_layer_importance_distribution(analyzer)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Visualizer failed with single layer: {e}")


class TestVisualizerMock(unittest.TestCase):
    """Test visualizer with mock analyzer (no framework required)"""
    
    def setUp(self):
        """Create mock analyzer with fake data"""
        # Create mock layer analyses
        self.mock_analyzer = type('MockAnalyzer', (), {})()
        self.mock_analyzer.layer_analyses = []
        
        for i in range(3):
            layer = LayerAnalysis(f"layer_{i}", "Dense")
            layer.importance_matrix = np.random.rand(10, 20)
            layer.compute_metrics()
            self.mock_analyzer.layer_analyses.append(layer)
        
        self.viz = Visualizer()
    
    def tearDown(self):
        """Clean up after tests"""
        plt.close('all')
    
    def test_visualizer_instantiation(self):
        """Test that Visualizer can be instantiated"""
        viz = Visualizer()
        self.assertIsInstance(viz, Visualizer)
    
    def test_visualizer_static_methods_exist(self):
        """Test that all visualization methods exist"""
        self.assertTrue(hasattr(Visualizer, 'plot_layer_importance_distribution'))
        self.assertTrue(hasattr(Visualizer, 'plot_layer_comparison'))
        self.assertTrue(hasattr(Visualizer, 'plot_efficiency_radar'))
        self.assertTrue(hasattr(Visualizer, 'plot_pruning_sensitivity'))


if __name__ == '__main__':
    unittest.main()
