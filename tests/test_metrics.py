# -*- coding: utf-8 -*-
"""
Unit tests for metrics module
"""

import unittest
import numpy as np
from nn_efficiency.metrics import (
    compute_entropy_metrics,
    compute_sparsity_metrics,
    compute_concentration_metrics,
    compute_statistical_metrics,
    compute_all_metrics
)


class TestMetrics(unittest.TestCase):
    """Test cases for metrics calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple importance matrix
        self.importance_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.05, 0.15, 0.25],
            [0.01, 0.02, 0.03]
        ])
        
        # Create a sparse matrix
        self.sparse_matrix = np.array([
            [0.001, 0.002, 0.5],
            [0.0005, 0.001, 0.3],
            [0.0001, 0.0002, 0.2]
        ])
    
    def test_compute_entropy_metrics(self):
        """Test entropy-based metrics calculation"""
        metrics = compute_entropy_metrics(self.importance_matrix)
        
        self.assertIn('entropy', metrics)
        self.assertIn('normalized_entropy', metrics)
        self.assertIn('effective_params', metrics)
        self.assertIn('utilization_ratio', metrics)
        
        # Check that values are in reasonable ranges
        self.assertGreater(metrics['entropy'], 0)
        self.assertGreater(metrics['normalized_entropy'], 0)
        self.assertLessEqual(metrics['normalized_entropy'], 1)
        self.assertGreater(metrics['effective_params'], 0)
        self.assertGreater(metrics['utilization_ratio'], 0)
        self.assertLessEqual(metrics['utilization_ratio'], 1)
    
    def test_compute_sparsity_metrics(self):
        """Test sparsity metrics calculation"""
        metrics = compute_sparsity_metrics(self.sparse_matrix)
        
        self.assertIn('sparsity_1e-2', metrics)
        self.assertIn('sparsity_1e-3', metrics)
        
        # Sparse matrix should have high sparsity
        self.assertGreater(metrics['sparsity_1e-2'], 0.5)
        self.assertGreater(metrics['sparsity_1e-3'], 0)
    
    def test_compute_concentration_metrics(self):
        """Test concentration metrics calculation"""
        metrics = compute_concentration_metrics(self.importance_matrix)
        
        self.assertIn('gini_coefficient', metrics)
        self.assertIn('top10_coverage', metrics)
        self.assertIn('top20_coverage', metrics)
        self.assertIn('top50_coverage', metrics)
        
        # Check ranges
        self.assertGreaterEqual(metrics['gini_coefficient'], 0)
        self.assertLessEqual(metrics['gini_coefficient'], 1)
    
    def test_compute_statistical_metrics(self):
        """Test statistical metrics calculation"""
        metrics = compute_statistical_metrics(self.importance_matrix)
        
        self.assertIn('mean_importance', metrics)
        self.assertIn('std_importance', metrics)
        self.assertIn('max_importance', metrics)
        
        # Verify values match numpy calculations
        flat = self.importance_matrix.flatten()
        self.assertAlmostEqual(metrics['mean_importance'], np.mean(flat))
        self.assertAlmostEqual(metrics['std_importance'], np.std(flat))
        self.assertAlmostEqual(metrics['max_importance'], np.max(flat))
    
    def test_compute_all_metrics(self):
        """Test that all metrics are computed together"""
        metrics = compute_all_metrics(self.importance_matrix)
        
        # Check that all expected keys are present
        expected_keys = [
            'total_params', 'entropy', 'normalized_entropy', 'effective_params',
            'utilization_ratio', 'sparsity_1e-2', 'sparsity_1e-3', 'gini_coefficient',
            'top10_coverage', 'top20_coverage', 'top50_coverage', 'mean_importance',
            'std_importance', 'max_importance', 'redundancy_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Verify total_params
        self.assertEqual(metrics['total_params'], self.importance_matrix.size)
        
        # Verify redundancy_score calculation
        self.assertAlmostEqual(
            metrics['redundancy_score'],
            1 - metrics['normalized_entropy']
        )
    
    def test_edge_case_zero_matrix(self):
        """Test handling of zero matrix"""
        zero_matrix = np.zeros((3, 3))
        metrics = compute_all_metrics(zero_matrix)
        
        # Should not raise errors and should return valid metrics
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['total_params'], 9)
    
    def test_edge_case_single_value(self):
        """Test handling of matrix with single non-zero value"""
        single_value = np.zeros((3, 3))
        single_value[1, 1] = 1.0
        metrics = compute_all_metrics(single_value)
        
        # Should have very high redundancy (all importance in one weight)
        self.assertGreater(metrics['redundancy_score'], 0.8)


if __name__ == '__main__':
    unittest.main()
