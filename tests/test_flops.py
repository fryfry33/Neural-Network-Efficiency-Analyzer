# -*- coding: utf-8 -*-
"""
Unit tests for FLOPs calculation
"""

import unittest
import numpy as np
from nn_efficiency.metrics import compute_flops_dense, compute_flops_conv2d


class TestFLOPsCalculation(unittest.TestCase):
    """Test cases for FLOPs calculations"""
    
    def test_compute_flops_dense_basic(self):
        """Test FLOPs calculation for a simple dense layer"""
        result = compute_flops_dense(
            input_size=10,
            output_size=5,
            batch_size=1,
            has_bias=True
        )
        
        # Check that all keys are present
        self.assertIn('inference_flops', result)
        self.assertIn('training_flops', result)
        self.assertIn('inference_flops_per_sample', result)
        self.assertIn('training_flops_per_sample', result)
        
        # Check that training FLOPs > inference FLOPs
        self.assertGreater(result['training_flops'], result['inference_flops'])
        
        # Check that FLOPs are positive
        self.assertGreater(result['inference_flops'], 0)
        self.assertGreater(result['training_flops'], 0)
    
    def test_compute_flops_dense_batch(self):
        """Test FLOPs calculation with batch size"""
        result1 = compute_flops_dense(
            input_size=10,
            output_size=5,
            batch_size=1,
            has_bias=True
        )
        
        result32 = compute_flops_dense(
            input_size=10,
            output_size=5,
            batch_size=32,
            has_bias=True
        )
        
        # Batch FLOPs should scale with batch size
        self.assertAlmostEqual(
            result32['inference_flops'] / result1['inference_flops'],
            32.0,
            delta=1.0
        )
        
        # Per-sample FLOPs should be similar
        self.assertAlmostEqual(
            result1['inference_flops_per_sample'],
            result32['inference_flops_per_sample'],
            delta=10.0
        )
    
    def test_compute_flops_dense_no_bias(self):
        """Test FLOPs calculation without bias"""
        result_with_bias = compute_flops_dense(
            input_size=10,
            output_size=5,
            batch_size=1,
            has_bias=True
        )
        
        result_no_bias = compute_flops_dense(
            input_size=10,
            output_size=5,
            batch_size=1,
            has_bias=False
        )
        
        # Without bias should have fewer FLOPs
        self.assertLess(result_no_bias['inference_flops'], result_with_bias['inference_flops'])
    
    def test_compute_flops_conv2d_basic(self):
        """Test FLOPs calculation for a Conv2D layer"""
        result = compute_flops_conv2d(
            input_shape=(28, 28, 1),
            kernel_size=(3, 3),
            in_channels=1,
            out_channels=32,
            stride=1,
            padding=0,
            batch_size=1,
            has_bias=True
        )
        
        # Check that all keys are present
        self.assertIn('inference_flops', result)
        self.assertIn('training_flops', result)
        self.assertIn('output_shape', result)
        
        # Check that training FLOPs > inference FLOPs
        self.assertGreater(result['training_flops'], result['inference_flops'])
        
        # Check output shape
        self.assertEqual(len(result['output_shape']), 3)
        self.assertEqual(result['output_shape'][2], 32)  # out_channels
    
    def test_compute_flops_conv2d_stride(self):
        """Test FLOPs calculation with different strides"""
        result_stride1 = compute_flops_conv2d(
            input_shape=(28, 28, 1),
            kernel_size=(3, 3),
            in_channels=1,
            out_channels=32,
            stride=1,
            padding=0,
            batch_size=1,
            has_bias=True
        )
        
        result_stride2 = compute_flops_conv2d(
            input_shape=(28, 28, 1),
            kernel_size=(3, 3),
            in_channels=1,
            out_channels=32,
            stride=2,
            padding=0,
            batch_size=1,
            has_bias=True
        )
        
        # Stride 2 should reduce FLOPs (fewer output positions)
        self.assertLess(result_stride2['inference_flops'], result_stride1['inference_flops'])
        
        # Check output dimensions
        self.assertLess(result_stride2['output_shape'][0], result_stride1['output_shape'][0])
        self.assertLess(result_stride2['output_shape'][1], result_stride1['output_shape'][1])
    
    def test_compute_flops_conv2d_padding(self):
        """Test FLOPs calculation with padding"""
        result_no_pad = compute_flops_conv2d(
            input_shape=(28, 28, 1),
            kernel_size=(3, 3),
            in_channels=1,
            out_channels=32,
            stride=1,
            padding=0,
            batch_size=1,
            has_bias=True
        )
        
        result_with_pad = compute_flops_conv2d(
            input_shape=(28, 28, 1),
            kernel_size=(3, 3),
            in_channels=1,
            out_channels=32,
            stride=1,
            padding=1,
            batch_size=1,
            has_bias=True
        )
        
        # Padding should increase output size and thus FLOPs
        self.assertGreater(result_with_pad['inference_flops'], result_no_pad['inference_flops'])


if __name__ == '__main__':
    unittest.main()
