# -*- coding: utf-8 -*-
"""
Utility functions for neural network efficiency analysis
"""

import numpy as np
from .analyzer import NNEfficiencyAnalyzer
from .visualizer import Visualizer


def quick_analyze(model, sample_data: np.ndarray, framework: str = 'auto', 
                  visualize: bool = True) -> NNEfficiencyAnalyzer:
    """
    Convenience function for quick model analysis
    
    Args:
        model: Neural network model (TensorFlow or PyTorch)
        sample_data: Representative input data
        framework: 'tensorflow', 'pytorch', or 'auto'
        visualize: Whether to show visualizations
    
    Returns:
        NNEfficiencyAnalyzer instance with results
    """
    analyzer = NNEfficiencyAnalyzer(model, framework)
    analyzer.analyze(sample_data)
    analyzer.print_report()
    
    if visualize:
        viz = Visualizer()
        viz.plot_layer_importance_distribution(analyzer)
        viz.plot_layer_comparison(analyzer)
        viz.plot_pruning_sensitivity(analyzer)
    
    return analyzer
