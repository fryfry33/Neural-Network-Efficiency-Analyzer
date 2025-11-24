# -*- coding: utf-8 -*-
"""
Neural Network Efficiency Analyzer (NNEfficiency)
A comprehensive toolkit for analyzing, visualizing, and optimizing neural network efficiency
through weight importance analysis, pruning recommendations, and computational cost estimation.
"""

from .analyzer import NNEfficiencyAnalyzer, LayerAnalysis
from .visualizer import Visualizer
from .utils import quick_analyze

__version__ = "0.1.0"
__author__ = "Neural Network Efficiency Analyzer Team"
__all__ = ["NNEfficiencyAnalyzer", "LayerAnalysis", "Visualizer", "quick_analyze"]
