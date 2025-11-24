# -*- coding: utf-8 -*-
"""
Visualization utilities for neural network efficiency analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import NNEfficiencyAnalyzer


class Visualizer:
    """Visualization utilities for network analysis"""
    
    @staticmethod
    def plot_layer_importance_distribution(analyzer: 'NNEfficiencyAnalyzer', figsize=(15, 10)):
        """Plot importance distribution for each layer"""
        num_layers = len(analyzer.layer_analyses)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=figsize)
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for idx, la in enumerate(analyzer.layer_analyses):
            ax = axes[idx]
            flat_imp = la.importance_matrix.flatten()
            
            ax.hist(flat_imp, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_title(f'{la.name}\nUtilization: {la.metrics["utilization_ratio"]*100:.1f}%', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add threshold line
            ax.axvline(1e-2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Weight Importance Distribution by Layer', fontsize=14, fontweight='bold', y=1.02)
        plt.show()
    
    @staticmethod
    def plot_layer_comparison(analyzer: 'NNEfficiencyAnalyzer', figsize=(12, 6)):
        """Compare metrics across layers"""
        layer_names = [la.name for la in analyzer.layer_analyses]
        metrics_to_plot = ['utilization_ratio', 'redundancy_score', 'sparsity_1e-2']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [la.metrics[metric] * 100 for la in analyzer.layer_analyses]
            colors = sns.color_palette("RdYlGn_r" if metric == 'redundancy_score' else "RdYlGn", len(values))
            
            bars = axes[idx].bar(range(len(values)), values, color=colors, edgecolor='black')
            axes[idx].set_xticks(range(len(layer_names)))
            axes[idx].set_xticklabels(layer_names, rotation=45, ha='right')
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle('Layer-wise Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.show()
    
    @staticmethod
    def plot_efficiency_radar(analyzer: 'NNEfficiencyAnalyzer', layer_idx: int = 0, figsize=(8, 8)):
        """Create radar chart for layer efficiency metrics"""
        from math import pi
        
        la = analyzer.layer_analyses[layer_idx]
        
        categories = ['Utilization', 'Sparsity', 'Gini\n(inequality)', 
                     'Top 10%\nCoverage', 'Redundancy']
        values = [
            la.metrics['utilization_ratio'],
            la.metrics['sparsity_1e-2'],
            la.metrics['gini_coefficient'],
            la.metrics['top10_coverage'],
            la.metrics['redundancy_score']
        ]
        
        # Normalize to 0-100
        values = [v * 100 for v in values]
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_title(f'Efficiency Profile: {la.name}', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pruning_sensitivity(analyzer: 'NNEfficiencyAnalyzer', figsize=(12, 5)):
        """Visualize potential impact of pruning at different thresholds"""
        thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        for la in analyzer.layer_analyses:
            flat_imp = la.importance_matrix.flatten()
            pruned_ratios = [np.sum(flat_imp < t) / len(flat_imp) * 100 for t in thresholds]
            ax1.plot(range(len(thresholds)), pruned_ratios, marker='o', label=la.name, linewidth=2)
        
        ax1.set_xticks(range(len(thresholds)))
        ax1.set_xticklabels([f'{t:.0e}' for t in thresholds], rotation=45)
        ax1.set_xlabel('Pruning Threshold')
        ax1.set_ylabel('Weights Pruned (%)')
        ax1.set_title('Pruning Sensitivity Analysis')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative importance
        all_imp = np.concatenate([la.importance_matrix.flatten() for la in analyzer.layer_analyses])
        sorted_imp = np.sort(all_imp)[::-1]
        cumsum = np.cumsum(sorted_imp) / np.sum(sorted_imp) * 100
        
        ax2.plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum, linewidth=2, color='darkgreen')
        ax2.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% importance')
        ax2.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80% importance')
        ax2.set_xlabel('Percentage of Weights (%)')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Weight Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
