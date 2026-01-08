#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate separate normalized comparison plots
One independent PDF file per metric
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Plot style settings - as requested by the user
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 6.0
plt.rcParams['lines.markersize'] = 15
plt.rcParams['titleweight'] = 'bold'
plt.rcParams['labelweight'] = 'bold'
# --- Font Weight ---
 #   'axes.titleweight': 'bold',  # Bold title
  #  'axes.labelweight': 'bold',  # Bold labels

# Color scheme
COLOR_1B = '#D62728'  # Red
COLOR_7B = '#1F77B4'  # Blue

def load_data(json_file):
    """Load JSON data"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_metric_normalized(data_1b, data_7b, metric_key, 
                                   ylabel, title, output_file, dpi=300):
    """
    Plot a normalized comparison chart for a single metric
    
    Args:
        data_1b: 1B model data
        data_7b: 7B model data
        metric_key: Metric key name
        ylabel: Y-axis label
        title: Plot title
        output_file: Output file path
        dpi: Resolution
    """
    # Create figure - use the user-specified figsize
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get data
    layers_1b = np.arange(len(data_1b[metric_key]))
    layers_7b = np.arange(len(data_7b[metric_key]))
    values_1b = np.array(data_1b[metric_key])
    values_7b = np.array(data_7b[metric_key])
    
    # Normalize to [0, 1]
    min_1b, max_1b = values_1b.min(), values_1b.max()
    min_7b, max_7b = values_7b.min(), values_7b.max()
    
    if max_1b - min_1b > 1e-10:
        norm_1b = (values_1b - min_1b) / (max_1b - min_1b)
    else:
        norm_1b = np.zeros_like(values_1b)
        
    if max_7b - min_7b > 1e-10:
        norm_7b = (values_7b - min_7b) / (max_7b - min_7b)
    else:
        norm_7b = np.zeros_like(values_7b)
    
    # Plot curves
    ax.plot(layers_1b, norm_1b,
           marker='o',
           color=COLOR_1B,
           label='1B',
           alpha=0.8)
    ax.plot(layers_7b, norm_7b,
           marker='s',
           color=COLOR_7B,
           linestyle='--',
           label='7B',
           alpha=0.8)
    
    # Set labels - use the user-specified labelpad
    ax.set_xlabel('Layer Index', labelpad=2)
    ax.set_ylabel(ylabel)
    
    # Set title - do not append "(Normalized)", keep the original title
    ax.set_title(title)
    
    # Set Y-axis range and format
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    # Legend - remove range info
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # Grid - do not set background color
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def main():
    # Load data
    data = load_data('/mnt/user-data/uploads/1763344449209_representation_analysis_results.json')
    
    data_1b = data['1B']['idiom']['metrics']
    data_7b = data['7B']['idiom']['metrics']
    
    # Define metrics to plot - keep ylabel and title consistent with the original code
    metrics = [
        {
            'key': 'cka',
            'ylabel': 'CKA',
            'title': 'CKA',
            'output': '/mnt/user-data/outputs/cka_1B.pdf'
        },
        {
            'key': 'cosine',
            'ylabel': 'Cosine Similarity',
            'title': 'Cosine Similarity'
            'output': '/mnt/user-data/outputs/cosine_1B.pdf'
        },
        {
            'key': 'effective_rank_diff',
            'ylabel': 'Effective Rank Difference\n(Base - SFT)',
            'title': 'Effective Rank Difference',
            'output': '/mnt/user-data/outputs/effective_rank_diff_1B.pdf'
        },
        {
            'key': 'layer_distance',
            'ylabel': 'Layer-wise Distance (Frobenius)',
            'title': 'Layer-wise Distance',
            'output': '/mnt/user-data/outputs/layer_distance_1B.pdf'
        }
    ]
    
    # Generate charts for each metric
    print("正在生成归一化对比图...")
    for metric in metrics:
        plot_single_metric_normalized(
            data_1b, data_7b,
            metric['key'],
            metric['ylabel'],
            metric['title'],
            metric['output']
        )
    
    print("\n所有图表生成完成！")


if __name__ == '__main__':
    main()
