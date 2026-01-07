#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Representation analysis for base and SFT models
Performance improvements:
- Parallel processing of layers (multiprocessing)
- Vectorized cosine similarity
- Optimized nearest neighbor computation
- Batch processing for metrics
- Memory-efficient operations
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # <--- 在这里添加这一行
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import gc

warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# --- [新添加] 全局统一绘图样式 ---
# (你可以在这里调整你想要的默认大小)
plt.rcParams.update({
    # --- 字体大小 (Font Sizes) ---
    'font.size': 14,  # 基础字体大小 (用于常规文本)
    'axes.titlesize': 18,  # 图表标题 (Title)
    'axes.labelsize': 16,  # X 和 Y 轴标签 (Label)
    'xtick.labelsize': 14,  # X 轴刻度 (Tick labels)
    'ytick.labelsize': 14,  # Y 轴刻度 (Tick labels)
    'legend.fontsize': 14,  # 图例 (Legend)

    # --- 线条和标记 (Lines & Markers) ---
    'lines.linewidth': 3.0,  # 全局线宽
    'lines.markersize': 10,  # 全局标记大小

    # --- 字体粗细 (Font Weight) ---
    'axes.titleweight': 'bold',  # 标题加粗
    'axes.labelweight': 'bold',  # 标签加粗

    # --- 图例设置 ---
    'legend.frameon': True,  # 显示图例边框
    'legend.shadow': True,  # 图例显示阴影
})


# --- 结束全局设置 ---


def linear_kernel(X, Y):
    """Linear kernel for CKA - optimized"""
    return np.dot(X, Y.T)


def center_kernel(K):
    """Center a kernel matrix - optimized"""
    n = K.shape[0]
    # More efficient centering
    col_mean = K.mean(axis=0, keepdims=True)
    row_mean = K.mean(axis=1, keepdims=True)
    total_mean = K.mean()
    return K - col_mean - row_mean + total_mean


def cka(X, Y):
    """
    Centered Kernel Alignment (CKA) - optimized version

    Args:
        X: Representations from model 1 [n_samples, n_features]
        Y: Representations from model 2 [n_samples, n_features]

    Returns:
        CKA similarity score [0, 1]
    """
    K_X = np.dot(X, X.T)
    K_Y = np.dot(Y, Y.T)

    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)

    hsic_xy = np.sum(K_X_centered * K_Y_centered)  # Faster than trace
    hsic_xx = np.sum(K_X_centered * K_X_centered)
    hsic_yy = np.sum(K_Y_centered * K_Y_centered)

    if hsic_xx * hsic_yy == 0:
        return 0.0

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def compute_cosine_similarity(X, Y):
    """
    Average cosine similarity - VECTORIZED version

    Args:
        X, Y: [n_samples, n_features]

    Returns:
        Mean cosine similarity
    """
    # Vectorized computation - much faster!
    similarity_matrix = sklearn_cosine_similarity(X, Y)
    # Get diagonal (corresponding pairs)
    cos_sims = np.diag(similarity_matrix)
    return np.mean(cos_sims)


def procrustes_distance(X, Y):
    """
    Orthogonal Procrustes distance - optimized

    Args:
        X, Y: [n_samples, n_features]

    Returns:
        Normalized Procrustes distance
    """
    # Center
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Normalize using frobenius norm
    X_norm_factor = np.linalg.norm(X_centered, 'fro') + 1e-8
    Y_norm_factor = np.linalg.norm(Y_centered, 'fro') + 1e-8

    X_norm = X_centered / X_norm_factor
    Y_norm = Y_centered / Y_norm_factor

    # Find optimal rotation using SVD
    U, _, Vt = np.linalg.svd(X_norm.T @ Y_norm, full_matrices=False)
    R = U @ Vt

    # Compute distance after alignment
    X_aligned = X_norm @ R
    distance = np.linalg.norm(X_aligned - Y_norm, 'fro')

    return distance


def effective_rank(X):
    """
    Effective rank based on singular value entropy - optimized

    Args:
        X: [n_samples, n_features]

    Returns:
        Effective rank (higher = more dimensions used)
    """
    # Compute SVD - only singular values needed
    s = np.linalg.svd(X, compute_uv=False, full_matrices=False)

    # Normalize singular values
    s_squared = s ** 2
    s_normalized = s_squared / (s_squared.sum() + 1e-12)

    # Compute entropy
    ent = -np.sum(s_normalized * np.log(s_normalized + 1e-12))

    # Return exp(entropy)
    return np.exp(ent)


def anisotropy(X, top_k=10):
    """
    Anisotropy: concentration of variance in top singular directions - optimized

    Args:
        X: [n_samples, n_features]
        top_k: Number of top singular values to consider

    Returns:
        Anisotropy coefficient [0, 1] (higher = more concentrated)
    """
    # Compute only singular values
    s = np.linalg.svd(X, compute_uv=False, full_matrices=False)

    # Variance explained by top k components
    s_squared = s ** 2
    total_var = s_squared.sum()
    top_k_var = s_squared[:top_k].sum()

    return top_k_var / (total_var + 1e-12)


def mean_representation_shift(X, Y):
    """
    L2 distance between representation centroids

    Args:
        X, Y: [n_samples, n_features]

    Returns:
        L2 distance between means
    """
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)

    return np.linalg.norm(mean_X - mean_Y)


def layer_wise_distance(X, Y):
    """
    Frobenius norm of difference - optimized

    Args:
        X, Y: [n_samples, n_features]

    Returns:
        Normalized Frobenius distance
    """
    diff = X - Y
    norm_factor = np.sqrt(X.shape[0] * X.shape[1])
    return np.linalg.norm(diff, 'fro') / norm_factor


def nearest_neighbor_preservation(X, Y, k=5):
    """
    Jaccard similarity of k-nearest neighbor sets - optimized

    Args:
        X, Y: [n_samples, n_features]
        k: Number of neighbors

    Returns:
        Mean Jaccard similarity [0, 1]
    """
    n_samples = X.shape[0]

    # Handle edge cases
    k = min(k, n_samples - 1)
    if k < 1:
        return 1.0

    # Find k-NN in both spaces using cosine metric
    nbrs_X = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1).fit(X)
    _, indices_X = nbrs_X.kneighbors(X)
    indices_X = indices_X[:, 1:]  # Exclude self

    nbrs_Y = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1).fit(Y)
    _, indices_Y = nbrs_Y.kneighbors(Y)
    indices_Y = indices_Y[:, 1:]  # Exclude self

    # Vectorized Jaccard computation
    jaccard_scores = []
    for i in range(n_samples):
        set_X = set(indices_X[i])
        set_Y = set(indices_Y[i])

        intersection = len(set_X & set_Y)
        union = len(set_X | set_Y)

        jaccard_scores.append(intersection / union if union > 0 else 1.0)

    return np.mean(jaccard_scores)


def compute_all_metrics(X, Y, k_neighbors=5):
    """
    Compute all representation similarity metrics

    Args:
        X, Y: [n_samples, n_features] representations
        k_neighbors: k for nearest neighbor preservation

    Returns:
        dict of metric values
    """
    metrics = {}

    try:
        # 1. CKA
        metrics['cka'] = cka(X, Y)

        # 2. Cosine Similarity (vectorized)
        metrics['cosine'] = compute_cosine_similarity(X, Y)

        # 3. Procrustes Distance
        # metrics['procrustes_distance'] = procrustes_distance(X, Y)

        # 4. Effective Rank (for both X and Y)
        metrics['effective_rank_base'] = effective_rank(X)
        metrics['effective_rank_sft'] = effective_rank(Y)
        metrics['effective_rank_diff'] = metrics['effective_rank_base'] - metrics['effective_rank_sft']

        # 5. Anisotropy
        top_k = min(10, X.shape[1])
        # metrics['anisotropy_base'] = anisotropy(X, top_k=top_k)
        # metrics['anisotropy_sft'] = anisotropy(Y, top_k=top_k)
        # metrics['anisotropy_diff'] = metrics['anisotropy_sft'] - metrics['anisotropy_base']

        # 6. Mean Representation Shift
        # metrics['mean_shift'] = mean_representation_shift(X, Y)

        # 7. Layer-wise Distance
        metrics['layer_distance'] = layer_wise_distance(X, Y)

        # 8. Nearest Neighbor Preservation
        # metrics['nn_preservation'] = nearest_neighbor_preservation(X, Y, k=k_neighbors)

    except Exception as e:
        print(f"    Error computing metrics: {e}")
        return None

    return metrics


def process_single_layer(args):
    """
    Process a single layer (for parallel processing)

    Args:
        args: tuple of (layer_idx, base_reprs, sft_reprs, k_neighbors)

    Returns:
        tuple of (layer_idx, metrics)
    """
    layer_idx, base_layer_data, sft_layer_data, k_neighbors = args

    X = np.array(base_layer_data)
    Y = np.array(sft_layer_data)

    metrics = compute_all_metrics(X, Y, k_neighbors=k_neighbors)

    return layer_idx, metrics


def load_single_model_results(file_path):
    """Load results from a single model file"""
    print(f"Loading: {os.path.basename(file_path)}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'results' not in data:
            print(f"  Warning: No results found in {file_path}")
            return None

        model_info = {
            'model_name': data['model_name'],
            'model_type': data['model_type'],
            'results': []
        }

        for result in data['results']:
            if result.get('has_representations', False) and result.get('all_hidden_states'):
                first_token_reprs = []
                hidden_states = result['all_hidden_states']

                if hidden_states and len(hidden_states) > 0:
                    first_token_reprs = hidden_states[0]

                    model_info['results'].append({
                        'sample_id': result['sample_id'],
                        'dataset_type': result['dataset_type'],
                        'first_token_representations': first_token_reprs,
                        'prompt': result.get('prompt', ''),
                        'generated': result.get('generated', '')
                    })

        print(f"  Loaded {len(model_info['results'])} samples with representations")
        return model_info

    except Exception as e:
        print(f"  Error loading file: {e}")
        return None


def analyze_model_pair(base_file, sft_file, cache_key=None, cache_dir=None, n_jobs=-1):
    """
    Analyze a pair of base and SFT models with parallel processing

    Args:
        base_file: Path to base model results
        sft_file: Path to SFT model results
        cache_key: Cache key for saving results
        cache_dir: Cache directory
        n_jobs: Number of parallel jobs (-1 = all cores)
    """
    # Check cache
    if cache_key and cache_dir:
        cache_file = os.path.join(cache_dir, f'{cache_key}_analysis.json')
        if os.path.exists(cache_file):
            print(f"  Loading cached analysis from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Load data
    base_data = load_single_model_results(base_file)
    sft_data = load_single_model_results(sft_file)

    if not base_data or not sft_data:
        return None

    # Match samples
    base_dict = {r['sample_id']: r for r in base_data['results']}
    sft_dict = {r['sample_id']: r for r in sft_data['results']}
    common_ids = set(base_dict.keys()) & set(sft_dict.keys())

    print(f"  Found {len(common_ids)} paired samples")

    if len(common_ids) == 0:
        return None

    # Collect representations
    base_reprs = []
    sft_reprs = []

    for sample_id in sorted(common_ids):
        base_reprs.append(base_dict[sample_id]['first_token_representations'])
        sft_reprs.append(sft_dict[sample_id]['first_token_representations'])

    num_layers = len(base_reprs[0])
    num_samples = len(base_reprs)

    print(f"  Processing {num_layers} layers with {num_samples} samples...")
    print(f"  Using parallel processing with {cpu_count() if n_jobs == -1 else n_jobs} workers")

    # Prepare data for parallel processing
    # Reorganize data: layer -> samples instead of sample -> layers
    layer_data = []
    for layer_idx in range(num_layers):
        base_layer = [base_reprs[i][layer_idx] for i in range(num_samples)]
        sft_layer = [sft_reprs[i][layer_idx] for i in range(num_samples)]
        layer_data.append((layer_idx, base_layer, sft_layer, 5))

    # Parallel processing of layers
    if n_jobs == -1:
        n_jobs = cpu_count()

    layer_metrics = defaultdict(list)

    with Pool(processes=n_jobs) as pool:
        # Process layers in parallel
        results = pool.map(process_single_layer, layer_data)

    # Collect results
    for layer_idx, metrics in sorted(results, key=lambda x: x[0]):
        if metrics:
            for metric_name, value in metrics.items():
                layer_metrics[metric_name].append(float(value))

    print(f"  ✓ Processing complete!")

    results = {
        'num_samples': num_samples,
        'num_layers': num_layers,
        'metrics': dict(layer_metrics)
    }

    # Save to cache
    if cache_key and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{cache_key}_analysis.json')
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to cache: {cache_file}")

    # Clean up
    del base_data, sft_data, base_reprs, sft_reprs
    gc.collect()

    return results


def plot_single_metric(all_results, metric_name, scale, output_dir, dpi=300):
    """Plot a single metric across layers for one model scale"""
    plt.figure(figsize=(8, 6))

    # --- 1. 应用新的样式和配色方案 ---

    # 3. 为每个图表类型定义专属配色 (idiom, poems)
    metric_colors = {
        'cka': {'idiom': '#d81159', 'poems': '#8f2d56'},  # 红色/紫色系
        'cosine': {'idiom': '#0081a7', 'poems': '#00afb9'},  # 蓝色/青色系
        'effective_rank_diff': {'idiom': '#f77f00', 'poems': '#fcbf49'},  # 橙色/黄色系
        'layer_distance': {'idiom': '#2d6a4f', 'poems': '#52b788'},  # 绿色系
        'default': {'idiom': '#2E86AB', 'poems': '#A23B72'}  # 默认
    }
    colors = metric_colors.get(metric_name, metric_colors['default'])

    # 2. 为每个图表类型定义专属标记 (idiom, poems)
    metric_markers = {
        'cka': {'idiom': 'X', 'poems': 'D'},  # CKA: X 和 菱形
        'cosine': {'idiom': '^', 'poems': 'v'},  # Cosine: 三角形 (上/下)
        'effective_rank_diff': {'idiom': 's', 'poems': 'P'},  # Rank: 方形 和 十字
        'layer_distance': {'idiom': 'o', 'poems': '*'},  # Distance: 圆形 和 星形
        'default': {'idiom': 'o', 'poems': 's'}  # 默认
    }
    markers = metric_markers.get(metric_name, metric_markers['default'])

    # --- [已删除] 旧的硬编码样式 ---
    # 移除了 metric_ylims 字典，Y轴将自动缩放

    for dataset_type in ['idiom', 'poems']:
        if dataset_type in all_results:
            result = all_results[dataset_type]
            if result and 'metrics' in result and metric_name in result['metrics']:
                values = result['metrics'][metric_name]
                num_layers = result['num_layers']
                layers = np.arange(num_layers)

                label = 'Idiom' if dataset_type == 'idiom' else 'Poem'

                # [已修改] plt.plot() 现在使用全局 rcParams 的线宽和标记大小
                plt.plot(layers, values,
                         marker=markers[dataset_type],
                         color=colors[dataset_type],
                         label=label,
                         alpha=0.8)

    # 移除了设置固定 plt.ylim() 的 if 语句

    # [已修改] xlabel 现在使用全局 rcParams 的字体
    plt.xlabel('Layer Index', labelpad=2)

    # Metric-specific y-label (不变)
    ylabel_map = {
        'cka': 'CKA Similarity',
        'cosine': 'Cosine Similarity',
        'procrustes_distance': 'Procrustes Distance',
        'effective_rank_diff': 'Effective Rank Difference\n(Base - SFT)',
        'anisotropy_diff': 'Anisotropy Difference\n(SFT - Base)',
        'mean_shift': 'Mean Representation Shift (L2)',
        'layer_distance': 'Layer-wise Distance (Frobenius)',
        'nn_preservation': 'Nearest Neighbor Preservation'
    }

    ylabel = ylabel_map.get(metric_name, metric_name.replace('_', ' ').title())
    # [已修改] ylabel 现在使用全局 rcParams 的字体
    # plt.ylabel(ylabel) # 你可以取消这行的注释并设置Y轴标签

    # 设置Y轴刻度格式为小数点后两位 (保留)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    # Title (不变)
    title_map = {
        'cka': 'CKA',
        'cosine': 'Cosine Similarity',
        'procrustes_distance': 'Procrustes Distance',
        'effective_rank_diff': 'Effective Rank Difference',
        'anisotropy_diff': 'Anisotropy Change',
        'mean_shift': 'Mean Representation Shift',
        'layer_distance': 'Layer-wise Distance',
        'nn_preservation': 'Nearest Neighbor Preservation'
    }

    title = title_map.get(metric_name, metric_name.replace('_', ' ').title())
    # [已修改] title 现在使用全局 rcParams 的字体
    plt.title(f'{title} ({scale})', pad=3)

    # [已修改] legend 现在使用全局 rcParams 的字体和样式
    # plt.legend() # 你可以取消这行的注释来显示图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save (不变)
    filename = f'{metric_name}_{scale}.pdf'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"  Saved: {save_path}")


def plot_rank_and_anisotropy_comparison(all_results, scale, output_dir, dpi=300):
    """Plot effective rank and anisotropy for both base and SFT"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    colors = {'base': '#E63946', 'sft': '#06FFA5'}

    # Plot 1: Effective Rank
    ax = axes[0]
    for dataset_type, dataset_label in [('idiom', 'Idiom'), ('poems', 'Poem')]:
        if dataset_type in all_results:
            result = all_results[dataset_type]
            if result and 'metrics' in result:
                metrics = result['metrics']
                num_layers = result['num_layers']
                layers = np.arange(num_layers)

                if 'effective_rank_base' in metrics and 'effective_rank_sft' in metrics:
                    # [已修改] 移除硬编码的 linewidth
                    ax.plot(layers, metrics['effective_rank_base'],
                            marker='o', label=f'{dataset_label} (Base)',
                            color=colors['base'], alpha=0.7)
                    ax.plot(layers, metrics['effective_rank_sft'],
                            marker='s', label=f'{dataset_label} (SFT)',
                            color=colors['sft'], alpha=0.7)

    # [已修改] 移除硬编码的 fontsize 和 fontweight
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Effective Rank')
    ax.set_title(f'Effective Rank Comparison ({scale})')
    ax.legend()  # [已修改] 移除硬编码的 fontsize
    ax.grid(True, alpha=0.3)

    # Plot 2: Anisotropy
    ax = axes[1]
    for dataset_type, dataset_label in [('idiom', 'Idiom'), ('poems', 'Poem')]:
        if dataset_type in all_results:
            result = all_results[dataset_type]
            if result and 'metrics' in result:
                metrics = result['metrics']
                num_layers = result['num_layers']
                layers = np.arange(num_layers)

                if 'anisotropy_base' in metrics and 'anisotropy_sft' in metrics:
                    # [已修改] 移除硬编码的 linewidth
                    ax.plot(layers, metrics['anisotropy_base'],
                            marker='o', label=f'{dataset_label} (Base)',
                            color=colors['base'], alpha=0.7)
                    ax.plot(layers, metrics['anisotropy_sft'],
                            marker='s', label=f'{dataset_label} (SFT)',
                            color=colors['sft'], alpha=0.7)

    # [已修改] 移除硬编码的 fontsize 和 fontweight
    ax.set_xlabel('Layer Index')
    # ax.set_ylabel('Anisotropy Coefficient') # [已修改] 移除硬编码的 fontsize 和 fontweight
    ax.set_title(f'Anisotropy Comparison - {scale}')
    ax.legend()  # [已修改] 移除硬编码的 fontsize
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f'rank_anisotropy_comparison_{scale}.pdf'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def save_final_results(all_results, output_dir):
    """Save all analysis results to JSON"""
    save_path = os.path.join(output_dir, 'representation_analysis_results.json')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved final results to: {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimized comprehensive representation analysis')

    parser.add_argument('--results_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='Directory containing result files')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_representation_analysis',
                        help='Output directory')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory')
    parser.add_argument('--model_scales', type=str, nargs='+',
                        default=['1B', '7B', '13B'],
                        help='Model scales to analyze')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for figures')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, 'cache')

    print("=" * 70)
    print("  OPTIMIZED Comprehensive Representation Analysis")
    print("  Base vs SFT Models with Parallel Processing")
    print("=" * 70)
    print(f"  Using {cpu_count() if args.n_jobs == -1 else args.n_jobs} CPU cores")
    print("=" * 70)

    # Check for existing results
    results_file = os.path.join(args.output_dir, 'representation_analysis_results.json')

    if os.path.exists(results_file):
        print(f"\nLoading existing results from: {results_file}")
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        print("\nStarting fresh analysis...")
        all_results = {}

        for scale in args.model_scales:
            print(f"\n{'=' * 70}")
            print(f"  Processing {scale} Model")
            print(f"{'=' * 70}")
            all_results[scale] = {}

            for dataset_type in ['idiom']:
                print(f"\nDataset: {dataset_type}")

                base_file = os.path.join(args.results_dir,
                                         f'{dataset_type}_{scale}_base_with_representations.json')
                sft_file = os.path.join(args.results_dir,
                                        f'{dataset_type}_{scale}_sft_with_representations.json')

                if not os.path.exists(base_file):
                    print(f"  Base file not found: {base_file}")
                    continue

                if not os.path.exists(sft_file):
                    print(f"  SFT file not found: {sft_file}")
                    continue

                cache_key = f'{dataset_type}_{scale}'
                result = analyze_model_pair(base_file, sft_file, cache_key,
                                            args.cache_dir, n_jobs=args.n_jobs)

                if result:
                    all_results[scale][dataset_type] = result
                    print(f"  ✓ Complete: {result['num_samples']} samples, "
                          f"{result['num_layers']} layers")

        save_final_results(all_results, args.output_dir)

    # Generate visualizations
    print(f"\n{'=' * 70}")
    print("  Generating Visualizations")
    print(f"{'=' * 70}")

    metrics_to_plot = [
        'cka',
        'cosine',
        # 'procrustes_distance',
        'effective_rank_diff',
        # 'anisotropy_diff',
        # 'mean_shift',
        'layer_distance',
        # 'nn_preservation'
    ]

    for scale in args.model_scales:
        if scale not in all_results or not all_results[scale]:
            continue

        print(f"\nGenerating plots for {scale} model...")

        # Plot each metric separately
        for metric in metrics_to_plot:
            plot_single_metric(all_results[scale], metric, scale, args.output_dir, args.dpi)

        # Special plot for rank and anisotropy comparison
        # plot_rank_and_anisotropy_comparison(all_results[scale], scale, args.output_dir, args.dpi)

    print(f"\n{'=' * 70}")
    print("  Analysis Complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()