#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representation analysis for base and SFT models
Analyzes representation changes using multiple metrics
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def linear_kernel(X, Y):
    """Linear kernel for CKA"""
    return X @ Y.T


def rbf_kernel(X, Y, sigma=None):
    """RBF kernel for CKA"""
    if sigma is None:
        # Use median heuristic
        pairwise_dists = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T
        sigma = np.median(pairwise_dists) ** 0.5

    gamma = 1 / (2 * sigma ** 2)
    pairwise_dists = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * X @ Y.T
    return np.exp(-gamma * pairwise_dists)


def center_kernel(K):
    """Center a kernel matrix"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka(X, Y, kernel='linear'):
    """
    Centered Kernel Alignment (CKA)

    Args:
        X: Representations from model 1 [n_samples, n_features]
        Y: Representations from model 2 [n_samples, n_features]
        kernel: 'linear' or 'rbf'

    Returns:
        CKA similarity score
    """
    if kernel == 'linear':
        K_X = linear_kernel(X, X)
        K_Y = linear_kernel(Y, Y)
    elif kernel == 'rbf':
        K_X = rbf_kernel(X, X)
        K_Y = rbf_kernel(Y, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Center the kernel matrices
    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)

    # Compute CKA
    hsic_xy = np.trace(K_X_centered @ K_Y_centered)
    hsic_xx = np.trace(K_X_centered @ K_X_centered)
    hsic_yy = np.trace(K_Y_centered @ K_Y_centered)

    if hsic_xx * hsic_yy == 0:
        return 0.0

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def unbiased_cka(X, Y):
    """
    Unbiased CKA estimator

    Args:
        X: Representations from model 1 [n_samples, n_features]
        Y: Representations from model 2 [n_samples, n_features]

    Returns:
        Unbiased CKA similarity score
    """
    n = X.shape[0]

    if n < 4:
        return cka(X, Y, kernel='linear')

    # Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Remove diagonal
    np.fill_diagonal(K_X, 0)
    np.fill_diagonal(K_Y, 0)

    # Compute unbiased HSIC
    hsic_xy = np.trace(K_X @ K_Y) / (n * (n - 3))
    hsic_xy += np.sum(K_X) * np.sum(K_Y) / (n * (n - 1) * (n - 2) * (n - 3))
    hsic_xy -= 2 * np.sum(K_X @ K_Y) / (n * (n - 2) * (n - 3))

    hsic_xx = np.trace(K_X @ K_X) / (n * (n - 3))
    hsic_xx += np.sum(K_X) ** 2 / (n * (n - 1) * (n - 2) * (n - 3))
    hsic_xx -= 2 * np.sum(K_X @ K_X) / (n * (n - 2) * (n - 3))

    hsic_yy = np.trace(K_Y @ K_Y) / (n * (n - 3))
    hsic_yy += np.sum(K_Y) ** 2 / (n * (n - 1) * (n - 2) * (n - 3))
    hsic_yy -= 2 * np.sum(K_Y @ K_Y) / (n * (n - 2) * (n - 3))

    if hsic_xx * hsic_yy <= 0:
        return 0.0

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def procrustes_similarity(X, Y):
    """
    Orthogonal Procrustes similarity

    Args:
        X: Representations from model 1 [n_samples, n_features]
        Y: Representations from model 2 [n_samples, n_features]

    Returns:
        Procrustes similarity score
    """
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Normalize
    X_norm = X_centered / np.linalg.norm(X_centered, 'fro')
    Y_norm = Y_centered / np.linalg.norm(Y_centered, 'fro')

    # Compute optimal rotation
    U, _, Vt = np.linalg.svd(X_norm.T @ Y_norm)
    R = U @ Vt

    # Compute similarity
    X_aligned = X_norm @ R
    similarity = np.trace(X_aligned.T @ Y_norm) / min(X.shape)

    return similarity


def load_single_model_results(file_path):
    """
    Load results from a single model file

    Args:
        file_path: Path to the JSON file

    Returns:
        dict: Loaded results
    """
    print(f"Loading: {os.path.basename(file_path)}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'results' not in data:
            print(f"  Warning: No results found in {file_path}")
            return None

        # Extract model info
        model_info = {
            'model_name': data['model_name'],
            'model_type': data['model_type'],
            'results': []
        }

        # Only keep samples with representations
        for result in data['results']:
            if result.get('has_representations', False) and result.get('all_hidden_states'):
                # Extract only the first token's representation from each layer
                first_token_reprs = []
                hidden_states = result['all_hidden_states']

                if hidden_states and len(hidden_states) > 0:
                    # hidden_states[0] is the first token's representations across all layers
                    first_token_reprs = hidden_states[0]  # [num_layers, hidden_size]

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


def compute_layer_similarities(base_reprs, sft_reprs,
                               metrics=['cosine', 'linear_cka', 'rbf_cka', 'unbiased_cka', 'procrustes']):
    """
    Compute multiple similarity metrics between base and SFT representations

    Args:
        base_reprs: Base model representations [num_samples, num_layers, hidden_size]
        sft_reprs: SFT model representations [num_samples, num_layers, hidden_size]
        metrics: List of metrics to compute

    Returns:
        dict: Similarity scores for each metric and layer
    """
    num_layers = len(base_reprs[0])
    num_samples = len(base_reprs)

    results = {metric: [] for metric in metrics}

    for layer_idx in range(num_layers):
        # Collect representations for this layer across all samples
        X = np.array([base_reprs[i][layer_idx] for i in range(num_samples)])  # [n_samples, hidden_size]
        Y = np.array([sft_reprs[i][layer_idx] for i in range(num_samples)])  # [n_samples, hidden_size]

        layer_results = {}

        # Compute each metric
        if 'cosine' in metrics:
            # Average cosine similarity
            cos_sims = [cosine_similarity([X[i]], [Y[i]])[0, 0] for i in range(num_samples)]
            layer_results['cosine'] = np.mean(cos_sims)

        if 'linear_cka' in metrics:
            layer_results['linear_cka'] = cka(X, Y, kernel='linear')

        if 'rbf_cka' in metrics:
            layer_results['rbf_cka'] = cka(X, Y, kernel='rbf')

        if 'unbiased_cka' in metrics:
            layer_results['unbiased_cka'] = unbiased_cka(X, Y)

        if 'procrustes' in metrics:
            layer_results['procrustes'] = procrustes_similarity(X, Y)

        # Store results
        for metric in metrics:
            if metric in layer_results:
                results[metric].append(layer_results[metric])

    return results


def analyze_model_pair(base_file, sft_file, cache_key=None, cache_dir=None):
    """
    Analyze a pair of base and SFT models

    Args:
        base_file: Path to base model results
        sft_file: Path to SFT model results
        cache_key: Key for caching results
        cache_dir: Directory for cache files

    Returns:
        dict: Analysis results
    """
    # Check cache first
    if cache_key and cache_dir:
        cache_file = os.path.join(cache_dir, f'{cache_key}_analysis.json')
        if os.path.exists(cache_file):
            print(f"  Loading cached analysis from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Load model results
    base_data = load_single_model_results(base_file)
    sft_data = load_single_model_results(sft_file)

    if not base_data or not sft_data:
        return None

    # Match samples by ID
    base_dict = {r['sample_id']: r for r in base_data['results']}
    sft_dict = {r['sample_id']: r for r in sft_data['results']}

    common_ids = set(base_dict.keys()) & set(sft_dict.keys())
    print(f"  Found {len(common_ids)} paired samples")

    if len(common_ids) == 0:
        return None

    # Collect paired representations
    base_reprs = []
    sft_reprs = []

    for sample_id in sorted(common_ids):
        base_sample = base_dict[sample_id]
        sft_sample = sft_dict[sample_id]

        base_reprs.append(base_sample['first_token_representations'])
        sft_reprs.append(sft_sample['first_token_representations'])

    # Compute similarities
    similarities = compute_layer_similarities(base_reprs, sft_reprs)

    # Get number of layers
    num_layers = len(base_reprs[0])

    results = {
        'num_samples': len(common_ids),
        'num_layers': num_layers,
        'similarities': similarities
    }

    # Save to cache
    if cache_key and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{cache_key}_analysis.json')
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved analysis to cache: {cache_file}")

    # Clean up memory
    del base_data, sft_data, base_reprs, sft_reprs
    gc.collect()

    return results


def plot_layer_similarities_per_scale(analysis_results, scale, output_dir, dpi=300):
    """
    Plot layer similarities for a specific model scale

    Args:
        analysis_results: Analysis results for this scale
        scale: Model scale (e.g., '1B', '7B')
        output_dir: Output directory
        dpi: DPI for saving figures
    """
    # Prepare data for plotting
    metrics_to_plot = ['cosine', 'linear_cka', 'rbf_cka', 'unbiased_cka']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot for each dataset
        for dataset_type, dataset_label in [('idiom', 'Idiom'), ('poems', 'Poem')]:
            if dataset_type in analysis_results:
                result = analysis_results[dataset_type]
                if result and 'similarities' in result:
                    similarities = result['similarities']
                    if metric in similarities:
                        values = similarities[metric]
                        num_layers = result['num_layers']

                        # Use actual layer indices
                        x = np.arange(num_layers)

                        ax.plot(x, values, marker='o', label=dataset_label, linewidth=2)

        ax.set_xlabel('Layer Index', fontsize=11)
        ax.set_ylabel(f'{metric.replace("_", " ").title()} Similarity', fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Set x-axis to show integer layer indices
        if 'num_layers' in locals():
            ax.set_xticks(range(0, num_layers, max(1, num_layers // 10)))

    plt.suptitle(f'Representation Similarities Between Base and SFT - {scale} Model', fontsize=14)
    plt.tight_layout()

    # Save figure
    filename = f'layer_similarities_{scale}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def plot_cka_comparison(all_results, output_dir, dpi=300):
    """
    Plot CKA comparison across all scales and datasets

    Args:
        all_results: All analysis results
        output_dir: Output directory
        dpi: DPI for saving figures
    """
    # Calculate average CKA across layers for each configuration
    data_for_plot = []

    for scale in all_results:
        for dataset_type in all_results[scale]:
            result = all_results[scale][dataset_type]
            if result and 'similarities' in result:
                similarities = result['similarities']

                # Calculate mean CKA values
                for cka_type in ['linear_cka', 'rbf_cka', 'unbiased_cka']:
                    if cka_type in similarities:
                        mean_cka = np.mean(similarities[cka_type])
                        data_for_plot.append({
                            'Scale': scale,
                            'Dataset': 'Idiom' if dataset_type == 'idiom' else 'Poem',
                            'CKA Type': cka_type.replace('_', ' ').title(),
                            'Mean CKA': mean_cka
                        })

    if not data_for_plot:
        print("  No CKA data to plot")
        return

    df = pd.DataFrame(data_for_plot)

    # Create grouped bar plot
    plt.figure(figsize=(12, 6))

    # Group by scale
    scales = df['Scale'].unique()
    x = np.arange(len(scales))
    width = 0.25

    for i, cka_type in enumerate(df['CKA Type'].unique()):
        cka_data = df[df['CKA Type'] == cka_type]

        # Average across datasets for each scale
        mean_values = []
        for scale in scales:
            scale_data = cka_data[cka_data['Scale'] == scale]
            if not scale_data.empty:
                mean_values.append(scale_data['Mean CKA'].mean())
            else:
                mean_values.append(0)

        plt.bar(x + i * width, mean_values, width, label=cka_type)

    plt.xlabel('Model Scale', fontsize=12)
    plt.ylabel('Mean CKA Score', fontsize=12)
    plt.title('CKA Comparison Across Model Scales (Base vs SFT)', fontsize=14)
    plt.xticks(x + width, scales)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Save figure
    filename = 'cka_comparison_all_scales.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def save_final_results(all_results, output_dir):
    """
    Save all analysis results to JSON

    Args:
        all_results: All analysis results
        output_dir: Output directory
    """
    save_path = os.path.join(output_dir, 'representation_analysis_results.json')

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    serializable_results = convert_for_json(all_results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Saved final results to: {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze representation changes between base and SFT models')

    # Input/output parameters
    parser.add_argument('--results_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='Directory containing result files')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_representation_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory (default: output_dir/cache)')

    # Model scales to analyze
    parser.add_argument('--model_scales', type=str, nargs='+',
                        default=['1B'],
                        help='Model scales to analyze')

    # Visualization parameters
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')

    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, 'cache')

    print("=" * 60)
    print("Representation Analysis for Base and SFT Models")
    print("=" * 60)

    # Check if results already exist
    results_file = os.path.join(args.output_dir, 'representation_analysis_results.json')

    if os.path.exists(results_file):
        print(f"\nLoading existing results from: {results_file}")
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        print("\nNo existing results found. Starting analysis...")
        all_results = {}

        # Process each model scale
        for scale in args.model_scales:
            print(f"\n--- Processing {scale} models ---")
            all_results[scale] = {}

            # Process each dataset type
            for dataset_type in ['idiom', 'poems']:
                print(f"\n  Dataset: {dataset_type}")

                # Find base and SFT files with representations
                base_file = os.path.join(args.results_dir, f'{dataset_type}_{scale}_base_with_representations.json')
                sft_file = os.path.join(args.results_dir, f'{dataset_type}_{scale}_sft_with_representations.json')

                if not os.path.exists(base_file):
                    print(f"    Base file not found: {base_file}")
                    continue

                if not os.path.exists(sft_file):
                    print(f"    SFT file not found: {sft_file}")
                    continue

                # Analyze this model pair
                cache_key = f'{dataset_type}_{scale}'
                result = analyze_model_pair(base_file, sft_file, cache_key, args.cache_dir)

                if result:
                    all_results[scale][dataset_type] = result
                    print(f"    Analysis complete: {result['num_samples']} samples, {result['num_layers']} layers")

        # Save results
        save_final_results(all_results, args.output_dir)

    # Generate visualizations
    print("\n--- Generating Visualizations ---")

    # Plot layer similarities for each scale
    for scale in args.model_scales:
        if scale in all_results and all_results[scale]:
            print(f"\nPlotting {scale} model similarities...")
            plot_layer_similarities_per_scale(all_results[scale], scale, args.output_dir, args.dpi)

    # Plot CKA comparison across all scales
    print("\nPlotting CKA comparison...")
    plot_cka_comparison(all_results, args.output_dir, args.dpi)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()