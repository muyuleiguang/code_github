#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import glob
import pandas as pd
from typing import Dict, List
import warnings
# Import memorization evaluation metrics
from memorization_metrics import MemorizationMetrics

warnings.filterwarnings('ignore')


def sort_model_scales(model_scales):
    """
    Sort model scales by their numeric size, e.g., 1B < 7B < 13B

    Args:
        model_scales: List of model scales

    Returns:
        sorted_scales: Sorted list of model scales
    """

    def extract_scale_value(scale_str):
        """Extract the numeric part of the model scale string for sorting"""
        try:
            # Remove the unit suffix (B, M, etc.)
            if scale_str.endswith('B'):
                return float(scale_str[:-1])
            elif scale_str.endswith('M'):
                return float(scale_str[:-1]) / 1000  # Convert to B units
            else:
                # If no unit, treat it as a number directly
                return float(scale_str)
        except:
            # If parsing fails, return a large value so it is placed at the end
            return float('inf')

    return sorted(model_scales, key=extract_scale_value)


def load_generation_results(results_dir: str,
                            model_scales: List[str],
                            datasets: List[str],
                            prefix_lengths: List[int],
                            max_new_tokens: List[int],
                            max_samples: int = None) -> Dict[str, Dict]:
    """
    Load generation result files across multiple datasets and model scales

    File naming format:
        {dataset}_prefix{prefix_length}_new{max_new_tokens}_{model_scale}_{model_type}_{max_samples}_samples.jsonl
    File location:
        results_dir/exp1_generation_{max_new_tokens}/

    Args:
        results_dir: Root results directory (e.g., /root/autodl-tmp/ift_memorization/results)
        model_scales: List of model scales (e.g., ["1B", "7B", "13B", "32B"])
        datasets: List of datasets (e.g., ['stackexchange', 'dclm-privacy', 'wiki-fact'])
        prefix_lengths: List of prefix lengths (e.g., [16, 32, 64])
        max_new_tokens: List of max new token values (e.g., [8, 16])
        max_samples: Maximum number of samples per condition

    Returns:
        results_dict: Results organized by dataset, model scale, model type, prefix length, and max_new_tokens
        Format: {dataset: {model_scale: {model_type: {prefix_length: {max_new_tokens: [samples]}}}}}
    """
    results_dict = {}

    print("\n开始加载文件...")
    print(f"数据集: {datasets}")
    print(f"模型规模: {model_scales}")
    print(f"前缀长度: {prefix_lengths}")
    print(f"最大新token数: {max_new_tokens}")
    print("-" * 80)

    # Iterate over all parameter combinations
    for dataset in datasets:
        for model_scale in model_scales:
            for model_type in ['base', 'sft']:
                for prefix_length in prefix_lengths:
                    for max_new_token in max_new_tokens:
                        # Build file path
                        # Files are under the exp1_generation_{max_new_token} folder
                        folder_name = f"exp1_generation_{max_new_token}"
                        file_name = f"{dataset}_prefix{prefix_length}_new{max_new_token}_{model_scale}_{model_type}_{max_samples}_samples.jsonl"
                        file_path = os.path.join(results_dir, folder_name, file_name)

                        # Check if file exists
                        if not os.path.exists(file_path):
                            print(f"✗ 文件不存在: {file_name}")
                            continue

                        # Read file
                        try:
                            samples = []
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        sample = json.loads(line)
                                        samples.append(sample)

                            # Apply sample limit
                            if max_samples and len(samples) > max_samples:
                                samples = samples[:max_samples]

                            # Organize data structure
                            if dataset not in results_dict:
                                results_dict[dataset] = {}
                            if model_scale not in results_dict[dataset]:
                                results_dict[dataset][model_scale] = {'base': {}, 'sft': {}}
                            if prefix_length not in results_dict[dataset][model_scale][model_type]:
                                results_dict[dataset][model_scale][model_type][prefix_length] = {}

                            results_dict[dataset][model_scale][model_type][prefix_length][max_new_token] = samples

                            print(
                                f"✓ {dataset}-{model_scale}-{model_type}-prefix{prefix_length}-new{max_new_token}: {len(samples)} 条样本")

                        except Exception as e:
                            print(f"✗ 读取文件出错 {file_name}: {e}")
                            continue

    print("-" * 80)
    print(f"文件加载完成！\n")

    return results_dict


def calculate_memorization_metrics_with_evaluator(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compute memorization metrics using MemorizationMetrics

    Args:
        results_dict: Generation results dictionary

    Returns:
        metrics_df: DataFrame of memorization metrics
    """
    # Initialize evaluator
    evaluator = MemorizationMetrics()
    metrics_data = []

    for dataset in results_dict:
        for model_scale in results_dict[dataset]:
            for model_type in ['base', 'sft']:
                if model_type not in results_dict[dataset][model_scale]:
                    continue

                for prefix_length in results_dict[dataset][model_scale][model_type]:
                    for max_new_tokens, samples in results_dict[dataset][model_scale][model_type][
                        prefix_length].items():
                        if not samples:
                            continue

                        print(
                            f"计算 {dataset}-{model_scale}-{model_type}-prefix{prefix_length}-maxtokens{max_new_tokens} 的记忆指标...")

                        # Use the new unified interface to compute all metrics
                        try:
                            metrics_results = evaluator.compute_all_metrics_from_data(samples)

                            # Extract metrics into a single record
                            metrics_entry = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'model_scale': model_scale,
                                'prefix_length': prefix_length,
                                'max_new_tokens': max_new_tokens,
                                'sample_count': len(samples),
                            }

                            # 1) Exact match
                            if 'exact_match' in metrics_results:
                                metrics_entry['exact_match_rate'] = metrics_results['exact_match']['exact_match_rate']

                            # 2) ROUGE/BLEU
                            if 'rouge_bleu' in metrics_results:
                                rouge_bleu = metrics_results['rouge_bleu']
                                metrics_entry['rouge_1_f'] = rouge_bleu.get('rouge_1_f', 0.0)
                                metrics_entry['rouge_2_f'] = rouge_bleu.get('rouge_2_f', 0.0)
                                metrics_entry['rouge_l_f'] = rouge_bleu.get('rouge_l_f', 0.0)
                                metrics_entry['bleu_1'] = rouge_bleu.get('bleu_1', 0.0)
                                metrics_entry['bleu_2'] = rouge_bleu.get('bleu_2', 0.0)
                                metrics_entry['bleu_4'] = rouge_bleu.get('bleu_4', 0.0)

                            # 3) Edit distance
                            if 'edit_distance' in metrics_results:
                                edit_dist = metrics_results['edit_distance']
                                metrics_entry['token_edit_distance'] = edit_dist.get('token_edit_distance', 0.0)
                                metrics_entry['normalized_token_distance'] = edit_dist.get('normalized_token_distance',
                                                                                           0.0)

                            # 4) Semantic similarity
                            if 'semantic' in metrics_results:
                                semantic = metrics_results['semantic']
                                metrics_entry['semantic_similarity'] = semantic.get('cosine_similarity', 0.0)

                            # 5) Likelihood-related metrics
                            if 'likelihood' in metrics_results:
                                likelihood = metrics_results['likelihood']
                                metrics_entry['target_token_probability'] = likelihood.get('target_token_probability',
                                                                                           0.0)
                                metrics_entry['target_token_rank'] = likelihood.get('target_token_rank', float('inf'))
                                metrics_entry['target_in_top1_rate'] = likelihood.get('target_in_top1_rate', 0.0)
                                metrics_entry['target_in_top3_rate'] = likelihood.get('target_in_top3_rate', 0.0)
                                metrics_entry['target_in_top5_rate'] = likelihood.get('target_in_top5_rate', 0.0)

                            metrics_data.append(metrics_entry)

                        except Exception as e:
                            print(
                                f"计算指标时出错 {dataset}-{model_scale}-{model_type}-prefix{prefix_length}-maxtokens{max_new_tokens}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

    return pd.DataFrame(metrics_data)


def generate_latex_tables(metrics_df: pd.DataFrame, output_dir: str, prefix_lengths: List[int],
                          max_new_tokens: List[int]):
    """
    Generate LaTeX tables with the correct structure:
    Column headers: Dataset name + Generation Length
    Row headers: Model scale + type + Prefix Length

    Args:
        metrics_df: DataFrame containing computed metrics
        output_dir: Output directory
        prefix_lengths: List of prefix lengths
        max_new_tokens: List of max new token values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Dataset mapping (for display)
    dataset_mapping = {
        'stackexchange': 'STACKEXCHANGE',
        'dclm-privacy': 'DCLM-PRIVACY',
        'wiki-fact': 'WIKI-FACT'
    }

    # Metrics to generate tables for
    metrics_to_generate = [
        ('exact_match_rate', 'Exact Match Rate'),
        ('rouge_1_f', 'ROUGE-1 F-score'),
        ('rouge_2_f', 'ROUGE-2 F-score'),
        ('rouge_l_f', 'ROUGE-L F-score'),
        ('bleu_1', 'BLEU-1'),
        ('bleu_2', 'BLEU-2'),
        ('bleu_4', 'BLEU-4'),
        ('token_edit_distance', 'Token Edit Distance'),
        ('normalized_token_distance', 'Normalized Token Distance'),
        ('semantic_similarity', 'Semantic Similarity'),
        ('target_token_probability', 'Target Token Probability'),
        ('target_in_top1_rate', 'Target in Top-1 Rate'),
        ('target_in_top3_rate', 'Target in Top-3 Rate'),
        ('target_in_top5_rate', 'Target in Top-5 Rate')
    ]

    # Collect all LaTeX tables
    all_latex_tables = []

    # Get unique datasets and models
    datasets = sorted(metrics_df['dataset'].unique())
    model_scales = sort_model_scales(metrics_df['model_scale'].unique())

    # Generate one large table per metric
    for metric_col, metric_name in metrics_to_generate:
        if metric_col not in metrics_df.columns:
            continue

        print(f"生成 {metric_name} 的大表格...")

        # Build table data following the correct row/column layout
        table_data = []
        row_headers = []

        # Row structure: model scale + type × prefix_length
        for model_scale in model_scales:
            for model_type in ['base', 'sft']:
                model_label = f"{model_scale}" if model_type == 'base' else f"{model_scale}-sft"

                for prefix_length in prefix_lengths:
                    row_data = []

                    # Column structure: dataset × generation_length
                    for dataset in datasets:
                        for max_new_token in max_new_tokens:
                            # Look up the corresponding value
                            mask = (metrics_df['dataset'] == dataset) & \
                                   (metrics_df['model_scale'] == model_scale) & \
                                   (metrics_df['model_type'] == model_type) & \
                                   (metrics_df['prefix_length'] == prefix_length) & \
                                   (metrics_df['max_new_tokens'] == max_new_token)

                            if mask.sum() > 0:
                                value = metrics_df.loc[mask, metric_col].iloc[0]
                                if pd.isna(value) or value == float('inf') or value == float('-inf'):
                                    row_data.append('N/A')
                                else:
                                    # Format based on metric type
                                    if metric_col in ['target_token_rank']:
                                        row_data.append(f"{value:.1f}")
                                    else:
                                        row_data.append(f"{value:.3f}")
                            else:
                                row_data.append('N/A')

                    table_data.append(row_data)
                    row_headers.append((model_label, prefix_length))

        # Generate LaTeX table
        latex_table = generate_correct_latex_table(
            table_data,
            row_headers,
            datasets,
            max_new_tokens,
            dataset_mapping,
            metric_name
        )

        all_latex_tables.append(latex_table)
        print(f"{metric_name} 表格已生成")

    # Save all tables to a single file
    prefix_str = '_'.join(map(str, prefix_lengths))
    tokens_str = '_'.join(map(str, max_new_tokens))
    output_file = os.path.join(output_dir,
                               f'memorization_prefix{prefix_str}_newtokens{tokens_str}.tex')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Memorization Metrics LaTeX Tables\n")
        f.write("% Generated automatically\n")
        f.write(f"% Prefix lengths: {prefix_lengths}\n")
        f.write(f"% Max new tokens: {max_new_tokens}\n\n")

        for i, table in enumerate(all_latex_tables):
            f.write(table)
            if i < len(all_latex_tables) - 1:
                f.write("\n\n\\clearpage\n\n")

    print(f"\n所有latex表格已保存到: {output_file}")
    print(f"总共生成了 {len(all_latex_tables)} 个表格")

    # Also generate a summary CSV file
    summary_file = os.path.join(output_dir, 'memorization_metrics_summary.csv')
    metrics_df.to_csv(summary_file, index=False)
    print(f"指标总结CSV已保存到: {summary_file}")


def generate_correct_latex_table(table_data: List[List[str]],
                                 row_headers: List[tuple],  # (model_label, prefix_length)
                                 datasets: List[str],
                                 max_new_tokens: List[int],
                                 dataset_mapping: Dict[str, str],
                                 table_title: str) -> str:
    """
    Generate a LaTeX table with the correct structure
    Column headers: Dataset name + Generation Length
    Row headers: Model scale + type + Prefix Length
    """
    # Compute total number of columns
    total_cols = len(datasets) * len(max_new_tokens)

    # Begin table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{table_title}}}",
        f"\\begin{{tabular}}{{ll{'c' * total_cols}}}",
        "\\toprule"
    ]

    # Top-level header (dataset names)
    top_header = "Model & Prefix L"
    for dataset in datasets:
        dataset_display = dataset_mapping.get(dataset, dataset)
        num_gen_lengths = len(max_new_tokens)
        if num_gen_lengths > 1:
            top_header += f" & \\multicolumn{{{num_gen_lengths}}}{{c}}{{{dataset_display}}}"
        else:
            top_header += f" & {dataset_display}"
    top_header += " \\\\"
    latex_lines.append(top_header)

    # Second-level header (Generation Length)
    sub_header = " & "
    for dataset in datasets:
        for max_new_token in max_new_tokens:
            sub_header += f" & {max_new_token}"
    sub_header += " \\\\"
    latex_lines.append(sub_header)
    latex_lines.append("\\midrule")

    # Table rows with grouped row headers
    current_model = None
    for i, ((model_label, prefix_length), row_data) in enumerate(zip(row_headers, table_data)):
        if model_label != current_model:
            # New model group
            current_model = model_label
            row_str = f"{model_label} & {prefix_length}"
        else:
            # Different prefix within the same model
            row_str = f" & {prefix_length}"

        # Add data cells
        row_str += " & " + " & ".join(row_data) + " \\\\"
        latex_lines.append(row_str)

    # End table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:{table_title.lower().replace(' ', '_').replace('-', '_')}}}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(description='分析多个数据集和模型规模的base和sft模型生成结果')

    parser.add_argument('--results_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results',
                        help='生成结果根目录（包含exp1_generation_8和exp1_generation_16文件夹）')
    parser.add_argument('--model_scales', type=str, nargs='+',
                        default=['1B', '7B', '13B', '32B'],
                        help='要分析的模型规模列表')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['stackexchange', 'dclm-privacy', 'wiki-fact'],
                        help='要分析的数据集列表')
    parser.add_argument('--prefix_lengths', type=int, nargs='+', default=[16, 32, 64],
                        help='要分析的前缀长度列表')
    parser.add_argument('--max_new_tokens', type=int, nargs='+', default=[8, 16],
                        help='要分析的最大新token数列表')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每种条件下的最大样本数')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp1_mem_score',
                        help='分析结果输出目录')

    args = parser.parse_args()

    print("=" * 80)
    print("开始分析多个模型和数据集的生成结果")
    print("=" * 80)
    print(f"结果根目录: {args.results_dir}")
    print(f"模型规模: {args.model_scales}")
    print(f"数据集: {args.datasets}")
    print(f"前缀长度: {args.prefix_lengths}")
    print(f"最大新token数: {args.max_new_tokens}")
    print(f"最大样本数: {args.max_samples}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)

    # Load generation results
    results_dict = load_generation_results(
        args.results_dir,
        args.model_scales,
        args.datasets,
        args.prefix_lengths,
        args.max_new_tokens,
        args.max_samples
    )

    if not results_dict:
        print("\n错误: 未能加载任何生成结果")
        return

    # Compute memorization metrics
    print("\n" + "=" * 80)
    print("开始计算记忆指标")
    print("=" * 80)
    metrics_df = calculate_memorization_metrics_with_evaluator(results_dict)

    if len(metrics_df) == 0:
        print("\n错误: 无法计算记忆指标")
        return

    print(f"\n计算完成，共 {len(metrics_df)} 条记录")
    print("\n指标概览:")
    print(metrics_df[['dataset', 'model_scale', 'model_type', 'prefix_length', 'max_new_tokens',
                      'exact_match_rate', 'rouge_1_f', 'bleu_1']].head(10))

    # Generate and save LaTeX tables
    print("\n" + "=" * 80)
    print("开始生成LaTeX表格")
    print("=" * 80)
    generate_latex_tables(metrics_df, args.output_dir, args.prefix_lengths, args.max_new_tokens)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
