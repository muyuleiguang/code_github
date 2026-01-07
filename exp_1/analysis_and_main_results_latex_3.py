#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import glob
import pandas as pd
from typing import Dict, List
import warnings
# 导入记忆评估指标
from memorization_metrics import MemorizationMetrics

warnings.filterwarnings('ignore')


def sort_model_scales(model_scales):
    """
    按照模型规模数值大小排序，如 1B < 7B < 13B

    Args:
        model_scales: 模型规模列表

    Returns:
        sorted_scales: 排序后的模型规模列表
    """

    def extract_scale_value(scale_str):
        """提取模型规模的数值部分用于排序"""
        try:
            # 移除末尾的单位（B、M等）
            if scale_str.endswith('B'):
                return float(scale_str[:-1])
            elif scale_str.endswith('M'):
                return float(scale_str[:-1]) / 1000  # 转换为B单位
            else:
                # 如果没有单位，直接当作数字处理
                return float(scale_str)
        except:
            # 如果解析失败，返回一个大数值，排在最后
            return float('inf')

    return sorted(model_scales, key=extract_scale_value)


def load_generation_results(results_dir: str,
                            model_scales: List[str],
                            datasets: List[str],
                            prefix_lengths: List[int],
                            max_new_tokens: List[int],
                            max_samples: int = None) -> Dict[str, Dict]:
    """
    加载多个数据集和模型规模的生成结果文件

    文件命名格式: {dataset}_prefix{prefix_length}_new{max_new_tokens}_{model_scale}_{model_type}_10000_samples.jsonl
    文件位置: results_dir/exp1_generation_{max_new_tokens}/

    Args:
        results_dir: 结果根目录 (例如: /root/autodl-tmp/ift_memorization/results)
        model_scales: 模型规模列表 (如 ["1B", "7B", "13B", "32B"])
        datasets: 数据集列表 (如 ['stackexchange', 'dclm-privacy', 'wiki-fact'])
        prefix_lengths: 前缀长度列表 (如 [16, 32, 64])
        max_new_tokens: 最大新token数列表 (如 [8, 16])
        max_samples: 每个条件下的最大样本数

    Returns:
        results_dict: 按数据集、模型规模、模型类型、前缀长度和max_new_tokens组织的结果
        格式: {dataset: {model_scale: {model_type: {prefix_length: {max_new_tokens: [samples]}}}}}
    """
    results_dict = {}

    print("\n开始加载文件...")
    print(f"数据集: {datasets}")
    print(f"模型规模: {model_scales}")
    print(f"前缀长度: {prefix_lengths}")
    print(f"最大新token数: {max_new_tokens}")
    print("-" * 80)

    # 遍历所有参数组合
    for dataset in datasets:
        for model_scale in model_scales:
            for model_type in ['base', 'sft']:
                for prefix_length in prefix_lengths:
                    for max_new_token in max_new_tokens:
                        # 构建文件路径
                        # 文件在 exp1_generation_{max_new_token} 文件夹下
                        folder_name = f"exp1_generation_{max_new_token}"
                        file_name = f"{dataset}_prefix{prefix_length}_new{max_new_token}_{model_scale}_{model_type}_{max_samples}_samples.jsonl"
                        file_path = os.path.join(results_dir, folder_name, file_name)

                        # 检查文件是否存在
                        if not os.path.exists(file_path):
                            print(f"✗ 文件不存在: {file_name}")
                            continue

                        # 读取文件
                        try:
                            samples = []
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        sample = json.loads(line)
                                        samples.append(sample)

                            # 应用样本数限制
                            if max_samples and len(samples) > max_samples:
                                samples = samples[:max_samples]

                            # 组织数据结构
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
    使用MemorizationMetrics计算记忆指标

    Args:
        results_dict: 生成结果字典

    Returns:
        metrics_df: 记忆指标的DataFrame
    """
    # 初始化评估器
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

                        # 使用新的统一接口计算所有指标
                        try:
                            metrics_results = evaluator.compute_all_metrics_from_data(samples)

                            # 提取各种指标数据
                            metrics_entry = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'model_scale': model_scale,
                                'prefix_length': prefix_length,
                                'max_new_tokens': max_new_tokens,
                                'sample_count': len(samples),
                            }

                            # 第一种：精确匹配
                            if 'exact_match' in metrics_results:
                                metrics_entry['exact_match_rate'] = metrics_results['exact_match']['exact_match_rate']

                            # 第二种：ROUGE/BLEU指标
                            if 'rouge_bleu' in metrics_results:
                                rouge_bleu = metrics_results['rouge_bleu']
                                metrics_entry['rouge_1_f'] = rouge_bleu.get('rouge_1_f', 0.0)
                                metrics_entry['rouge_2_f'] = rouge_bleu.get('rouge_2_f', 0.0)
                                metrics_entry['rouge_l_f'] = rouge_bleu.get('rouge_l_f', 0.0)
                                metrics_entry['bleu_1'] = rouge_bleu.get('bleu_1', 0.0)
                                metrics_entry['bleu_2'] = rouge_bleu.get('bleu_2', 0.0)
                                metrics_entry['bleu_4'] = rouge_bleu.get('bleu_4', 0.0)

                            # 第三种：编辑距离
                            if 'edit_distance' in metrics_results:
                                edit_dist = metrics_results['edit_distance']
                                metrics_entry['token_edit_distance'] = edit_dist.get('token_edit_distance', 0.0)
                                metrics_entry['normalized_token_distance'] = edit_dist.get('normalized_token_distance',
                                                                                           0.0)

                            # 第四种：语义相似度
                            if 'semantic' in metrics_results:
                                semantic = metrics_results['semantic']
                                metrics_entry['semantic_similarity'] = semantic.get('cosine_similarity', 0.0)

                            # 第五种：likelihood相关指标
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
    生成latex表格，按照正确的结构：
    列头：数据集名称 + Generation Length
    行头：模型规模+类型 + Prefix Length

    Args:
        metrics_df: 包含指标的DataFrame
        output_dir: 输出目录
        prefix_lengths: 前缀长度列表
        max_new_tokens: 最大新token数列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据集映射（显示用）
    dataset_mapping = {
        'stackexchange': 'STACKEXCHANGE',
        'dclm-privacy': 'DCLM-PRIVACY',
        'wiki-fact': 'WIKI-FACT'
    }

    # 定义要生成表格的指标
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

    # 收集所有latex表格
    all_latex_tables = []

    # 获取唯一的数据集和模型
    datasets = sorted(metrics_df['dataset'].unique())
    model_scales = sort_model_scales(metrics_df['model_scale'].unique())

    # 为每个指标生成一个大表格
    for metric_col, metric_name in metrics_to_generate:
        if metric_col not in metrics_df.columns:
            continue

        print(f"生成 {metric_name} 的大表格...")

        # 构建表格数据 - 按照正确的行列结构
        table_data = []
        row_headers = []

        # 行结构：模型规模+类型 × prefix_length
        for model_scale in model_scales:
            for model_type in ['base', 'sft']:
                model_label = f"{model_scale}" if model_type == 'base' else f"{model_scale}-sft"

                for prefix_length in prefix_lengths:
                    row_data = []

                    # 列结构：数据集 × generation_length
                    for dataset in datasets:
                        for max_new_token in max_new_tokens:
                            # 查找对应的值
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
                                    # 根据指标类型决定格式
                                    if metric_col in ['target_token_rank']:
                                        row_data.append(f"{value:.1f}")
                                    else:
                                        row_data.append(f"{value:.3f}")
                            else:
                                row_data.append('N/A')

                    table_data.append(row_data)
                    row_headers.append((model_label, prefix_length))

        # 生成latex表格
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

    # 保存所有表格到一个文件
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

    # 也生成一个总结的CSV文件
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
    生成正确结构的latex表格
    列头：数据集名称 + Generation Length
    行头：模型规模+类型 + Prefix Length
    """
    # 计算总列数
    total_cols = len(datasets) * len(max_new_tokens)

    # 开始表格
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{table_title}}}",
        f"\\begin{{tabular}}{{ll{'c' * total_cols}}}",
        "\\toprule"
    ]

    # 第一层表头（数据集名称）
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

    # 第二层表头（Generation Length）
    sub_header = " & "
    for dataset in datasets:
        for max_new_token in max_new_tokens:
            sub_header += f" & {max_new_token}"
    sub_header += " \\\\"
    latex_lines.append(sub_header)
    latex_lines.append("\\midrule")

    # 表格数据 - 带有行头分组
    current_model = None
    for i, ((model_label, prefix_length), row_data) in enumerate(zip(row_headers, table_data)):
        if model_label != current_model:
            # 新的模型组
            current_model = model_label
            row_str = f"{model_label} & {prefix_length}"
        else:
            # 同一模型的不同prefix
            row_str = f" & {prefix_length}"

        # 添加数据
        row_str += " & " + " & ".join(row_data) + " \\\\"
        latex_lines.append(row_str)

    # 结束表格
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

    # 加载生成结果
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

    # 计算记忆指标
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

    # 生成并保存latex表格
    print("\n" + "=" * 80)
    print("开始生成LaTeX表格")
    print("=" * 80)
    generate_latex_tables(metrics_df, args.output_dir, args.prefix_lengths, args.max_new_tokens)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()