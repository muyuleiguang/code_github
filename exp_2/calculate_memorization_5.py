#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算模型记忆准确率（Exact Match）并生成LaTeX表格
基于generation_two_datasets_2.py保存的结果文件
修复版本：正确处理poems和idiom数据集
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import string
import glob





def normalize_text(text, lang='en'):
    """
    文本标准化，用于准确比较

    Args:
        text: 输入文本
        lang: 语言类型 ('en' 或 'zh')

    Returns:
        str: 标准化后的文本
    """
    if not text:
        return ""

    text = str(text).strip()

    if lang == 'en':
        # 英文处理：转小写，移除标点，标准化空格
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
    else:
        # 中文处理：移除空格和标点，只保留中文字符、数字和基本标点
        # 保留中文字符、数字，移除其他字符
        text = re.sub(r'[^\u4e00-\u9fff0-9]', '', text)

    return text


def extract_prediction_from_generation(generated_text, dataset_type):
    """
    从模型生成的文本中提取预测答案

    Args:
        generated_text: 模型生成的完整文本
        dataset_type: 数据集类型 ('idiom' 或 'poems')

    Returns:
        str: 提取的预测答案
    """
    if not generated_text:
        return ""

    # 清理生成文本
    text = generated_text.strip()

    if dataset_type == 'idiom':
        # 对于俚语，提取第一个词作为预测
        # 移除可能的标点符号和换行符
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words[0] if words else ""

    else:  # poems
        # 对于诗词，提取生成的诗句
        # 移除换行符，取第一行
        lines = text.split('\n')
        prediction = lines[0].strip() if lines else text.strip()

        # 进一步清理：移除可能的提示词
        prediction = re.sub(r'^(下句：|答案：|回答：)', '', prediction)
        prediction = prediction.strip()

        return prediction


def calculate_exact_match_accuracy(predicted, expected, dataset_type):
    """
    计算精确匹配准确率

    Args:
        predicted: 预测文本
        expected: 期望文本
        dataset_type: 数据集类型

    Returns:
        float: 准确率 (0 或 1)
    """
    lang = 'zh' if dataset_type == 'poems' else 'en'

    pred_norm = normalize_text(predicted, lang)
    expected_norm = normalize_text(expected, lang)

    return 1.0 if pred_norm == expected_norm or expected_norm in pred_norm else 0.0


def parse_filename(filename):
    """
    解析文件名以提取模型信息

    Args:
        filename: 文件名，例如 'generation_results_allenai_OLMo_2_0425_1B_base.json'

    Returns:
        tuple: (model_name, scale, model_type) 或 (None, None, None) 如果解析失败
    """
    if not filename.startswith('idiom_') or not filename.endswith('.json'):
        return None, None, None

    # 移除前缀和后缀
    # name_part = filename[19:-5]  # 移除 'generation_results_' 和 '.json'

    # 分割并提取信息
    parts = filename.replace('.json', '').split('_')

    if len(parts) < 2:
        return None, None, None

    # 最后一部分是模型类型
    model_type = parts[-1]
    if model_type not in ['base', 'sft']:
        return None, None, None

    # 倒数第二部分是模型规模
    scale = parts[-2]
    if scale not in ['1B', '7B', '13B', '32B']:
        return None, None, None

    # 其余部分组成模型名称
    # model_name_parts = parts[:-2]
    # model_name = '_'.join(model_name_parts)

    return scale, model_type


def load_and_process_results(results_dir, pattern):
    """
    加载并处理所有结果文件

    Args:
        results_dir: 结果目录
        pattern: 文件匹配模式

    Returns:
        dict: 按模型规模、数据集类型、模型类型组织的结果
    """
    # 查找所有结果文件
    search_pattern = os.path.join(results_dir, pattern)
    result_files = glob.glob(search_pattern)

    print(f"找到 {len(result_files)} 个结果文件")

    # 组织数据结构: {scale: {dataset_type: {model_type: [results]}}}
    organized_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in result_files:
        print(f"处理文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'results' not in data:
                print(f"  - 警告: 文件格式不正确，缺少'results'字段")
                continue

            # 从文件名提取模型信息
            filename = os.path.basename(file_path)
            scale, model_type = parse_filename(filename)

            if not all([scale, model_type]):
                print(f"  - 警告: 无法解析文件名格式: {filename}")
                continue

            print(f"  - 解析结果: 规模={scale}, 类型={model_type}")

            # 统计数据集类型
            dataset_stats = defaultdict(int)

            # 处理每个样本的结果
            for result in data['results']:
                dataset_type = result.get('dataset_type', '')
                generated_text = result.get('generated_text', '')
                target_output = result.get('target_output', '')

                # 统计数据集类型
                dataset_stats[dataset_type] += 1

                # 从生成文本中提取预测
                prediction = extract_prediction_from_generation(generated_text, dataset_type)

                # 计算准确率
                accuracy = calculate_exact_match_accuracy(prediction, target_output, dataset_type)

                # 保存处理后的结果
                processed_result = {
                    'sample_id': result.get('sample_id', 0),
                    'generated_text': generated_text,
                    'prediction': prediction,
                    'target_output': target_output,
                    'accuracy': accuracy,
                    'dataset_type': dataset_type,
                    'model_name': scale,
                    'model_type': model_type,
                    'scale': scale
                }

                organized_results[scale][dataset_type][model_type].append(processed_result)

            print(f"  - 成功处理 {len(data['results'])} 个样本")
            print(f"  - 数据集分布: {dict(dataset_stats)}")

        except Exception as e:
            print(f"  - 错误: 处理文件失败: {e}")

    return organized_results


def calculate_accuracy_statistics(organized_results):
    """
    计算准确率统计信息

    Args:
        organized_results: 组织好的结果数据

    Returns:
        dict: 准确率统计结果
    """
    stats = {}

    for scale in organized_results:
        stats[scale] = {}

        for dataset_type in organized_results[scale]:
            stats[scale][dataset_type] = {}

            for model_type in organized_results[scale][dataset_type]:
                results = organized_results[scale][dataset_type][model_type]

                if results:
                    accuracies = [r['accuracy'] for r in results]
                    stats[scale][dataset_type][model_type] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'num_samples': len(accuracies),
                        'correct_count': sum(accuracies),
                        'total_count': len(accuracies)
                    }
                else:
                    stats[scale][dataset_type][model_type] = {
                        'mean_accuracy': 0.0,
                        'std_accuracy': 0.0,
                        'num_samples': 0,
                        'correct_count': 0,
                        'total_count': 0
                    }

    return stats


def create_latex_table(stats, model_scales, output_path):
    """
    创建LaTeX表格，格式匹配提供的图片样式

    Args:
        stats: 准确率统计结果
        model_scales: 模型规模列表
        output_path: 输出文件路径
    """
    # 准备表格数据
    latex_content = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{模型记忆准确率 (Exact Match \\%)}",
        "\\label{tab:memorization_accuracy}",
        "\\begin{tabular}{|c|c|" + "c|" * len(model_scales) + "}",
        "\\hline"
    ]

    # 第一行：模型规模标题
    header_row1 = " & "
    for scale in model_scales:
        header_row1 += f" & {scale}"
    header_row1 += " \\\\"
    latex_content.append(header_row1)
    latex_content.append("\\hline")

    # PEOM数据集部分
    latex_content.append("\\multirow{2}{*}{PEOM} & Base")
    for scale in model_scales:
        if (scale in stats and
                'poems' in stats[scale] and
                'base' in stats[scale]['poems']):
            accuracy = stats[scale]['poems']['base']['mean_accuracy']
            count = stats[scale]['poems']['base']['total_count']
            if count > 0:
                latex_content[-1] += f" & {accuracy:.1%}"
            else:
                latex_content[-1] += " & --"
        else:
            latex_content[-1] += " & --"
    latex_content[-1] += " \\\\"
    latex_content.append("\\cline{2-" + str(len(model_scales) + 2) + "}")

    # PEOM SFT行
    latex_content.append(" & SFT")
    for scale in model_scales:
        if (scale in stats and
                'poems' in stats[scale] and
                'sft' in stats[scale]['poems']):
            accuracy = stats[scale]['poems']['sft']['mean_accuracy']
            count = stats[scale]['poems']['sft']['total_count']
            if count > 0:
                latex_content[-1] += f" & {accuracy:.1%}"
            else:
                latex_content[-1] += " & --"
        else:
            latex_content[-1] += " & --"
    latex_content[-1] += " \\\\"
    latex_content.append("\\hline")

    # IDIOM数据集部分
    latex_content.append("\\multirow{2}{*}{IDIOM} & Base")
    for scale in model_scales:
        if (scale in stats and
                'idiom' in stats[scale] and
                'base' in stats[scale]['idiom']):
            accuracy = stats[scale]['idiom']['base']['mean_accuracy']
            count = stats[scale]['idiom']['base']['total_count']
            if count > 0:
                latex_content[-1] += f" & {accuracy:.1%}"
            else:
                latex_content[-1] += " & --"
        else:
            latex_content[-1] += " & --"
    latex_content[-1] += " \\\\"
    latex_content.append("\\cline{2-" + str(len(model_scales) + 2) + "}")

    # IDIOM SFT行
    latex_content.append(" & SFT")
    for scale in model_scales:
        if (scale in stats and
                'idiom' in stats[scale] and
                'sft' in stats[scale]['idiom']):
            accuracy = stats[scale]['idiom']['sft']['mean_accuracy']
            count = stats[scale]['idiom']['sft']['total_count']
            if count > 0:
                latex_content[-1] += f" & {accuracy:.1%}"
            else:
                latex_content[-1] += " & --"
        else:
            latex_content[-1] += " & --"
    latex_content[-1] += " \\\\"
    latex_content.append("\\hline")

    # 表格结尾
    latex_content.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% 表格说明:",
        "% 数值表示精确匹配准确率（百分比）",
        "% PEOM: 中文唐诗数据集",
        "% IDIOM: 英语俚语数据集",
        "% Base: 基础模型",
        "% SFT: 指令微调模型",
        "% --: 表示没有相应数据"
    ])

    # 保存LaTeX文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))

    print(f"LaTeX表格已保存到: {output_path}")


def save_detailed_results(stats, organized_results, args):
    """
    保存详细结果到CSV和JSON文件

    Args:
        stats: 统计结果
        organized_results: 原始组织结果
        args: 命令行参数
    """
    # 保存CSV结果
    csv_data = []

    for scale in stats:
        for dataset_type in stats[scale]:
            for model_type in stats[scale][dataset_type]:
                stat = stats[scale][dataset_type][model_type]
                csv_data.append({
                    'Scale': scale,
                    'Dataset': dataset_type,
                    'ModelType': model_type,
                    'Accuracy': stat['mean_accuracy'],
                    'StdDev': stat['std_accuracy'],
                    'CorrectCount': stat['correct_count'],
                    'TotalCount': stat['total_count'],
                    'SampleCount': stat['num_samples']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(args.output_dir, args.csv_filename)
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"CSV结果已保存到: {csv_path}")

    # 保存详细JSON结果
    json_path = os.path.join(args.output_dir, 'detailed_memorization_results.json')

    # 转换为可序列化格式
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    save_data = {
        'args': vars(args),
        'statistics': convert_numpy(stats),
        'sample_count_summary': {
            scale: {
                dataset: {
                    model_type: len(results)
                    for model_type, results in datasets.items()
                }
                for dataset, datasets in scale_data.items()
            }
            for scale, scale_data in organized_results.items()
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存到: {json_path}")


def print_summary_statistics(stats):
    """
    打印摘要统计信息

    Args:
        stats: 统计结果
    """
    print("\n" + "=" * 80)
    print("记忆准确率摘要统计")
    print("=" * 80)

    for scale in sorted(stats.keys()):
        print(f"\n{scale} 规模模型:")

        for dataset_type in ['poems', 'idiom']:
            if dataset_type in stats[scale]:
                dataset_name = 'PEOM(诗词)' if dataset_type == 'poems' else 'IDIOM(俚语)'
                print(f"\n  {dataset_name}:")

                for model_type in ['base', 'sft']:
                    if model_type in stats[scale][dataset_type]:
                        stat = stats[scale][dataset_type][model_type]
                        if stat['total_count'] > 0:
                            print(f"    {model_type.upper()}: "
                                  f"{stat['mean_accuracy']:.1%} "
                                  f"({stat['correct_count']}/{stat['total_count']})")
                        else:
                            print(f"    {model_type.upper()}: 无数据")


def analyze_sample_predictions(organized_results, num_examples=5):
    """
    分析样本预测，帮助调试

    Args:
        organized_results: 组织好的结果数据
        num_examples: 显示的示例数量
    """
    print("\n" + "=" * 80)
    print("样本预测分析")
    print("=" * 80)

    for scale in organized_results:
        for dataset_type in organized_results[scale]:
            for model_type in organized_results[scale][dataset_type]:
                results = organized_results[scale][dataset_type][model_type]

                if not results:
                    continue

                print(f"\n{scale} {dataset_type} {model_type} - 前{num_examples}个样本:")

                for i, result in enumerate(results[:num_examples]):
                    print(f"  样本 {i + 1}:")
                    print(f"    生成文本: '{result['generated_text']}'")
                    print(f"    提取预测: '{result['prediction']}'")
                    print(f"    期望输出: '{result['target_output']}'")
                    print(f"    准确率: {result['accuracy']}")
                    print()


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='计算记忆准确率并生成LaTeX表格')

    # 输入参数
    parser.add_argument('--results_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='生成结果文件目录')
    parser.add_argument('--pattern', type=str, default='idiom_*.json',
                        help='结果文件的匹配模式')

    # 输出参数
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='输出结果保存目录')
    parser.add_argument('--latex_filename', type=str, default='idiom_table.tex',
                        help='LaTeX表格文件名')
    parser.add_argument('--csv_filename', type=str, default='idiom_results.csv',
                        help='CSV结果文件名')
    # 模型规模
    parser.add_argument('--model_scales', type=str, nargs='+', default=['1B', '7B', '13B', '32B'], help='模型规模列表')

    return parser.parse_args()


def main():
    """主函数"""
    args = setup_args()

    print("开始计算记忆准确率...")
    print(f"输入目录: {args.results_dir}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载和处理结果
    print("\n加载结果文件...")
    organized_results = load_and_process_results(args.results_dir, args.pattern)

    if not organized_results:
        print("错误: 没有找到可处理的结果文件")
        return

    # 分析样本预测（调试用）
    analyze_sample_predictions(organized_results, num_examples=3)

    # 计算准确率统计
    print("\n计算准确率统计...")
    stats = calculate_accuracy_statistics(organized_results)

    # 打印摘要统计
    print_summary_statistics(stats)

    # 创建LaTeX表格
    print("\n生成LaTeX表格...")
    latex_path = os.path.join(args.output_dir, args.latex_filename)
    create_latex_table(stats, args.model_scales, latex_path)

    # 保存详细结果
    print("\n保存详细结果...")
    save_detailed_results(stats, organized_results, args)

    print(f"\n分析完成！所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()