#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute model memorization accuracy (Exact Match) and generate a LaTeX table
Based on the result files saved by generation_two_datasets_2.py
Fixed version: correctly handles the poems and idiom datasets
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
    Text normalization for exact comparison

    Args:
        text: Input text
        lang: Language type ('en' or 'zh')

    Returns:
        str: Normalized text
    """
    if not text:
        return ""

    text = str(text).strip()

    if lang == 'en':
        # English processing: lowercase, remove punctuation, normalize spaces
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
    else:
        # Chinese processing: remove spaces and punctuation; keep only Chinese chars, digits, and basic punctuation
        # Keep Chinese characters and digits, remove other characters
        text = re.sub(r'[^\u4e00-\u9fff0-9]', '', text)

    return text


def extract_prediction_from_generation(generated_text, dataset_type):
    """
    Extract the predicted answer from the model-generated text

    Args:
        generated_text: Full generated text from the model
        dataset_type: Dataset type ('idiom' or 'poems')

    Returns:
        str: Extracted prediction
    """
    if not generated_text:
        return ""

    # Clean generated text
    text = generated_text.strip()

    if dataset_type == 'idiom':
        # For idioms, take the first word as the prediction
        # Remove possible punctuation and newlines
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words[0] if words else ""

    else:  # poems
        # For poems, extract the generated poetic line
        # Remove newlines and take the first line
        lines = text.split('\n')
        prediction = lines[0].strip() if lines else text.strip()

        # Further cleanup: remove possible prompt prefixes
        prediction = re.sub(r'^(下句：|答案：|回答：)', '', prediction)
        prediction = prediction.strip()

        return prediction


def calculate_exact_match_accuracy(predicted, expected, dataset_type):
    """
    Compute exact match accuracy

    Args:
        predicted: Predicted text
        expected: Expected text
        dataset_type: Dataset type

    Returns:
        float: Accuracy (0 or 1)
    """
    lang = 'zh' if dataset_type == 'poems' else 'en'

    pred_norm = normalize_text(predicted, lang)
    expected_norm = normalize_text(expected, lang)

    return 1.0 if pred_norm == expected_norm or expected_norm in pred_norm else 0.0


def parse_filename(filename):
    """
    Parse filename to extract model information

    Args:
        filename: Filename, e.g., 'generation_results_allenai_OLMo_2_0425_1B_base.json'

    Returns:
        tuple: (model_name, scale, model_type) or (None, None, None) if parsing fails
    """
    if not filename.startswith('idiom_') or not filename.endswith('.json'):
        return None, None, None

    # Remove prefix and suffix
    # name_part = filename[19:-5]  # Remove 'generation_results_' and '.json'

    # Split and extract information
    parts = filename.replace('.json', '').split('_')

    if len(parts) < 2:
        return None, None, None

    # The last part is model type
    model_type = parts[-1]
    if model_type not in ['base', 'sft']:
        return None, None, None

    # The second-to-last part is model scale
    scale = parts[-2]
    if scale not in ['1B', '7B', '13B', '32B']:
        return None, None, None

    # The remaining parts form the model name
    # model_name_parts = parts[:-2]
    # model_name = '_'.join(model_name_parts)

    return scale, model_type


def load_and_process_results(results_dir, pattern):
    """
    Load and process all result files

    Args:
        results_dir: Results directory
        pattern: File matching pattern

    Returns:
        dict: Results organized by model scale, dataset type, and model type
    """
    # Find all result files
    search_pattern = os.path.join(results_dir, pattern)
    result_files = glob.glob(search_pattern)

    print(f"找到 {len(result_files)} 个结果文件")

    # Organize data structure: {scale: {dataset_type: {model_type: [results]}}}
    organized_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_path in result_files:
        print(f"处理文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'results' not in data:
                print(f"  - 警告: 文件格式不正确，缺少'results'字段")
                continue

            # Extract model info from filename
            filename = os.path.basename(file_path)
            scale, model_type = parse_filename(filename)

            if not all([scale, model_type]):
                print(f"  - 警告: 无法解析文件名格式: {filename}")
                continue

            print(f"  - 解析结果: 规模={scale}, 类型={model_type}")

            # Dataset type statistics
            dataset_stats = defaultdict(int)

            # Process each sample result
            for result in data['results']:
                dataset_type = result.get('dataset_type', '')
                generated_text = result.get('generated_text', '')
                target_output = result.get('target_output', '')

                # Count dataset types
                dataset_stats[dataset_type] += 1

                # Extract prediction from generated text
                prediction = extract_prediction_from_generation(generated_text, dataset_type)

                # Compute accuracy
                accuracy = calculate_exact_match_accuracy(prediction, target_output, dataset_type)

                # Save processed result
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
    Compute accuracy statistics

    Args:
        organized_results: Organized result data

    Returns:
        dict: Accuracy statistics
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
    Create a LaTeX table matching the provided screenshot style

    Args:
        stats: Accuracy statistics
        model_scales: List of model scales
        output_path: Output file path
    """
    # Prepare table content
    latex_content = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{模型记忆准确率 (Exact Match \\%)}",
        "\\label{tab:memorization_accuracy}",
        "\\begin{tabular}{|c|c|" + "c|" * len(model_scales) + "}",
        "\\hline"
    ]

    # First row: model scale header
    header_row1 = " & "
    for scale in model_scales:
        header_row1 += f" & {scale}"
    header_row1 += " \\\\"
    latex_content.append(header_row1)
    latex_content.append("\\hline")

    # PEOM dataset section
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

    # PEOM SFT row
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

    # IDIOM dataset section
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

    # IDIOM SFT row
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

    # Table ending
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

    # Save LaTeX file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))

    print(f"LaTeX表格已保存到: {output_path}")


def save_detailed_results(stats, organized_results, args):
    """
    Save detailed results to CSV and JSON files

    Args:
        stats: Statistics results
        organized_results: Raw organized results
        args: Command-line arguments
    """
    # Save CSV results
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

    # Save detailed JSON results
    json_path = os.path.join(args.output_dir, 'detailed_memorization_results.json')

    # Convert to a JSON-serializable format
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
    Print summary statistics

    Args:
        stats: Statistics results
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
    Analyze sample predictions for debugging

    Args:
        organized_results: Organized result data
        num_examples: Number of examples to display
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
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description='计算记忆准确率并生成LaTeX表格')

    # Input arguments
    parser.add_argument('--results_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='生成结果文件目录')
    parser.add_argument('--pattern', type=str, default='idiom_*.json',
                        help='结果文件的匹配模式')

    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/ift_memorization/results/exp2_memorization_poem',
                        help='输出结果保存目录')
    parser.add_argument('--latex_filename', type=str, default='idiom_table.tex',
                        help='LaTeX表格文件名')
    parser.add_argument('--csv_filename', type=str, default='idiom_results.csv',
                        help='CSV结果文件名')
    # Model scales
    parser.add_argument('--model_scales', type=str, nargs='+', default=['1B', '7B', '13B', '32B'], help='模型规模列表')

    return parser.parse_args()


def main():
    """Main function"""
    args = setup_args()

    print("开始计算记忆准确率...")
    print(f"输入目录: {args.results_dir}")
    print(f"输出目录: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process results
    print("\n加载结果文件...")
    organized_results = load_and_process_results(args.results_dir, args.pattern)

    if not organized_results:
        print("错误: 没有找到可处理的结果文件")
        return

    # Analyze sample predictions (for debugging)
    analyze_sample_predictions(organized_results, num_examples=3)

    # Compute accuracy statistics
    print("\n计算准确率统计...")
    stats = calculate_accuracy_statistics(organized_results)

    # Print summary statistics
    print_summary_statistics(stats)

    # Create LaTeX table
    print("\n生成LaTeX表格...")
    latex_path = os.path.join(args.output_dir, args.latex_filename)
    create_latex_table(stats, args.model_scales, latex_path)

    # Save detailed results
    print("\n保存详细结果...")
    save_detailed_results(stats, organized_results, args)

    print(f"\n分析完成！所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
