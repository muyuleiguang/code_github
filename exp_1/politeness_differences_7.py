#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data analysis and visualization tool for the impact of instruction fine-tuning on model memorization
Used to analyze characteristic differences between base and SFT models under different model scales,
dataset types, and generation lengths
"""

import json
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys


class MemorizationAnalyzer:
    """Main memorization analysis class for processing and analyzing the impact of instruction fine-tuning on model memorization"""

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize analyzer.

        Args:
            data_dir: Directory path containing all JSON result files
            output_dir: Output directory path for saving processed results and tables
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define experimental configuration parameters
        self.datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']  # three datasets
        self.model_scales = ['1B', '7B', '13B', '32B']  # four model scales
        self.generation_lengths = [8, 16, 128]  # three generation lengths
        self.model_types = ['base', 'sft']  # two model types: base model and instruction fine-tuned model

        # Define feature metrics to analyze
        self.features = [
            'politeness_ratio',  # politeness ratio
            'structured_ratio',  # structured ratio
            'question_ratio',  # question ratio
            'avg_certainty_density',  # average certainty density
            'avg_uncertainty_density',  # average uncertainty density
            'avg_transition_density'  # average transition density
        ]

        # Feature name mapping for more user-friendly table titles
        self.feature_names = {
            'politeness_ratio': 'Politeness',
            'structured_ratio': 'Structure',
            'question_ratio': 'Question',
            'avg_certainty_density': 'Certainty',
            'avg_uncertainty_density': 'Uncertainty',
            'avg_transition_density': 'Transition'
        }

    def load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Load a JSON file and return the parsed data.

        Args:
            filepath: Path to the JSON file

        Returns:
            Parsed JSON dict; returns an empty dict if the file does not exist or parsing fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载文件 {filepath}: {e}")
            return {}

    def extract_features_from_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract required feature values from the JSON data.

        Args:
            data: Raw data loaded from the JSON file

        Returns:
            Dict of extracted features; keys are feature names and values are numeric values
        """
        features = {}

        try:
            # Extract politeness analysis results
            if 'politeness_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['politeness_analysis']:
                        features[f'{model_type}_politeness_ratio'] = data['politeness_analysis'][model_type].get(
                            'politeness_ratio', 0.0)

            # Extract structure analysis results
            if 'structure_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['structure_analysis']:
                        features[f'{model_type}_structured_ratio'] = data['structure_analysis'][model_type].get(
                            'structured_ratio', 0.0)

            # Extract question analysis results
            if 'question_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['question_analysis']:
                        features[f'{model_type}_question_ratio'] = data['question_analysis'][model_type].get(
                            'question_ratio', 0.0)

            # Extract certainty analysis results
            if 'certainty_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['certainty_analysis']:
                        certainty_data = data['certainty_analysis'][model_type]
                        features[f'{model_type}_avg_certainty_density'] = certainty_data.get('avg_certainty_density',
                                                                                             0.0)
                        features[f'{model_type}_avg_uncertainty_density'] = certainty_data.get(
                            'avg_uncertainty_density', 0.0)
                        features[f'{model_type}_avg_transition_density'] = certainty_data.get('avg_transition_density',
                                                                                              0.0)

        except Exception as e:
            print(f"警告: 提取特征时出错: {e}")

        return features

    def collect_all_data(self) -> pd.DataFrame:
        """
        Collect all experimental data and organize it into a DataFrame.

        Returns:
            Full data table containing all experimental conditions and corresponding feature values
        """
        all_data = []

        # Iterate over all possible experimental configuration combinations
        for dataset in self.datasets:
            for scale in self.model_scales:
                for length in self.generation_lengths:
                    # Filename pattern: exp1_differences_{dataset}_{scale}_length_{length}.json
                    filename = f"exp1_differences_{dataset}_{scale}_length_{length}.json"
                    filepath = self.data_dir / filename

                    if filepath.exists():
                        print(f"正在处理文件: {filename}")

                        # Load data and extract features
                        data = self.load_json_file(filepath)
                        features = self.extract_features_from_data(data)

                        if features:  # if features were successfully extracted
                            # Create one row per model type
                            for model_type in self.model_types:
                                row_data = {
                                    'Dataset': dataset,
                                    'Model_Scale': scale,
                                    'Generation_Length': length,
                                    'Model_Type': model_type
                                }

                                # Add all feature values for this model type
                                for feature in self.features:
                                    feature_key = f"{model_type}_{feature}"
                                    row_data[feature] = features.get(feature_key, 0.0)

                                all_data.append(row_data)
                        else:
                            print(f"警告: 文件 {filename} 中未提取到有效特征")
                    else:
                        print(f"警告: 文件不存在: {filename}")

        # Convert to DataFrame and return
        df = pd.DataFrame(all_data)
        print(f"\n成功收集 {len(df)} 行数据")
        return df

    def create_latex_table(self, df: pd.DataFrame, save_path: str) -> str:
        """
        Create a LaTeX table following the style of the attached Figure 2.

        Args:
            df: DataFrame containing all data
            save_path: Path to save the table

        Returns:
            Generated LaTeX table as a string
        """
        # Table structure: model scale x generation length x feature x model type
        latex_lines = []

        # Begin table
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{指令微调对模型记忆特征的影响分析}")
        latex_lines.append("\\label{tab:memorization_analysis}")

        # Column specification: model scale | generation length | 6 features (base/sft two columns each)
        col_spec = "l|l|" + "cc|" * len(self.features)
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # Build header
        header_line1 = "Model & L"
        header_line2 = " & "

        for feature in self.features:
            display_name = self.feature_names.get(feature, feature)
            header_line1 += f" & \\multicolumn{{2}}{{c|}}{{{display_name}}}"
            header_line2 += f" & base & sft"

        header_line1 += " \\\\"
        header_line2 += " \\\\"

        latex_lines.append(header_line1)
        latex_lines.append(header_line2)
        latex_lines.append("\\hline")

        # Create one section per dataset
        for dataset in self.datasets:
            latex_lines.append(f"\\multicolumn{{{2 + len(self.features) * 2}}}{{c|}}{{{dataset.upper()}}} \\\\")
            latex_lines.append("\\hline")

            # Subset data for this dataset
            dataset_df = df[df['Dataset'] == dataset]

            # Group by model scale and generation length
            for scale in self.model_scales:
                scale_df = dataset_df[dataset_df['Model_Scale'] == scale]
                if scale_df.empty:
                    continue

                first_row_for_scale = True
                for length in self.generation_lengths:
                    length_df = scale_df[scale_df['Generation_Length'] == length]
                    if length_df.empty:
                        continue

                    # Build data row
                    if first_row_for_scale:
                        row = f"{scale} & {length}"
                        first_row_for_scale = False
                    else:
                        row = f" & {length}"

                    # Add base and sft values for each feature
                    for feature in self.features:
                        base_df = length_df[length_df['Model_Type'] == 'base']
                        sft_df = length_df[length_df['Model_Type'] == 'sft']

                        base_val = base_df[feature].iloc[0] if not base_df.empty else 0.0
                        sft_val = sft_df[feature].iloc[0] if not sft_df.empty else 0.0

                        # Format numeric values (3 decimal places)
                        row += f" & {base_val:.3f} & {sft_val:.3f}"

                    row += " \\\\"
                    latex_lines.append(row)

            latex_lines.append("\\hline")

        # End table
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        latex_content = "\n".join(latex_lines)

        # Save LaTeX file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        return latex_content

    def print_data_summary(self, df: pd.DataFrame):
        """
        Print a data summary for validating completeness of the collected data.

        Args:
            df: DataFrame containing all data
        """
        print("\n" + "=" * 80)
        print("数据收集摘要")
        print("=" * 80)

        print(f"总数据行数: {len(df)}")
        print(f"数据集类型: {sorted(df['Dataset'].unique())}")
        print(f"模型规模: {sorted(df['Model_Scale'].unique())}")
        print(f"生成长度: {sorted(df['Generation_Length'].unique())}")
        print(f"模型类型: {sorted(df['Model_Type'].unique())}")

        print("\n各数据集的数据分布:")
        for dataset in self.datasets:
            count = len(df[df['Dataset'] == dataset])
            print(f"  {dataset}: {count} 行")

        print("\n各模型规模的数据分布:")
        for scale in self.model_scales:
            count = len(df[df['Model_Scale'] == scale])
            print(f"  {scale}: {count} 行")

        print("\n特征统计信息:")
        for feature in self.features:
            feature_data = df[feature]
            print(f"  {self.feature_names.get(feature, feature)}:")
            print(f"    均值: {feature_data.mean():.4f}")
            print(f"    标准差: {feature_data.std():.4f}")
            print(f"    范围: [{feature_data.min():.4f}, {feature_data.max():.4f}]")

    def run_analysis(self):
        """
        Run the full analysis pipeline:
        data collection, processing, table generation, and summary output.
        """
        print("开始指令微调记忆影响分析...")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")

        # Step 1: Collect all data
        print("\n步骤1: 收集实验数据...")
        df = self.collect_all_data()

        if df.empty:
            print("错误: 未收集到任何有效数据，请检查数据目录和文件格式")
            return

        # Step 2: Save raw data to CSV
        csv_path = self.output_dir / "memorization_analysis_data.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n原始数据已保存到: {csv_path}")

        # Step 3: Generate LaTeX table
        print("\n步骤2: 生成LaTeX表格...")
        latex_path = self.output_dir / "memorization_analysis_table.tex"
        latex_content = self.create_latex_table(df, latex_path)
        print(f"LaTeX表格已保存到: {latex_path}")

        # Step 4: Print data summary
        self.print_data_summary(df)

        # Step 5: Display LaTeX table content
        print("\n" + "=" * 80)
        print("生成的LaTeX表格内容:")
        print("=" * 80)
        print(latex_content)

        print(f"\n分析完成！所有结果已保存到: {self.output_dir}")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description='指令微调对模型记忆影响的数据分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python memorization_analysis.py --data_dir /path/to/results --output_dir /path/to/output
  python memorization_analysis.py  # use default paths
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/root/autodl-tmp/ift_memorization/results/exp1_differences',
        help='包含JSON结果文件的数据目录路径 (默认: /root/autodl-tmp/ift_memorization/results/exp1_differences)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/autodl-tmp/ift_memorization/results/exp1_differences',
        help='输出目录路径，用于保存分析结果 (默认: ./memorization_analysis_output)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['stackexchange', 'dclm-privacy', 'wiki-fact'],
        help='要分析的数据集列表 (默认: stackexchange dclm-privacy wiki-fact)'
    )

    parser.add_argument(
        '--model_scales',
        nargs='+',
        default=['1B', '7B', '13B', '32B'],
        help='要分析的模型规模列表 (默认: 1B 7B 13B 32B)'
    )

    parser.add_argument(
        '--generation_lengths',
        nargs='+',
        type=int,
        default=[8, 16, 128],
        help='要分析的生成长度列表 (默认: 8 16 128)'
    )

    return parser.parse_args()


def main():
    """Main function / program entry point."""
    # Parse command-line arguments
    args = parse_arguments()

    # Validate that the input directory exists
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # Create analyzer instance
    analyzer = MemorizationAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    # If the user specified custom parameters, update analyzer configuration
    if hasattr(args, 'datasets'):
        analyzer.datasets = args.datasets
    if hasattr(args, 'model_scales'):
        analyzer.model_scales = args.model_scales
    if hasattr(args, 'generation_lengths'):
        analyzer.generation_lengths = args.generation_lengths

    # Run analysis
    try:
        analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\n\n用户中断了分析过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
