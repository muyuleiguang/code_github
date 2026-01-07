#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指令微调对模型记忆影响的数据分析和可视化工具
用于分析不同模型规模、数据集类型和生成长度下base模型和SFT模型的特征差异
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
    """记忆分析主类，用于处理和分析指令微调对模型记忆的影响"""

    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化分析器

        Args:
            data_dir: 数据文件目录路径，包含所有JSON结果文件
            output_dir: 输出目录路径，用于保存处理结果和表格
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 定义实验配置参数
        self.datasets = ['stackexchange', 'dclm-privacy', 'wiki-fact']  # 三种数据集
        self.model_scales = ['1B', '7B', '13B', '32B']  # 四种模型规模
        self.generation_lengths = [8, 16, 128]  # 三种生成长度
        self.model_types = ['base', 'sft']  # 两种模型类型：基础模型和指令微调模型

        # 定义需要分析的特征指标
        self.features = [
            'politeness_ratio',  # 礼貌程度比率
            'structured_ratio',  # 结构化程度比率
            'question_ratio',  # 问句比率
            'avg_certainty_density',  # 平均确定性密度
            'avg_uncertainty_density',  # 平均不确定性密度
            'avg_transition_density'  # 平均转换密度
        ]

        # 特征名称映射，用于生成更友好的表格标题
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
        加载JSON文件并返回数据

        Args:
            filepath: JSON文件路径

        Returns:
            解析后的JSON数据字典，如果文件不存在或解析失败则返回空字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载文件 {filepath}: {e}")
            return {}

    def extract_features_from_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        从JSON数据中提取所需的特征值

        Args:
            data: 从JSON文件加载的原始数据

        Returns:
            包含所有特征值的字典，键为特征名，值为对应的数值
        """
        features = {}

        try:
            # 提取礼貌程度分析结果
            if 'politeness_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['politeness_analysis']:
                        features[f'{model_type}_politeness_ratio'] = data['politeness_analysis'][model_type].get(
                            'politeness_ratio', 0.0)

            # 提取结构化分析结果
            if 'structure_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['structure_analysis']:
                        features[f'{model_type}_structured_ratio'] = data['structure_analysis'][model_type].get(
                            'structured_ratio', 0.0)

            # 提取问句分析结果
            if 'question_analysis' in data:
                for model_type in self.model_types:
                    if model_type in data['question_analysis']:
                        features[f'{model_type}_question_ratio'] = data['question_analysis'][model_type].get(
                            'question_ratio', 0.0)

            # 提取确定性分析结果
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
        收集所有实验数据并整理成DataFrame格式

        Returns:
            包含所有实验条件和对应特征值的完整数据表
        """
        all_data = []

        # 遍历所有可能的实验配置组合
        for dataset in self.datasets:
            for scale in self.model_scales:
                for length in self.generation_lengths:
                    # 构造文件名模式：exp1_differences_{dataset}_{scale}_length_{length}.json
                    filename = f"exp1_differences_{dataset}_{scale}_length_{length}.json"
                    filepath = self.data_dir / filename

                    if filepath.exists():
                        print(f"正在处理文件: {filename}")

                        # 加载数据并提取特征
                        data = self.load_json_file(filepath)
                        features = self.extract_features_from_data(data)

                        if features:  # 如果成功提取到特征
                            # 为每个模型类型创建一行数据
                            for model_type in self.model_types:
                                row_data = {
                                    'Dataset': dataset,
                                    'Model_Scale': scale,
                                    'Generation_Length': length,
                                    'Model_Type': model_type
                                }

                                # 添加该模型类型的所有特征值
                                for feature in self.features:
                                    feature_key = f"{model_type}_{feature}"
                                    row_data[feature] = features.get(feature_key, 0.0)

                                all_data.append(row_data)
                        else:
                            print(f"警告: 文件 {filename} 中未提取到有效特征")
                    else:
                        print(f"警告: 文件不存在: {filename}")

        # 转换为DataFrame并返回
        df = pd.DataFrame(all_data)
        print(f"\n成功收集 {len(df)} 行数据")
        return df

    def create_latex_table(self, df: pd.DataFrame, save_path: str) -> str:
        """
        根据附件图2的样式创建LaTeX表格

        Args:
            df: 包含所有数据的DataFrame
            save_path: 表格保存路径

        Returns:
            生成的LaTeX表格字符串
        """
        # 创建表格结构：模型规模 x 生成长度 x 特征 x 模型类型
        latex_lines = []

        # 表格开始
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{指令微调对模型记忆特征的影响分析}")
        latex_lines.append("\\label{tab:memorization_analysis}")

        # 表格列定义：模型规模 | 生成长度 | 6个特征(base/sft各一列)
        col_spec = "l|l|" + "cc|" * len(self.features)
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")

        # 创建表头
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

        # 为每个数据集创建一个表格部分
        for dataset in self.datasets:
            latex_lines.append(f"\\multicolumn{{{2 + len(self.features) * 2}}}{{c|}}{{{dataset.upper()}}} \\\\")
            latex_lines.append("\\hline")

            # 获取该数据集的数据
            dataset_df = df[df['Dataset'] == dataset]

            # 按模型规模和生成长度分组
            for scale in self.model_scales:
                scale_df = dataset_df[dataset_df['Model_Scale'] == scale]
                if scale_df.empty:
                    continue

                first_row_for_scale = True
                for length in self.generation_lengths:
                    length_df = scale_df[scale_df['Generation_Length'] == length]
                    if length_df.empty:
                        continue

                    # 构建数据行
                    if first_row_for_scale:
                        row = f"{scale} & {length}"
                        first_row_for_scale = False
                    else:
                        row = f" & {length}"

                    # 为每个特征添加base和sft的值
                    for feature in self.features:
                        base_df = length_df[length_df['Model_Type'] == 'base']
                        sft_df = length_df[length_df['Model_Type'] == 'sft']

                        base_val = base_df[feature].iloc[0] if not base_df.empty else 0.0
                        sft_val = sft_df[feature].iloc[0] if not sft_df.empty else 0.0

                        # 格式化数值显示（保留3位小数）
                        row += f" & {base_val:.3f} & {sft_val:.3f}"

                    row += " \\\\"
                    latex_lines.append(row)

            latex_lines.append("\\hline")

        # 表格结束
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        latex_content = "\n".join(latex_lines)

        # 保存LaTeX文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        return latex_content

    def print_data_summary(self, df: pd.DataFrame):
        """
        打印数据摘要信息，用于验证数据收集的完整性

        Args:
            df: 包含所有数据的DataFrame
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
        执行完整的数据分析流程
        包括数据收集、处理、表格生成和摘要输出
        """
        print("开始指令微调记忆影响分析...")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")

        # 步骤1: 收集所有数据
        print("\n步骤1: 收集实验数据...")
        df = self.collect_all_data()

        if df.empty:
            print("错误: 未收集到任何有效数据，请检查数据目录和文件格式")
            return

        # 步骤2: 保存原始数据CSV文件
        csv_path = self.output_dir / "memorization_analysis_data.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n原始数据已保存到: {csv_path}")

        # 步骤3: 生成LaTeX表格
        print("\n步骤2: 生成LaTeX表格...")
        latex_path = self.output_dir / "memorization_analysis_table.tex"
        latex_content = self.create_latex_table(df, latex_path)
        print(f"LaTeX表格已保存到: {latex_path}")

        # 步骤4: 打印数据摘要
        self.print_data_summary(df)

        # 步骤5: 显示LaTeX表格内容
        print("\n" + "=" * 80)
        print("生成的LaTeX表格内容:")
        print("=" * 80)
        print(latex_content)

        print(f"\n分析完成！所有结果已保存到: {self.output_dir}")


def parse_arguments():
    """
    解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='指令微调对模型记忆影响的数据分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python memorization_analysis.py --data_dir /path/to/results --output_dir /path/to/output
  python memorization_analysis.py  # 使用默认路径
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
    """主函数，程序入口点"""
    # 解析命令行参数
    args = parse_arguments()

    # 验证输入目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)

    # 创建分析器实例
    analyzer = MemorizationAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    # 如果用户指定了自定义参数，更新分析器配置
    if hasattr(args, 'datasets'):
        analyzer.datasets = args.datasets
    if hasattr(args, 'model_scales'):
        analyzer.model_scales = args.model_scales
    if hasattr(args, 'generation_lengths'):
        analyzer.generation_lengths = args.generation_lengths

    # 执行分析
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