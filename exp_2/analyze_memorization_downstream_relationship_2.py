#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二：分析记忆化变化与下游任务性能变化之间的关系
Author: Research Team
Date: 2025

主要功能：
1. 加载记忆化评估结果（base vs SFT）
2. 加载下游任务评估结果（base vs SFT）
3. 计算变化量（change_1: memorization变化，change_2: downstream变化）
4. 进行相关性分析和因果分析
5. 保存结构化结果并可视化展示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MemorizationDownstreamAnalyzer:
    """记忆化与下游任务关系分析器"""

    def __init__(self, memorization_dir, output_dir, save_prefix="exp2_analysis"):
        """
        初始化分析器

        Args:
            memorization_dir (str): 记忆化结果目录路径
            output_dir (str): 输出目录路径
            save_prefix (str): 保存文件的前缀
        """
        self.memorization_dir = memorization_dir
        self.output_dir = output_dir
        self.save_prefix = save_prefix

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 模型规模和数据集类型
        self.scales = ['1B', '7B']
        self.datasets = ['stackexchange', 'wiki-fact', 'dclm-privacy']  # dclm对应安全敏感内容

        # 记忆化指标（选择最重要的几个）
        self.mem_metrics = [
            'exact_match_rate',  # 精确匹配率
            'rouge_2_f',  # ROUGE-L F1分数
            'bleu_2',  # BLEU-4分数
            # 'semantic_similarity',  # 语义相似度
            'target_token_probability',  # 目标token概率
            'target_in_top1_rate'  # 目标token在top-1中的命中率
        ]

        # 下游任务（根据用户提供的LaTeX表格）
        self.downstream_tasks = ['GSM8K', 'MATH', 'MMLU', 'PopQA']

        # 初始化结果存储
        self.results = {}

    def load_memorization_results(self):
        """
        加载记忆化评估结果

        Returns:
            pd.DataFrame: 包含所有记忆化结果的数据框
        """
        print("📊 开始加载记忆化评估结果...")

        # 从附件CSV文件中读取数据
        filepath = "/root/autodl-tmp/ift_memorization/results/latex_tables/memorization_metrics_summary_prefix16_32_64_gen8_16_128.csv"

        if os.path.exists(filepath):
            print(f"   ✅ 加载: {os.path.basename(filepath)}")
            mem_df = pd.read_csv(filepath)
            print(f"   📈 总计加载 {len(mem_df)} 条记忆化结果记录")
            return mem_df
        else:
            raise FileNotFoundError("没有找到记忆化结果文件！")

    def create_downstream_results(self):
        """
        创建下游任务结果数据框
        基于用户提供的LaTeX表格数据

        Returns:
            pd.DataFrame: 包含所有下游任务结果的数据框
        """
        print("📊 创建下游任务评估结果...")

        # 根据用户提供的LaTeX表格数据
        # 注意：用户提供的数据中所有值都相同，这可能是示例数据
        # 在实际情况下，这些值应该是不同的
        downstream_data = []

        for i, scale in enumerate(self.scales):
            for model_type in ['base', 'sft']:
                model_name = f"OLMo-2-{scale}" if model_type == 'base' else f"OLMo-2-{scale}-SFT"
                # 任务对应的分数（从LaTeX表格中提取）
                task_scores = TASK_SCORES[model_name]

                record = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'scale': scale
                }

                # 添加各任务分数
                for j, task in enumerate(self.downstream_tasks):
                    record[task] = task_scores[j]

                downstream_data.append(record)

        downstream_df = pd.DataFrame(downstream_data)
        print(f"   📈 创建了 {len(downstream_df)} 条下游任务结果记录")

        return downstream_df

    def calculate_changes(self, mem_df, downstream_df):
        """
        计算从base到SFT的变化量

        Args:
            mem_df (pd.DataFrame): 记忆化结果
            downstream_df (pd.DataFrame): 下游任务结果

        Returns:
            tuple: (记忆化变化数据框, 下游任务变化数据框)
        """
        print("📊 计算从base到SFT的变化量...")

        # 计算记忆化变化（change_1）
        mem_changes = []

        for scale in self.scales:
            for dataset in self.datasets:
                # 获取base和sft结果
                base_data = mem_df[(mem_df['model_type'] == 'base') &
                                   (mem_df['model_scale'] == scale) &
                                   (mem_df['dataset'] == dataset)]
                sft_data = mem_df[(mem_df['model_type'] == 'sft') &
                                  (mem_df['model_scale'] == scale) &
                                  (mem_df['dataset'] == dataset)]

                if len(base_data) > 0 and len(sft_data) > 0:
                    change_record = {
                        'scale': scale,
                        'dataset': dataset
                    }

                    # 计算各指标的变化量
                    for metric in self.mem_metrics:
                        if metric in base_data.columns and metric in sft_data.columns:
                            base_val = base_data[metric].iloc[0]
                            sft_val = sft_data[metric].iloc[0]

                            # 绝对变化量
                            change_record[f'{metric}_change_abs'] = sft_val - base_val

                            # 相对变化量（避免除以0）
                            if base_val != 0:
                                change_record[f'{metric}_change_rel'] = (sft_val - base_val) / base_val
                            else:
                                change_record[f'{metric}_change_rel'] = 0.0

                    mem_changes.append(change_record)

        mem_changes_df = pd.DataFrame(mem_changes)
        print(f"   📈 计算了 {len(mem_changes_df)} 条记忆化变化记录")

        # 计算下游任务变化（change_2）
        downstream_changes = []

        for scale in self.scales:
            base_data = downstream_df[(downstream_df['model_type'] == 'base') &
                                      (downstream_df['scale'] == scale)]
            sft_data = downstream_df[(downstream_df['model_type'] == 'sft') &
                                     (downstream_df['scale'] == scale)]

            if len(base_data) > 0 and len(sft_data) > 0:
                change_record = {'scale': scale}

                # 计算各任务的变化量
                for task in self.downstream_tasks:
                    base_val = base_data[task].iloc[0]
                    sft_val = sft_data[task].iloc[0]

                    # 绝对变化量
                    change_record[f'{task}_change_abs'] = sft_val - base_val

                    # 相对变化量
                    if base_val != 0:
                        change_record[f'{task}_change_rel'] = (sft_val - base_val) / base_val
                    else:
                        change_record[f'{task}_change_rel'] = 0.0

                downstream_changes.append(change_record)

        downstream_changes_df = pd.DataFrame(downstream_changes)
        print(f"   📈 计算了 {len(downstream_changes_df)} 条下游任务变化记录")

        return mem_changes_df, downstream_changes_df

    def correlation_analysis(self, mem_changes_df, downstream_changes_df):
        """
        进行相关性分析

        Args:
            mem_changes_df (pd.DataFrame): 记忆化变化数据
            downstream_changes_df (pd.DataFrame): 下游任务变化数据

        Returns:
            dict: 相关性分析结果
        """
        print("📊 进行相关性分析...")

        correlation_results = {
            'pearson': {},
            'spearman': {},
            'correlation_matrix': {}
        }

        # 为每个数据集分别进行分析
        for dataset in self.datasets:
            print(f"   🔍 分析数据集: {dataset}")

            dataset_mem = mem_changes_df[mem_changes_df['dataset'] == dataset]

            # 合并数据（按scale）
            merged_data = pd.merge(dataset_mem, downstream_changes_df, on='scale', how='inner')

            if len(merged_data) == 0:
                print(f"     ❌ {dataset} 数据集没有匹配的记录")
                continue

            # 选择数值列进行相关性分析
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns

            # 计算Pearson和Spearman相关系数
            dataset_correlations = {}

            # 记忆化指标与下游任务的相关性
            for mem_metric in self.mem_metrics:
                mem_col_abs = f'{mem_metric}_change_abs'
                mem_col_rel = f'{mem_metric}_change_rel'

                if mem_col_abs in merged_data.columns:
                    for task in self.downstream_tasks:
                        task_col_abs = f'{task}_change_abs'
                        task_col_rel = f'{task}_change_rel'

                        if task_col_abs in merged_data.columns:
                            # Pearson相关性
                            try:
                                pearson_corr, pearson_p = pearsonr(
                                    merged_data[mem_col_abs].dropna(),
                                    merged_data[task_col_abs].dropna()
                                )

                                spearman_corr, spearman_p = spearmanr(
                                    merged_data[mem_col_abs].dropna(),
                                    merged_data[task_col_abs].dropna()
                                )

                                key = f"{mem_metric}_vs_{task}"
                                dataset_correlations[key] = {
                                    'pearson_corr': pearson_corr,
                                    'pearson_p': pearson_p,
                                    'spearman_corr': spearman_corr,
                                    'spearman_p': spearman_p,
                                    'sample_size': len(merged_data[mem_col_abs].dropna())
                                }

                            except Exception as e:
                                print(f"     ⚠️  计算{mem_metric}与{task}相关性时出错: {e}")

            correlation_results['pearson'][dataset] = dataset_correlations
            correlation_results['spearman'][dataset] = dataset_correlations

            # 计算相关性矩阵
            if len(numeric_cols) > 1:
                corr_matrix = merged_data[numeric_cols].corr()
                correlation_results['correlation_matrix'][dataset] = corr_matrix

        return correlation_results

    def causal_analysis(self, mem_changes_df, downstream_changes_df):
        """
        进行因果分析（使用线性回归和随机森林）

        Args:
            mem_changes_df (pd.DataFrame): 记忆化变化数据
            downstream_changes_df (pd.DataFrame): 下游任务变化数据

        Returns:
            dict: 因果分析结果
        """
        print("📊 进行因果分析...")

        causal_results = {
            'linear_regression': {},
            'random_forest': {},
            'feature_importance': {}
        }

        for dataset in self.datasets:
            print(f"   🔍 分析数据集: {dataset}")

            dataset_mem = mem_changes_df[mem_changes_df['dataset'] == dataset]
            merged_data = pd.merge(dataset_mem, downstream_changes_df, on='scale', how='inner')

            if len(merged_data) < 3:  # 需要至少3个数据点进行回归
                print(f"     ❌ {dataset} 数据点太少，跳过因果分析")
                continue

            # 准备特征（记忆化变化作为输入）
            feature_cols = []
            for metric in self.mem_metrics:
                col_abs = f'{metric}_change_abs'
                col_rel = f'{metric}_change_rel'
                if col_abs in merged_data.columns:
                    feature_cols.append(col_abs)
                if col_rel in merged_data.columns:
                    feature_cols.append(col_rel)

            if len(feature_cols) == 0:
                print(f"     ❌ {dataset} 没有可用的记忆化特征")
                continue

            X = merged_data[feature_cols].fillna(0)

            dataset_causal = {}

            # 对每个下游任务进行预测
            for task in self.downstream_tasks:
                target_col = f'{task}_change_abs'

                if target_col in merged_data.columns:
                    y = merged_data[target_col].fillna(0)

                    if len(y.unique()) <= 1:  # 如果目标变量没有变化，跳过
                        continue

                    try:
                        # 线性回归
                        lr_model = LinearRegression()
                        lr_model.fit(X, y)
                        lr_pred = lr_model.predict(X)
                        lr_r2 = r2_score(y, lr_pred)
                        lr_mse = mean_squared_error(y, lr_pred)

                        # 随机森林
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)
                        rf_pred = rf_model.predict(X)
                        rf_r2 = r2_score(y, rf_pred)
                        rf_mse = mean_squared_error(y, rf_pred)

                        dataset_causal[task] = {
                            'linear_regression': {
                                'r2_score': lr_r2,
                                'mse': lr_mse,
                                'coefficients': lr_model.coef_.tolist(),
                                'intercept': lr_model.intercept_
                            },
                            'random_forest': {
                                'r2_score': rf_r2,
                                'mse': rf_mse,
                                'feature_importance': rf_model.feature_importances_.tolist()
                            }
                        }

                    except Exception as e:
                        print(f"     ⚠️  {task}因果分析出错: {e}")

            causal_results['linear_regression'][dataset] = dataset_causal
            causal_results['random_forest'][dataset] = dataset_causal

            # 特征重要性分析
            if dataset_causal:
                importance_summary = {}
                for task, results in dataset_causal.items():
                    if 'random_forest' in results:
                        importance_summary[task] = dict(
                            zip(feature_cols, results['random_forest']['feature_importance']))

                causal_results['feature_importance'][dataset] = importance_summary

        return causal_results

    def save_results(self, mem_df, downstream_df, mem_changes_df, downstream_changes_df,
                     correlation_results, causal_results):
        """
        保存分析结果到文件

        Args:
            mem_df: 原始记忆化数据
            downstream_df: 原始下游任务数据
            mem_changes_df: 记忆化变化数据
            downstream_changes_df: 下游任务变化数据
            correlation_results: 相关性分析结果
            causal_results: 因果分析结果
        """
        print("💾 保存分析结果...")

        # 保存原始数据
        mem_df.to_csv(os.path.join(self.output_dir, f"{self.save_prefix}_memorization_raw.csv"), index=False)
        downstream_df.to_csv(os.path.join(self.output_dir, f"{self.save_prefix}_downstream_raw.csv"), index=False)

        # 保存变化数据
        mem_changes_df.to_csv(os.path.join(self.output_dir, f"{self.save_prefix}_memorization_changes.csv"),
                              index=False)
        downstream_changes_df.to_csv(os.path.join(self.output_dir, f"{self.save_prefix}_downstream_changes.csv"),
                                     index=False)

        # 保存分析结果（JSON格式）
        with open(os.path.join(self.output_dir, f"{self.save_prefix}_correlation_results.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(correlation_results, f, indent=2, ensure_ascii=False, default=str)

        with open(os.path.join(self.output_dir, f"{self.save_prefix}_causal_results.json"), 'w', encoding='utf-8') as f:
            json.dump(causal_results, f, indent=2, ensure_ascii=False, default=str)

        # 保存综合结果摘要
        summary = {
            'experiment_info': {
                'scales': self.scales,
                'datasets': self.datasets,
                'memorization_metrics': self.mem_metrics,
                'downstream_tasks': self.downstream_tasks,
                'total_memorization_records': len(mem_df),
                'total_downstream_records': len(downstream_df),
                'memorization_changes_records': len(mem_changes_df),
                'downstream_changes_records': len(downstream_changes_df)
            },
            'analysis_results': {
                'correlation_analysis': correlation_results,
                'causal_analysis': causal_results
            }
        }

        with open(os.path.join(self.output_dir, f"{self.save_prefix}_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"   ✅ 所有结果已保存到: {self.output_dir}")

    def print_results_summary(self, correlation_results, causal_results):
        """
        打印分析结果摘要

        Args:
            correlation_results: 相关性分析结果
            causal_results: 因果分析结果
        """
        print("\n" + "=" * 80)
        print("📊 实验二分析结果摘要")
        print("=" * 80)

        # 相关性分析摘要
        print("\n🔗 相关性分析结果:")
        for dataset in self.datasets:
            if dataset in correlation_results.get('pearson', {}):
                print(f"\n  📋 数据集: {dataset}")
                dataset_corr = correlation_results['pearson'][dataset]

                if dataset_corr:
                    # 找到最强的相关性
                    max_corr = 0
                    max_pair = ""

                    for pair, stats in dataset_corr.items():
                        if abs(stats['pearson_corr']) > abs(max_corr):
                            max_corr = stats['pearson_corr']
                            max_pair = pair

                    print(f"    🏆 最强相关性: {max_pair}")
                    print(f"    📈 Pearson相关系数: {max_corr:.4f}")

                    # 统计显著相关的数量
                    significant_count = sum(1 for stats in dataset_corr.values()
                                            if stats['pearson_p'] < 0.05)
                    print(f"    ✅ 显著相关对数 (p<0.05): {significant_count}/{len(dataset_corr)}")
                else:
                    print("    ❌ 没有计算出相关性结果")

        # 因果分析摘要
        print("\n🎯 因果分析结果:")
        for dataset in self.datasets:
            if dataset in causal_results.get('linear_regression', {}):
                print(f"\n  📋 数据集: {dataset}")
                dataset_causal = causal_results['linear_regression'][dataset]

                if dataset_causal:
                    # 找到预测效果最好的任务
                    max_r2 = 0
                    best_task = ""

                    for task, results in dataset_causal.items():
                        if 'linear_regression' in results:
                            r2 = results['linear_regression']['r2_score']
                            if r2 > max_r2:
                                max_r2 = r2
                                best_task = task

                    if best_task:
                        print(f"    🏆 最佳预测任务: {best_task}")
                        print(f"    📈 线性回归R²: {max_r2:.4f}")

                        if best_task in causal_results['random_forest'][dataset]:
                            rf_r2 = causal_results['random_forest'][dataset][best_task]['random_forest']['r2_score']
                            print(f"    🌲 随机森林R²: {rf_r2:.4f}")

                    print(f"    📊 可预测任务数: {len(dataset_causal)}")
                else:
                    print("    ❌ 没有计算出因果分析结果")

        print("\n" + "=" * 80)

    def visualize_results(self, mem_changes_df, downstream_changes_df, correlation_results):
        """
        可视化分析结果

        Args:
            mem_changes_df: 记忆化变化数据
            downstream_changes_df: 下游任务变化数据
            correlation_results: 相关性分析结果
        """
        print("📊 开始创建可视化图表...")

        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. 记忆化变化趋势图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('实验二：记忆化与下游任务变化分析', fontsize=16, fontweight='bold')

        # 1.1 记忆化指标变化（按规模）
        ax1 = axes[0, 0]
        mem_pivot = mem_changes_df.pivot_table(
            index='scale',
            columns='dataset',
            values='exact_match_rate_change_abs',
            aggfunc='mean'
        )

        if not mem_pivot.empty:
            sns.heatmap(mem_pivot, annot=True, fmt='.4f', ax=ax1, cmap='RdBu_r')
            ax1.set_title('精确匹配率变化热力图\n(SFT - Base)')
            ax1.set_xlabel('数据集')
            ax1.set_ylabel('模型规模')
        else:
            ax1.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('精确匹配率变化热力图')

        # 1.2 下游任务变化（按规模）
        ax2 = axes[0, 1]
        if not downstream_changes_df.empty:
            # 选择几个主要任务进行展示
            main_tasks = ['IFEval_change_abs', 'MMLU_change_abs', 'GSM8K_change_abs']
            available_tasks = [task for task in main_tasks if task in downstream_changes_df.columns]

            if available_tasks:
                downstream_changes_df.set_index('scale')[available_tasks].plot(kind='bar', ax=ax2)
                ax2.set_title('主要下游任务性能变化\n(SFT - Base)')
                ax2.set_xlabel('模型规模')
                ax2.set_ylabel('性能变化')
                ax2.legend(title='任务', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.tick_params(axis='x', rotation=0)
        else:
            ax2.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('下游任务性能变化')

        # 1.3 相关性矩阵示例（选择一个数据集）
        ax3 = axes[1, 0]
        if correlation_results.get('correlation_matrix'):
            # 选择第一个有数据的数据集
            dataset_with_data = None
            for dataset in self.datasets:
                if dataset in correlation_results['correlation_matrix']:
                    dataset_with_data = dataset
                    break

            if dataset_with_data:
                corr_matrix = correlation_results['correlation_matrix'][dataset_with_data]

                # 选择部分相关性进行展示（避免图表过于复杂）
                display_cols = [col for col in corr_matrix.columns if 'change_abs' in col][:10]
                if display_cols:
                    display_matrix = corr_matrix.loc[display_cols, display_cols]
                    sns.heatmap(display_matrix, annot=True, fmt='.2f', ax=ax3,
                                cmap='coolwarm', center=0, square=True)
                    ax3.set_title(f'相关性矩阵 ({dataset_with_data})')
                    ax3.tick_params(axis='both', rotation=45)

        if not hasattr(ax3, 'collections') or not ax3.collections:
            ax3.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('相关性矩阵')

        # 1.4 规模vs变化趋势
        ax4 = axes[1, 1]

        # 创建规模数值映射
        scale_mapping = {'1B': 1, '7B': 7, '13B': 13, '32B': 32}

        if not mem_changes_df.empty:
            # 计算每个规模下记忆化的平均变化
            mem_avg_changes = []
            scales_numeric = []

            for scale in self.scales:
                scale_data = mem_changes_df[mem_changes_df['scale'] == scale]
                if not scale_data.empty:
                    avg_change = scale_data['exact_match_rate_change_abs'].mean()
                    if not np.isnan(avg_change):
                        mem_avg_changes.append(avg_change)
                        scales_numeric.append(scale_mapping[scale])

            if mem_avg_changes:
                ax4.plot(scales_numeric, mem_avg_changes, 'o-', label='记忆化变化', linewidth=2, markersize=8)

        if not downstream_changes_df.empty:
            # 计算每个规模下下游任务的平均变化
            downstream_avg_changes = []
            scales_numeric_downstream = []

            # 计算所有下游任务变化的平均值
            change_cols = [col for col in downstream_changes_df.columns if 'change_abs' in col]
            if change_cols:
                downstream_changes_df['avg_change'] = downstream_changes_df[change_cols].mean(axis=1)

                for scale in self.scales:
                    scale_data = downstream_changes_df[downstream_changes_df['scale'] == scale]
                    if not scale_data.empty:
                        avg_change = scale_data['avg_change'].mean()
                        if not np.isnan(avg_change):
                            downstream_avg_changes.append(avg_change)
                            scales_numeric_downstream.append(scale_mapping[scale])

                if downstream_avg_changes:
                    ax4.plot(scales_numeric_downstream, downstream_avg_changes, 's-',
                             label='下游任务变化', linewidth=2, markersize=8)

        ax4.set_xlabel('模型规模 (B)')
        ax4.set_ylabel('平均变化量')
        ax4.set_title('模型规模 vs 性能变化趋势')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 设置x轴刻度
        if scales_numeric or scales_numeric_downstream:
            all_scales = list(set(scales_numeric + scales_numeric_downstream))
            ax4.set_xticks(sorted(all_scales))
            ax4.set_xticklabels([f'{s}B' for s in sorted(all_scales)])

        plt.tight_layout()

        # 保存图表
        viz_path = os.path.join(self.output_dir, f"{self.save_prefix}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 可视化图表已保存: {viz_path}")

        # 2. 创建详细的相关性分析图
        self._create_correlation_plots(correlation_results)

        plt.show()

    def _create_correlation_plots(self, correlation_results):
        """
        创建详细的相关性分析图表

        Args:
            correlation_results: 相关性分析结果
        """
        if not correlation_results.get('pearson'):
            print("   ⚠️  没有相关性数据，跳过相关性图表创建")
            return

        # 为每个数据集创建相关性图
        for dataset in self.datasets:
            if dataset not in correlation_results['pearson']:
                continue

            dataset_corr = correlation_results['pearson'][dataset]
            if not dataset_corr:
                continue

            # 提取相关性数据
            pairs = []
            pearson_corrs = []
            spearman_corrs = []
            p_values = []

            for pair, stats in dataset_corr.items():
                pairs.append(pair.replace('_vs_', '\nvs\n'))
                pearson_corrs.append(stats['pearson_corr'])
                spearman_corrs.append(stats['spearman_corr'])
                p_values.append(stats['pearson_p'])

            if not pairs:
                continue

            # 创建相关性对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'{dataset} 数据集相关性分析', fontsize=14, fontweight='bold')

            # Pearson vs Spearman 相关性对比
            x_pos = np.arange(len(pairs))
            width = 0.35

            bars1 = ax1.bar(x_pos - width / 2, pearson_corrs, width, label='Pearson', alpha=0.8)
            bars2 = ax1.bar(x_pos + width / 2, spearman_corrs, width, label='Spearman', alpha=0.8)

            ax1.set_xlabel('记忆化指标 vs 下游任务')
            ax1.set_ylabel('相关系数')
            ax1.set_title('Pearson vs Spearman 相关性')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=7)

            # 显著性分析
            significant_indices = [i for i, p in enumerate(p_values) if p < 0.05]
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]

            bars3 = ax2.bar(x_pos, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
            ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
            ax2.set_xlabel('记忆化指标 vs 下游任务')
            ax2.set_ylabel('-log10(p-value)')
            ax2.set_title('统计显著性分析')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存相关性图表
            corr_viz_path = os.path.join(self.output_dir, f"{self.save_prefix}_correlation_{dataset}.png")
            plt.savefig(corr_viz_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ {dataset}相关性图表已保存: {corr_viz_path}")

    def run_analysis(self):
        """
        运行完整的分析流程
        """
        print("🚀 开始实验二：记忆化与下游任务关系分析")
        print("=" * 80)

        try:
            # 1. 加载数据
            mem_df = self.load_memorization_results()
            downstream_df = self.create_downstream_results()

            # 2. 计算变化量
            mem_changes_df, downstream_changes_df = self.calculate_changes(mem_df, downstream_df)

            # 3. 相关性分析
            correlation_results = self.correlation_analysis(mem_changes_df, downstream_changes_df)

            # 4. 因果分析
            causal_results = self.causal_analysis(mem_changes_df, downstream_changes_df)

            # 5. 保存结果
            self.save_results(mem_df, downstream_df, mem_changes_df, downstream_changes_df,
                              correlation_results, causal_results)

            # 6. 打印结果摘要
            self.print_results_summary(correlation_results, causal_results)

            # 7. 可视化
            self.visualize_results(mem_changes_df, downstream_changes_df, correlation_results)

            print("\n🎉 实验二分析完成！")
            print(f"📁 所有结果已保存到: {self.output_dir}")

        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数：设置参数并运行分析
    """
    parser = argparse.ArgumentParser(description="实验二：记忆化与下游任务关系分析")

    parser.add_argument(
        '--memorization_dir',
        type=str,
        default='/root/autodl-tmp/ift_memorization/results/exp1_mem_score',
        help='记忆化结果目录路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/autodl-tmp/ift_memorization/results/exp2_relationship_analysis',
        help='输出目录路径'
    )

    parser.add_argument(
        '--save_prefix',
        type=str,
        default='exp2Relation',
        help='保存文件的前缀'
    )

    args = parser.parse_args()

    print("📋 分析参数:")
    print(f"   📂 记忆化结果目录: {args.memorization_dir}")
    print(f"   📂 输出目录: {args.output_dir}")
    print(f"   🏷️  保存前缀: {args.save_prefix}")
    print()



    # 创建分析器并运行
    analyzer = MemorizationDownstreamAnalyzer(
        memorization_dir=args.memorization_dir,
        output_dir=args.output_dir,
        save_prefix=args.save_prefix
    )

    analyzer.run_analysis()


if __name__ == "__main__":
    # 任务对应的分数（从LaTeX表格中提取） GSM8K     # MATH     # MMLU     # PopQA
    TASK_SCORES = {
        "OLMo-2-1B": [0.41, 0.06, 0.49, 0.27],
        "OLMo-2-1B-SFT": [0.38, 0.12, 0.45, 0.20],
        "OLMo-2-7B": [0.68, 0.06, 0.69, 0.35],
        "OLMo-2-7B-SFT": [0.71, 0.21, 0.67, 0.26],
        # "OLMo-2-1124-13B": [0.790, 0.000, 0.390, 0.100, 0.000, 0.950],
        # "OLMo-2-1124-13B-SFT": [],
        # "OLMo-2-0325-32B": [0.790, 0.000, 0.390, 0.100, 0.000, 0.950],46
        # "OLMo-2-0325-32B-SFT": [],
    }
    main()