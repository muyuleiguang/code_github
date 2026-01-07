#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成独立的归一化对比图
每个指标一个独立的PDF文件
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 设置绘图风格 - 按用户要求
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 6.0
plt.rcParams['lines.markersize'] = 15
plt.rcParams['titleweight'] = 'bold'
plt.rcParams['labelweight'] = 'bold'
# --- 字体粗细 (Font Weight) ---
 #   'axes.titleweight': 'bold',  # 标题加粗
  #  'axes.labelweight': 'bold',  # 标签加粗

# 配色方案
COLOR_1B = '#D62728'  # 红色
COLOR_7B = '#1F77B4'  # 蓝色

def load_data(json_file):
    """加载JSON数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_metric_normalized(data_1b, data_7b, metric_key, 
                                   ylabel, title, output_file, dpi=300):
    """
    绘制单个指标的归一化对比图
    
    参数:
        data_1b: 1B模型数据
        data_7b: 7B模型数据
        metric_key: 指标键名
        ylabel: Y轴标签
        title: 图标题
        output_file: 输出文件路径
        dpi: 分辨率
    """
    # 创建图表 - 使用用户指定的figsize
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 获取数据
    layers_1b = np.arange(len(data_1b[metric_key]))
    layers_7b = np.arange(len(data_7b[metric_key]))
    values_1b = np.array(data_1b[metric_key])
    values_7b = np.array(data_7b[metric_key])
    
    # 归一化到[0, 1]
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
    
    # 绘制曲线
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
    
    # 设置标签 - 使用用户指定的labelpad
    ax.set_xlabel('Layer Index', labelpad=2)
    ax.set_ylabel(ylabel)
    
    # 设置标题 - 不添加(Normalized)后缀，保持原标题
    ax.set_title(title)
    
    # 设置Y轴范围和格式
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    # 图例 - 去掉range信息
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # 网格 - 不设置背景色
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()


def main():
    # 加载数据
    data = load_data('/mnt/user-data/uploads/1763344449209_representation_analysis_results.json')
    
    data_1b = data['1B']['idiom']['metrics']
    data_7b = data['7B']['idiom']['metrics']
    
    # 定义要绘制的指标 - 保持与原代码一致的ylabel和title
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
    
    # 生成每个指标的图表
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
