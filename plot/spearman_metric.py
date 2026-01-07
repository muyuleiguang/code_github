import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ==================== 数据部分 ====================
# Spearman相关系数矩阵数据
# 行和列的顺序：BLEU-1, BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, EM, NTD
correlation_matrix = np.array([
    [1.00,  0.98,  0.94,  0.97,  0.92,  0.96,  0.91, -0.95],  # BLEU-1
    [0.98,  1.00,  0.96,  0.96,  0.95,  0.96,  0.94, -0.96],  # BLEU-2
    [0.94,  0.96,  1.00,  0.93,  0.97,  0.95,  0.96, -0.94],  # BLEU-4
    [0.97,  0.96,  0.93,  1.00,  0.94,  0.98,  0.92, -0.95],  # ROUGE-1
    [0.92,  0.95,  0.97,  0.94,  1.00,  0.95,  0.96, -0.93],  # ROUGE-2
    [0.96,  0.96,  0.95,  0.98,  0.95,  1.00,  0.93, -0.96],  # ROUGE-L
    [0.91,  0.94,  0.96,  0.92,  0.96,  0.93,  1.00, -0.93],  # EM
    [-0.95, -0.96, -0.94, -0.95, -0.93, -0.96, -0.93, 1.00]   # NTD
])

# 指标名称
metrics = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'EM', 'NTD']

# 创建下三角掩码（True的位置将被隐藏，显示上三角）
# 如果要显示下三角，设置 k=1；如果要显示上三角，设置 k=0
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# ==================== 图形参数设置 ====================
# 图形大小参数
FIGURE_WIDTH = 8          # 图形宽度（英寸）
FIGURE_HEIGHT = 6          # 图形高度（英寸）
DPI = 300                  # 分辨率，用于保存图片时的清晰度

# 颜色方案参数
COLORMAP = 'seismic'        # 色图选项：'RdYlGn'(红黄绿), 'coolwarm'(冷暖), 'RdBu_r'(红蓝反转)
                           # 其他选项: 'viridis', 'plasma', 'seismic', 'bwr'
VMIN = -1.0               # 色图最小值（相关系数范围）
VMAX = 1.0                # 色图最大值（相关系数范围）
CENTER = 0.0              # 色图中心值（0表示无相关）

# 如果想自定义颜色渐变，可以使用以下代码（取消注释）：
colors = ['#d73027', '#f46d43', '#fee090', '#e0f3f8', '#74add1', '#4575b4']  # 自定义颜色列表
COLORMAP = LinearSegmentedColormap.from_list('custom', colors)

# 热力图单元格参数
LINEWIDTHS = 0.01          # 单元格边框线宽度（0表示无边框）
LINECOLOR = 'white'       # 单元格边框颜色

# 数值标注参数
ANNOT = True              # 是否在单元格中显示数值（True/False）
ANNOT_FONTSIZE = 16       # 数值字体大小
ANNOT_FORMAT = '.2f'      # 数值格式：'.2f'(两位小数), '.3f'(三位小数), '.0%'(百分比)
ANNOT_COLOR_THRESHOLD = 0.92  # 数值颜色切换阈值（绝对值大于此值用白色，否则用黑色）

# 轴标签参数
XLABEL_FONTSIZE = 20      # X轴标签字体大小
YLABEL_FONTSIZE = 20      # Y轴标签字体大小
TICK_FONTSIZE = 11        # 刻度标签字体大小
LABEL_ROTATION_X = 45     # X轴标签旋转角度（0-90度）
LABEL_ROTATION_Y = 0      # Y轴标签旋转角度
LABEL_HA = 'right'        # X轴标签水平对齐：'right', 'center', 'left'
LABEL_VA = 'top'          # X轴标签垂直对齐：'top', 'center', 'bottom'

# 标题参数
# TITLE = 'Spearman\'s Rank Correlation Between Evaluation Metrics'  # 图形标题
# TITLE_FONTSIZE = 14       # 标题字体大小
# TITLE_PAD = 20            # 标题与图形的间距

# 颜色条参数
# CBAR_LABEL = 'Correlation Coefficient'  # 颜色条标签
CBAR_FONTSIZE = 18        # 颜色条标签字体大小
CBAR_SHRINK = 0.9         # 颜色条长度缩放比例（0-1）
CBAR_ASPECT = 20          # 颜色条宽高比
CBAR_PAD = 0.0           # 颜色条与热力图的间距

# 布局参数
TIGHT_LAYOUT = True       # 是否自动调整布局（True/False）
TOP_ADJUST = 0.95         # 顶部边距调整（0-1）
BOTTOM_ADJUST = 0.1       # 底部边距调整（0-1）
LEFT_ADJUST = 0.1         # 左侧边距调整（0-1）
RIGHT_ADJUST = 0.95       # 右侧边距调整（0-1）

# 保存参数
SAVE_FORMAT = 'pdf'       # 保存格式：'png', 'pdf', 'svg', 'jpg'
SAVE_FILENAME = './correlation_heatmap_lower_triangle.pdf'  # 保存文件名
SAVE_DPI = 300            # 保存时的DPI
SAVE_BBOX = 'tight'       # 保存时的边界框：'tight'(紧凑), None(标准)

# ==================== 创建图形 ====================
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

# 创建热力图
heatmap = sns.heatmap(
    correlation_matrix,           # 数据矩阵
    mask=mask,                    # 掩码（控制显示上三角或下三角）
    annot=ANNOT,                  # 是否显示数值
    fmt=ANNOT_FORMAT,             # 数值格式
    cmap=COLORMAP,                # 颜色方案
    vmin=VMIN,                    # 最小值
    vmax=VMAX,                    # 最大值
    center=CENTER,                # 中心值
    square=True,                  # 单元格是否为正方形
    linewidths=LINEWIDTHS,        # 边框线宽
    linecolor=LINECOLOR,          # 边框颜色
    cbar_kws={                    # 颜色条参数字典
        # 'label': CBAR_LABEL,
        'shrink': CBAR_SHRINK,
        'aspect': CBAR_ASPECT,
        'pad': CBAR_PAD
    },
    xticklabels=metrics,          # X轴标签
    yticklabels=metrics,          # Y轴标签
    ax=ax                         # 绘图轴
)

# ==================== 自定义数值颜色（根据背景色自动调整文字颜色）====================
# 这段代码让深色背景上显示白色文字，浅色背景上显示黑色文字
if ANNOT:
    for text in heatmap.texts:
        # 获取文本值
        value = float(text.get_text())
        # 根据值的绝对值决定文字颜色
        if abs(value) > ANNOT_COLOR_THRESHOLD:
            text.set_color('white')
        else:
            text.set_color('black')
        # 设置字体大小
        text.set_fontsize(ANNOT_FONTSIZE)

# ==================== 设置轴标签 ====================
ax.set_xlabel('', fontsize=XLABEL_FONTSIZE)  # X轴标签（通常留空，因为有tick labels）
ax.set_ylabel('', fontsize=YLABEL_FONTSIZE)  # Y轴标签

# 设置刻度标签
ax.set_xticklabels(
    metrics,
    rotation=LABEL_ROTATION_X,
    ha=LABEL_HA,
    va=LABEL_VA,
    fontsize=TICK_FONTSIZE
)
ax.set_yticklabels(
    metrics,
    rotation=LABEL_ROTATION_Y,
    fontsize=TICK_FONTSIZE
)

# ==================== 设置标题 ====================
# ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD, fontweight='bold')

# ==================== 设置颜色条标签 ====================
cbar = heatmap.collections[0].colorbar
# cbar.set_label(CBAR_LABEL, fontsize=CBAR_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

# ==================== 调整布局 ====================
if TIGHT_LAYOUT:
    plt.tight_layout()
else:
    plt.subplots_adjust(
        top=TOP_ADJUST,
        bottom=BOTTOM_ADJUST,
        left=LEFT_ADJUST,
        right=RIGHT_ADJUST
    )

# ==================== 保存图形 ====================
plt.savefig(
    SAVE_FILENAME,
    format=SAVE_FORMAT,
    dpi=SAVE_DPI,
    bbox_inches=SAVE_BBOX
)

# ==================== 显示图形 ====================
plt.show()

# ==================== 额外说明 ====================
"""
常用参数快速修改指南：

1. 改变颜色方案：
   - 修改 COLORMAP 变量
   - 常用选项：'RdYlGn', 'coolwarm', 'RdBu_r', 'viridis', 'plasma'

2. 显示上三角而非下三角：
   - 修改 mask 定义：mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)

3. 调整图形大小：
   - 修改 FIGURE_WIDTH 和 FIGURE_HEIGHT

4. 修改数值显示格式：
   - 修改 ANNOT_FORMAT：'.2f'(两位小数), '.3f'(三位小数), '.1f'(一位小数)

5. 隐藏数值：
   - 设置 ANNOT = False

6. 改变保存格式为矢量图（适合论文）：
   - 设置 SAVE_FORMAT = 'pdf' 或 'svg'
   - 设置 SAVE_FILENAME = 'correlation_heatmap.pdf'

7. 调整数值字体大小：
   - 修改 ANNOT_FONTSIZE

8. 添加网格线：
   - 修改 LINEWIDTHS = 2.0（增加值）

9. 移除颜色条：
   - 在 sns.heatmap() 中添加参数 cbar=False

10. 改变标签旋转角度（让X轴标签水平显示）：
    - 设置 LABEL_ROTATION_X = 0
    - 设置 LABEL_HA = 'center'
"""

print(f"热力图已保存为: {SAVE_FILENAME}")
print(f"图形尺寸: {FIGURE_WIDTH}x{FIGURE_HEIGHT} 英寸")
print(f"分辨率: {SAVE_DPI} DPI")