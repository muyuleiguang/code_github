import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ==================== Data Section ====================
# Spearman correlation coefficient matrix
# Row/column order: BLEU-1, BLEU-2, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, EM, NTD
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

# Metric names
metrics = ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'EM', 'NTD']

# Create an upper-triangle mask (True positions will be hidden; show lower triangle)
# To show lower triangle, set k=1; to show upper triangle, set k=0
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# ==================== Figure Parameter Settings ====================
# Figure size parameters
FIGURE_WIDTH = 8          # Figure width (inches)
FIGURE_HEIGHT = 6          # Figure height (inches)
DPI = 300                  # Resolution for saved figure clarity

# Colormap parameters
COLORMAP = 'seismic'        # Colormap options: 'RdYlGn'(red-yellow-green), 'coolwarm', 'RdBu_r'(reversed red-blue)
                           # Other options: 'viridis', 'plasma', 'seismic', 'bwr'
VMIN = -1.0               # Colormap minimum (correlation range)
VMAX = 1.0                # Colormap maximum (correlation range)
CENTER = 0.0              # Colormap center (0 = no correlation)

# If you want a custom color gradient, use the following code (uncomment):
colors = ['#d73027', '#f46d43', '#fee090', '#e0f3f8', '#74add1', '#4575b4']  # Custom color list
COLORMAP = LinearSegmentedColormap.from_list('custom', colors)

# Heatmap cell parameters
LINEWIDTHS = 0.01          # Cell border line width (0 = no border)
LINECOLOR = 'white'       # Cell border color

# Annotation parameters
ANNOT = True              # Whether to show values in cells (True/False)
ANNOT_FONTSIZE = 16       # Annotation font size
ANNOT_FORMAT = '.2f'      # Value format: '.2f', '.3f', '.0%' etc.
ANNOT_COLOR_THRESHOLD = 0.92  # Switch text color threshold (abs(value) > threshold -> white, else black)

# Axis label parameters
XLABEL_FONTSIZE = 20      # X-axis label font size
YLABEL_FONTSIZE = 20      # Y-axis label font size
TICK_FONTSIZE = 11        # Tick label font size
LABEL_ROTATION_X = 45     # X-axis label rotation (0-90)
LABEL_ROTATION_Y = 0      # Y-axis label rotation
LABEL_HA = 'right'        # X-axis label horizontal alignment: 'right', 'center', 'left'
LABEL_VA = 'top'          # X-axis label vertical alignment: 'top', 'center', 'bottom'

# Title parameters
# TITLE = 'Spearman\'s Rank Correlation Between Evaluation Metrics'  # Figure title
# TITLE_FONTSIZE = 14       # Title font size
# TITLE_PAD = 20            # Padding between title and figure

# Colorbar parameters
# CBAR_LABEL = 'Correlation Coefficient'  # Colorbar label
CBAR_FONTSIZE = 18        # Colorbar label font size
CBAR_SHRINK = 0.9         # Colorbar length scaling (0-1)
CBAR_ASPECT = 20          # Colorbar aspect ratio
CBAR_PAD = 0.0           # Padding between colorbar and heatmap

# Layout parameters
TIGHT_LAYOUT = True       # Whether to automatically adjust layout (True/False)
TOP_ADJUST = 0.95         # Top margin adjustment (0-1)
BOTTOM_ADJUST = 0.1       # Bottom margin adjustment (0-1)
LEFT_ADJUST = 0.1         # Left margin adjustment (0-1)
RIGHT_ADJUST = 0.95       # Right margin adjustment (0-1)

# Save parameters
SAVE_FORMAT = 'pdf'       # Save format: 'png', 'pdf', 'svg', 'jpg'
SAVE_FILENAME = './correlation_heatmap_lower_triangle.pdf'  # Output filename
SAVE_DPI = 300            # DPI when saving
SAVE_BBOX = 'tight'       # Bounding box: 'tight' or None

# ==================== Create Figure ====================
fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

# Create heatmap
heatmap = sns.heatmap(
    correlation_matrix,           # Data matrix
    mask=mask,                    # Mask (control upper/lower triangle display)
    annot=ANNOT,                  # Show values
    fmt=ANNOT_FORMAT,             # Value format
    cmap=COLORMAP,                # Colormap
    vmin=VMIN,                    # Minimum
    vmax=VMAX,                    # Maximum
    center=CENTER,                # Center
    square=True,                  # Square cells
    linewidths=LINEWIDTHS,        # Border width
    linecolor=LINECOLOR,          # Border color
    cbar_kws={                    # Colorbar kwargs
        # 'label': CBAR_LABEL,
        'shrink': CBAR_SHRINK,
        'aspect': CBAR_ASPECT,
        'pad': CBAR_PAD
    },
    xticklabels=metrics,          # X tick labels
    yticklabels=metrics,          # Y tick labels
    ax=ax                         # Axis
)

# ==================== Custom annotation colors (auto-adjust text color by background) ====================
# This makes text white on dark cells and black on light cells
if ANNOT:
    for text in heatmap.texts:
        # Get the numeric value from the annotation
        value = float(text.get_text())
        # Decide text color based on absolute value
        if abs(value) > ANNOT_COLOR_THRESHOLD:
            text.set_color('white')
        else:
            text.set_color('black')
        # Set font size
        text.set_fontsize(ANNOT_FONTSIZE)

# ==================== Set axis labels ====================
ax.set_xlabel('', fontsize=XLABEL_FONTSIZE)  # X-axis label (usually empty because tick labels are used)
ax.set_ylabel('', fontsize=YLABEL_FONTSIZE)  # Y-axis label

# Set tick labels
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

# ==================== Set title ====================
# ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD, fontweight='bold')

# ==================== Set colorbar label ====================
cbar = heatmap.collections[0].colorbar
# cbar.set_label(CBAR_LABEL, fontsize=CBAR_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

# ==================== Adjust layout ====================
if TIGHT_LAYOUT:
    plt.tight_layout()
else:
    plt.subplots_adjust(
        top=TOP_ADJUST,
        bottom=BOTTOM_ADJUST,
        left=LEFT_ADJUST,
        right=RIGHT_ADJUST
    )

# ==================== Save figure ====================
plt.savefig(
    SAVE_FILENAME,
    format=SAVE_FORMAT,
    dpi=SAVE_DPI,
    bbox_inches=SAVE_BBOX
)

# ==================== Show figure ====================
plt.show()

# ==================== Additional Notes ====================
"""
Quick guide for commonly modified parameters:

1. Change colormap:
   - Modify the COLORMAP variable
   - Common options: 'RdYlGn', 'coolwarm', 'RdBu_r', 'viridis', 'plasma'

2. Show upper triangle instead of lower triangle:
   - Modify the mask definition: mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)

3. Adjust figure size:
   - Modify FIGURE_WIDTH and FIGURE_HEIGHT

4. Change annotation format:
   - Modify ANNOT_FORMAT: '.2f' (2 decimals), '.3f' (3 decimals), '.1f' (1 decimal)

5. Hide annotations:
   - Set ANNOT = False

6. Save as vector format (recommended for papers):
   - Set SAVE_FORMAT = 'pdf' or 'svg'
   - Set SAVE_FILENAME = 'correlation_heatmap.pdf'

7. Adjust annotation font size:
   - Modify ANNOT_FONTSIZE

8. Add grid lines:
   - Modify LINEWIDTHS = 2.0 (increase the value)

9. Remove colorbar:
   - Add cbar=False in sns.heatmap()

10. Change label rotation (make X labels horizontal):
    - Set LABEL_ROTATION_X = 0
    - Set LABEL_HA = 'center'
"""

print(f"热力图已保存为: {SAVE_FILENAME}")
print(f"图形尺寸: {FIGURE_WIDTH}x{FIGURE_HEIGHT} 英寸")
print(f"分辨率: {SAVE_DPI} DPI")
