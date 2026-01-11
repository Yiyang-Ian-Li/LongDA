"""
Figure: Match Rate vs Average Tokens
Scatter plot showing relationship between model performance and token usage.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from adjustText import adjust_text
from scipy.optimize import curve_fit

# Configure matplotlib for high-quality academic figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Configuration
FIGURE_SIZE = (3.5, 2)
OUTPUT_DIR = Path(__file__).parent
TITLE = "Relationship Between Match Rate and Total Token Consumption"

# ============================================================================
# DATA PREPARATION
# ============================================================================
# Data from experimental results (Table: Model Performance Comparison)

data = pd.DataFrame({
    'model': [
        'GPT-5-High', 'GPT-5', 'GPT-5-mini', 
        'DeepSeek-V3.2', 'DeepSeek-V3.2-Thinking',
        'GPT-OSS-120B', 'Qwen3-Coder-480B', 'Qwen3-235B',
        'Qwen3-Next-80B', 'Kimi-K2', 'GLM-4.7'
    ],
    'match_rate': [68.91, 69.16, 40.81, 53.00, 47.82, 12.15, 23.44, 27.17, 21.85, 28.27, 19.18],
    'total_tokens_m': [8.25, 6.40, 10.66, 68.91, 36.38, 14.66, 75.49, 61.73, 22.36, 38.21, 120.01]
})

# Assign a unique color to each model
colors = sns.color_palette("viridis", n_colors=len(data))
data['color'] = colors

# ============================================================================
# PLOTTING
# ============================================================================
print(f"Generating: {TITLE}")

fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Scatter plot with different colors for each model
for idx, row in data.iterrows():
    ax.scatter(row['total_tokens_m'], row['match_rate'], 
               s=60, alpha=0.85, c=[row['color']], edgecolors='white', zorder=3)

# Add logarithmic fitted curve: y = a * log(x) + b
# Exclude GPT-5-High, GPT-5, and GLM-4.7 from fitting
def log_fit(x, a, b):
    return a * np.log(x) + b

exclude_models = ['GPT-5-High', 'GPT-5', 'GLM-4.7']
data_fit = data[~data['model'].isin(exclude_models)]

# Sort by x values for smooth curve
x_sorted = np.sort(data['total_tokens_m'].values)
popt, _ = curve_fit(log_fit, data_fit['total_tokens_m'], data_fit['match_rate'], 
                   p0=[5, 30], maxfev=10000)
x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 100)
y_fit = log_fit(x_fit, *popt)
# ax.plot(x_fit, y_fit, '--', color='#E74C3C', alpha=0.6, linewidth=1.5, 
#         label=f'Log fit: y={popt[0]:.1f}ln(x)+{popt[1]:.1f}', zorder=2)
ax.plot(x_fit, y_fit, '--', color='#3B8B8D', alpha=0.6, linewidth=1.5, 
        label=f'Log fit', zorder=2)

# Add model labels with automatic adjustment to avoid overlap
texts = []
for idx, row in data.iterrows():
    # Shorten model names for cleaner display
    label = row['model'].replace('-Instruct', '').replace('Qwen3-', 'Q-')
    label = label.replace('DeepSeek-V3.2-Thinking', 'DS-V3.2-Think')
    label = label.replace('DeepSeek-V3.2', 'DS-V3.2')
    label = label.replace('GPT-OSS-120B', 'GPT-OSS')
    label = label.replace('Kimi-K2', 'Kimi')
    
    texts.append(ax.text(row['total_tokens_m'], row['match_rate'], label,
                        fontsize=7, ha='center', va='bottom', fontweight='bold',
                        color=row['color']))

# Adjust text positions to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
            ax=ax, expand_points=(1.2, 1.3), force_points=0.5)

# Styling
ax.set_xlabel('Total Tokens (M)', fontweight='bold', fontsize=9)
ax.set_ylabel('Match Rate (%)', fontweight='bold', fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax.set_xlim(-2, data['total_tokens_m'].max() * 1.1)
ax.set_ylim(0, data['match_rate'].max() * 1.15)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, loc='best', framealpha=0.9)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "match_vs_tokens.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ Saved: {output_path}")

plt.show()
