"""
Figure: Single Number vs List Answer Comparison
Bar chart comparing coverage and match rates between scalar and list answers.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import glob

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

# Set seaborn style
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# Configuration
FIGURE_SIZE = (3.5, 1.8)
COLORS = {
    'coverage': '#8EC0DB',  # Sky blue (lighter) - matching survey_comparison
    'match': '#FF6B6B',     # Coral red - matching survey_comparison
}
OUTPUT_DIR = Path(__file__).parent
TITLE = "Performance Comparison: Scalar vs List Answers"

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Load all evaluated_answers files
base_dir = Path(__file__).parent.parent
results_dir = base_dir / "results"
answer_files = list(results_dir.glob("*/evaluated_answers.csv"))

print(f"Found {len(answer_files)} evaluated_answers files")

# Collect all data
all_data = []
for answer_file in answer_files:
    df = pd.read_csv(answer_file)
    all_data.append(df)

# Combine all data
combined_data = pd.concat(all_data, ignore_index=True)
print(f"Total questions: {len(combined_data)}")

# Classify as scalar or list based on answer_structure
combined_data['answer_type'] = combined_data['answer_structure'].apply(
    lambda x: 'Scalar' if x == 'single_number' else 'List'
)

# Calculate metrics for each type
results = []
for answer_type in ['Scalar', 'List']:
    subset = combined_data[combined_data['answer_type'] == answer_type]
    total = len(subset)
    coverage_rate = (subset['coverage'].sum() / total * 100) if total > 0 else 0
    match_rate = (subset['match'].sum() / total * 100) if total > 0 else 0
    
    results.append({
        'answer_type': answer_type,
        'coverage_rate': coverage_rate,
        'match_rate': match_rate,
        'n': total
    })
    print(f"{answer_type}: n={total}, coverage={coverage_rate:.2f}%, match={match_rate:.2f}%")

data = pd.DataFrame(results)

# ============================================================================
# PLOTTING
# ============================================================================
print(f"Generating: {TITLE}")

fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Bar positions
x = np.arange(len(data))
width = 0.35

# Create grouped bars without borders
bars1 = ax.bar(x - width/2, data['coverage_rate'], width, 
               label='Coverage Rate', color=COLORS['coverage'], 
               edgecolor='none', alpha=0.8)
bars2 = ax.bar(x + width/2, data['match_rate'], width,
               label='Match Rate', color=COLORS['match'],
               edgecolor='none', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

# Styling
# ax.set_xlabel('Answer Type', fontweight='bold', fontsize=9)
ax.set_ylabel('Rate (%)', fontweight='bold', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(data['answer_type'], fontsize=8)
ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
ax.set_ylim(0, 95)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(labelsize=8)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "single_vs_list.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ Saved: {output_path}")

plt.show()
