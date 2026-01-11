"""
Figure: Survey-wise Performance Comparison
Horizontal bar chart showing coverage and match rates across all surveys.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
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

# Configuration
FIGURE_SIZE = (8, 2.5)
COLORS = {
    'coverage': '#8EC0DB',  # Sky blue (lighter)
    'match': '#FF6B6B',     # Coral red
    'context': '#808080',   # Gray for context length
}
OUTPUT_DIR = Path(__file__).parent
TITLE = "Performance Across Surveys"

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Load all evaluation summary files
base_dir = Path(__file__).parent.parent
results_dir = base_dir / "results"
summary_files = list(results_dir.glob("*/evaluation_summary.json"))

print(f"Found {len(summary_files)} evaluation summary files")

# Load benchmark.csv to get survey-source mapping
benchmark_df = pd.read_csv(base_dir / "benchmark" / "benchmark.csv")
source_to_survey = dict(zip(benchmark_df['source'], benchmark_df['survey']))
print(f"Loaded survey-source mapping: {len(source_to_survey)} sources across {len(benchmark_df['survey'].unique())} surveys")

# Collect data from all summaries, grouped by survey
survey_data = {}

for summary_file in summary_files:
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Extract by_source data and map to survey
    if 'by_source' in summary:
        for source, metrics in summary['by_source'].items():
            # Map source to survey
            survey = source_to_survey.get(source, None)
            if survey is None:
                print(f"Warning: source '{source}' not found in benchmark.csv")
                continue
            
            if survey not in survey_data:
                survey_data[survey] = {
                    'coverage_rates': [],
                    'match_rates': []
                }
            survey_data[survey]['coverage_rates'].append(metrics['coverage'])
            survey_data[survey]['match_rates'].append(metrics['match_rate'])

# Calculate average for each survey
data = []
for survey, metrics in survey_data.items():
    avg_coverage = np.mean(metrics['coverage_rates'])
    avg_match = np.mean(metrics['match_rates'])
    data.append({
        'survey': survey,
        'coverage_rate': avg_coverage,
        'match_rate': avg_match
    })

data = pd.DataFrame(data)
print(f"Processed {len(data)} surveys")

# Load context length data
docs_length_path = base_dir / "docs_length_analysis.csv"
if docs_length_path.exists():
    docs_length_df = pd.read_csv(docs_length_path)
    print(f"Loaded context length for surveys: {list(docs_length_df['survey'].values)}")
    # Merge with existing data
    data = data.merge(docs_length_df, on='survey', how='left')
    data['total_doc_tokens'] = data['total_doc_tokens'].fillna(0)
    print(f"Loaded context length data for {len(docs_length_df)} surveys")
else:
    print(f"Warning: {docs_length_path} not found, context length will be 0")
    data['total_doc_tokens'] = 0

# ============================================================================
# PLOTTING
# ============================================================================
print(f"Generating: {TITLE}")

fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)

# Sort by coverage rate for better visualization
data_sorted = data.sort_values('coverage_rate', ascending=False)

# X positions for surveys
x_pos = np.arange(len(data_sorted))
bar_width = 0.3

# Create bars with coverage and match overlaid
# First draw coverage (wider, more transparent)
bars_coverage = ax1.bar(x_pos - bar_width/2, data_sorted['coverage_rate'], 
                        width=bar_width, 
                        label='Coverage Rate',
                        color=COLORS['coverage'], alpha=0.7, 
                        edgecolor='none', zorder=2)

# Then draw match on top (same position, more opaque)
bars_match = ax1.bar(x_pos - bar_width/2, data_sorted['match_rate'], 
                     width=bar_width, 
                     label='Match Rate',
                     color=COLORS['match'], alpha=0.9, 
                     edgecolor='none', zorder=3)

# Add value labels
# for i, (cov, match) in enumerate(zip(data_sorted['coverage_rate'], 
#                                        data_sorted['match_rate'])):
#     # Coverage label (at coverage height)
#     ax1.text(i - bar_width/2, cov + 1, f'{cov:.0f}', 
#             ha='center', va='bottom', fontsize=5, color=COLORS['coverage'], fontweight='bold')
#     # Match label (at match height)
#     ax1.text(i - bar_width/2, match + 1, f'{match:.0f}', 
#             ha='center', va='bottom', fontsize=5, color=COLORS['match'], fontweight='bold')

# Create secondary y-axis for context length
ax2 = ax1.twinx()

# Plot context length as bars on the right side
bars_context = ax2.bar(x_pos + bar_width/2, data_sorted['total_doc_tokens'] / 1000,  # Convert to K tokens
                       width=bar_width, alpha=0.6,
                       color=COLORS['context'], label='Context Length',
                       edgecolor='none', zorder=1)

# Add value labels for context length
for i, tokens in enumerate(data_sorted['total_doc_tokens']):
    if tokens > 0:  # Only show non-zero values
        ax2.text(i + bar_width/2, tokens / 1000 + data_sorted['total_doc_tokens'].max() / 1000 * 0.02, 
                f'{int(tokens/1000)}', 
                ha='center', va='bottom', fontsize=5, 
                color=COLORS['context'], fontweight='bold')

# Styling for left y-axis (rates)
# ax1.set_xlabel('Survey', fontweight='bold', fontsize=9)
ax1.set_ylabel('Rate (%)', fontweight='bold', fontsize=9, color='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(data_sorted['survey'], fontsize=7, rotation=45, ha='right')
ax1.set_ylim(0, 108)
ax1.tick_params(axis='y', labelsize=8, colors='black')
ax1.grid(True, axis='y', alpha=0.3, linestyle='--', zorder=0)
ax1.set_axisbelow(True)

# Styling for right y-axis (context length)
ax2.set_ylabel('Context (K tokens)', fontweight='bold', fontsize=9, color=COLORS['context'])
ax2.tick_params(axis='y', labelsize=8, labelcolor=COLORS['context'])
ax2.set_ylim(0, 1080)  # Fixed range 0-1000 for alignment

# Grid
# ax1.grid(True, axis='both', linestyle=':', linewidth=0.6, alpha=0.6, zorder=0)

# box
for spine in ax1.spines.values():
    spine.set_color('#CCCCCC')
    spine.set_alpha(0.8)
    spine.set_linewidth(0.8)
for spine in ax2.spines.values():
    spine.set_color('#CCCCCC')
    spine.set_alpha(0.8)
    spine.set_linewidth(0.8)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=1.0, fontsize=7, frameon=True)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "survey_comparison.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ Saved: {output_path}")

plt.show()
