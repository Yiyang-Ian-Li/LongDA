"""
Figure: Match Rate vs Tolerance Sensitivity Analysis
Line plot showing how match rate varies with different tolerance levels.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_answer_value

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
TITLE = "Match Rate Sensitivity to Tolerance Threshold"

# ============================================================================
# DATA PREPARATION
# ============================================================================

EPSILON = 1e-10

def calculate_match_with_tolerance(agent_val, truth_val, tolerance):
    """Calculate if values match within given tolerance."""
    # Handle single number comparison
    if isinstance(agent_val, (int, float)) and isinstance(truth_val, (int, float)):
        agent_num = float(agent_val)
        truth_num = float(truth_val)
        
        if math.isnan(agent_num) or math.isnan(truth_num):
            return None
        
        abs_error = abs(agent_num - truth_num)
        tol_value = max(abs(truth_num) * tolerance, 1.0)
        return 1.0 if abs_error <= tol_value else 0.0
    
    # Handle list comparison
    if isinstance(agent_val, list) and isinstance(truth_val, list):
        if len(agent_val) != len(truth_val):
            return None
        
        matches = []
        for a_item, t_item in zip(agent_val, truth_val):
            if not isinstance(a_item, (int, float)) or not isinstance(t_item, (int, float)):
                continue
            
            a_num = float(a_item)
            t_num = float(t_item)
            
            if math.isnan(a_num) or math.isnan(t_num):
                continue
            
            abs_err = abs(a_num - t_num)
            tol_value = max(abs(t_num) * tolerance, 1.0)
            matches.append(1.0 if abs_err <= tol_value else 0.0)
        
        if not matches:
            return None
        
        return float(np.mean(matches))
    
    return None

def evaluate_with_tolerance(df, tolerance):
    """Evaluate match rate with a specific tolerance level."""
    matches = []
    
    for idx, row in df.iterrows():
        ground_truth = normalize_answer_value(row.get("answer"))
        my_answer = row.get("my_answer")
        
        # Parse my_answer if it's a string
        if isinstance(my_answer, str):
            try:
                # Try to parse as float
                my_answer = float(my_answer)
            except (ValueError, TypeError):
                # Try to evaluate as list
                try:
                    import ast
                    my_answer = ast.literal_eval(my_answer)
                except:
                    my_answer = None
        
        if my_answer is None or (isinstance(my_answer, float) and math.isnan(my_answer)):
            matches.append(0.0)
            continue
        
        if ground_truth is None or (isinstance(ground_truth, float) and math.isnan(ground_truth)):
            matches.append(0.0)
            continue
        
        match = calculate_match_with_tolerance(my_answer, ground_truth, tolerance)
        if match is None:
            matches.append(0.0)
        else:
            matches.append(match)
    
    return (sum(matches) / len(matches)) * 100 if matches else 0.0

# Load data from the three models
data_files = {
    'GPT-5': 'results/20251225_192048_gpt-5/answers_progress.csv',
    'DeepSeek-V3.2': 'results/20251225_020458_deepseek-chat/answers_progress.csv',
    'GPT-5 mini': 'results/20251224_234245_gpt-5-mini/answers_progress.csv',
    # 'Qwen3-Coder-480B': 'results/20251225_001120_qwen3-coder-480b-a35b-instruct/answers_progress.csv',
    'Kimi-K2': 'results/20251226_004310_kimi-k2-0905/answers_progress.csv',
}

base_dir = Path(__file__).parent.parent
tolerances = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
data = []

print("Loading data and calculating match rates for different tolerances...")
for model_name, file_path in data_files.items():
    full_path = base_dir / file_path
    print(f"Loading {model_name}: {full_path}")
    df = pd.read_csv(full_path)
    
    for tol in tolerances:
        match_rate = evaluate_with_tolerance(df, tol)
        data.append({
            'model': model_name,
            'tolerance': tol,
            'match_rate': match_rate
        })
        print(f"  Tolerance {tol:.2f}: {match_rate:.2f}%")

data = pd.DataFrame(data)
models = list(data_files.keys())

# ============================================================================
# PLOTTING
# ============================================================================
print(f"Generating: {TITLE}")

fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Get color palette that supports multiple models
colors = sns.color_palette("viridis", n_colors=len(models))

# Plot each model as a separate line
for i, model in enumerate(models):
    model_data = data[data['model'] == model].sort_values('tolerance')
    
    line = ax.plot(model_data['tolerance'], model_data['match_rate'], 
                   marker='o', linewidth=1.5, markersize=4,
                   color=colors[i], alpha=0.8)
    
    # Add label at the end of the line
    last_x = model_data['tolerance'].iloc[-1]
    last_y = model_data['match_rate'].iloc[-1]
    ax.text(last_x - 0.02, last_y + 2, model, 
            fontsize=7, va='bottom', ha='center',
            color=colors[i], fontweight='bold')

# Styling
ax.set_xlabel('Tolerance (%)', fontweight='bold', fontsize=9)
ax.set_ylabel('Match Rate (%)', fontweight='bold', fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
# Extend x-axis to make room for labels
ax.set_xlim(-0.005, max(data['tolerance']) + 0.02)
y_min = max(0, data['match_rate'].min() - 5)
y_max = min(100, data['match_rate'].max() + 10)
ax.set_ylim(y_min, y_max)
ax.tick_params(labelsize=8)

# Format x-axis as percentage
ax.set_xticks(data['tolerance'].unique())
ax.set_xticklabels([f'{int(t*100)}' for t in data['tolerance'].unique()])

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / "tolerance_sensitivity.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✓ Saved: {output_path}")

plt.show()
