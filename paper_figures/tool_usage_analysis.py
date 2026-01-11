"""
Tool Usage Trajectory Analysis
Analyzes tool call patterns across agent execution steps to visualize
which tools are used at different stages of problem-solving.

This script processes message logs from the DA-Benchmark results to:
1. Extract tool usage sequences from agent execution traces
2. Normalize the temporal progress (0-100%) across different trajectory lengths
3. Create heatmaps showing the probability distribution of tool usage over time
4. Compare tool usage patterns across different models

Key insights from the visualization:
- Early stages (0-20%): Information gathering tools (read_doc, retriever, search_doc, prompt)
- Middle stages (20-60%): Mixed documentation lookup and code execution
- Late stages (60-100%): Answer submission and code saving

Generated figures:
- tool_usage_gpt5.pdf: Single model tool usage heatmap
- tool_usage_comparison.pdf: Multi-model comparison with 4 models
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style (no grid, clean look like reference)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    # 'font.weight': 'bold',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})

# Define tool list (ordered by typical usage sequence)
TOOLS = [
    'prompt',
    'notes', 
    'read_doc',
    'search_doc',
    'retriever',
    'answer',
    'save_code'
]

# Shorter names for display
TOOL_DISPLAY_NAMES = {
    'prompt': 'prompt',
    'notes': 'note',
    'read_doc': 'read',
    'search_doc': 'search',
    'retriever': 'retr.',
    'answer': 'answer',
    'save_code': 'save',
    'code': 'code'
}

def extract_tool_trajectories(results_dir):
    """
    Extract tool usage trajectories from message files.
    
    Args:
        results_dir: Path to results directory (e.g., results/20251225_192048_gpt-5)
        
    Returns:
        List of trajectories, where each trajectory is a list of (step, tool) tuples
    """
    messages_dir = Path(results_dir) / "messages"
    trajectories = []
    
    for msg_file in sorted(messages_dir.glob("*.json")):
        with open(msg_file, 'r') as f:
            data = json.load(f)
        
        trajectory = []
        messages = data.get('messages', [])
        
        # Skip first element (task description)
        for i, msg_str in enumerate(messages[1:], start=1):
            if not isinstance(msg_str, str):
                continue
            
            # Extract ALL code_action fields (there may be multiple steps in one message)
            # Find all occurrences of 'code_action'
            pos = 0
            while True:
                code_action_pos = msg_str.find("'code_action':", pos)
                if code_action_pos == -1:
                    break
                
                # Extract this code_action content
                start = code_action_pos + len("'code_action':")
                # Find the end (next field)
                end = msg_str.find("'observations':", start)
                if end == -1:
                    end = msg_str.find("'model_output':", start)
                if end == -1:
                    end = msg_str.find("', '", start)  # End of this field
                
                if end != -1:
                    code_action = msg_str[start:end]
                    
                    # Count tool calls in this code_action
                    step_tools = []
                    
                    step_tools.extend(['prompt'] * code_action.count('prompt('))
                    step_tools.extend(['notes'] * code_action.count('notes('))
                    step_tools.extend(['read_doc'] * code_action.count('read_doc('))
                    step_tools.extend(['search_doc'] * code_action.count('search_doc('))
                    step_tools.extend(['retriever'] * code_action.count('retriever('))
                    step_tools.extend(['answer'] * code_action.count('answer('))
                    step_tools.extend(['save_code'] * code_action.count('save_code('))
                    
                    # Add all tool calls
                    for tool in step_tools:
                        trajectory.append((i, tool))
                    
                    pos = end
                else:
                    break
        
        if trajectory:
            trajectories.append(trajectory)
    
    return trajectories

def normalize_trajectories(trajectories):
    """
    Normalize trajectory positions to 0-1 range (progress).
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Dict mapping (progress_bin, tool) to count
    """
    num_bins = 20  # Discretize progress into 20 bins
    heatmap_data = {tool: np.zeros(num_bins) for tool in TOOLS + ['code']}
    
    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue
        
        max_step = max(step for step, _ in trajectory)
        if max_step == 0:
            max_step = 1
        
        for step, tool in trajectory:
            normalized_pos = (step) / (max_step)
            # print(normalized_pos)
            bin_idx = min(int(normalized_pos * num_bins), num_bins - 1)
            # print(bin_idx, tool)
            heatmap_data[tool][bin_idx] += 1

    return heatmap_data, num_bins

def plot_tool_usage_heatmap(heatmap_data, num_bins, model_name, output_path):
    """
    Create heatmap showing tool usage probability vs normalized progress.
    
    Args:
        heatmap_data: Dict mapping tool to count array
        num_bins: Number of progress bins
        model_name: Name of the model
        output_path: Path to save figure
    """
    # Filter out tools with no usage and get display names
    active_tools = [tool for tool in TOOLS + ['code'] if heatmap_data[tool].sum() > 0]
    display_names = [TOOL_DISPLAY_NAMES.get(tool, tool) for tool in active_tools]
    
    # Create matrix for heatmap (normalize to probabilities per bin)
    matrix = np.array([heatmap_data[tool] for tool in active_tools])
    
    # Normalize each column (progress bin) to show probability distribution
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / col_sums
    
    # Create figure (match reference style)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Create heatmap
    im = ax.imshow(matrix_normalized, aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_yticks(np.arange(len(active_tools)))
    ax.set_yticklabels(display_names)
    
    # Set x-axis to show progress percentage (0%, 50%, 100% like reference)
    ax.set_xticks([0, num_bins//2, num_bins-1])
    ax.set_xticklabels(['0%', '50%', '100%'])
    
    # Labels (match reference style)
    # ax.set_xlabel('Normalized progress (0% → 100%)', fontdict=dict(weight='bold'))
    ax.set_ylabel('Tool', fontdict=dict(weight='bold'))
    ax.set_title(model_name, fontdict=dict(weight='bold'))
    
    # Colorbar (match reference parameters)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved tool usage heatmap to {output_path}")
    plt.close()

def analyze_tool_usage_multiple_models(results_base_dir, model_configs):
    """
    Analyze tool usage for multiple models and create comparison plot.
    
    Args:
        results_base_dir: Base directory containing results folders
        model_configs: List of (folder_name, display_name) tuples
    """
    # Calculate grid layout
    ncols = min(3, len(model_configs))
    nrows = int(np.ceil(len(model_configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(5*ncols, 4*nrows), 
                             dpi=300, squeeze=False)
    
    for idx, (folder_name, display_name) in enumerate(model_configs):
        results_dir = Path(results_base_dir) / folder_name
        
        print(f"Processing {display_name}...")
        trajectories = extract_tool_trajectories(results_dir)
        print(f"  Found {len(trajectories)} trajectories")
        
        # Load actual tool usage counts from tool_usage.json
        tool_usage_file = results_dir / "tool_usage.json"
        if tool_usage_file.exists():
            with open(tool_usage_file) as f:
                tool_usage = json.load(f)
            print(f"  Tool counts: Prompt={tool_usage.get('prompt', 0)}, "
                  f"Read={tool_usage.get('read_doc', 0)}, "
                  f"Search={tool_usage.get('search_doc', 0)}, "
                  f"Retr.={tool_usage.get('retriever', 0)}, "
                  f"Note={tool_usage.get('notes', 0)}, "
                  f"Answer={tool_usage.get('answer', 0)}, "
                  f"Save={tool_usage.get('save_code', 0)}")
        
        heatmap_data, num_bins = normalize_trajectories(trajectories)
        
        # Filter active tools and get display names
        active_tools = [tool for tool in TOOLS + ['code'] if heatmap_data[tool].sum() > 0]
        display_names = [TOOL_DISPLAY_NAMES.get(tool, tool) for tool in active_tools]
        # print(heatmap_data)
        
        # Create matrix
        matrix = np.array([heatmap_data[tool] for tool in active_tools])
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        matrix_normalized = matrix / col_sums
        # print(matrix_normalized)
        # exit()
        
        # Plot on subplot (match reference style)
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        im = ax.imshow(matrix_normalized, aspect='auto', interpolation='nearest')
        
        ax.set_yticks(np.arange(len(active_tools)))
        ax.set_yticklabels(display_names, fontsize=16)
        
        # x-axis: 0%, 50%, 100% like reference
        ax.set_xticks([0, num_bins//2, num_bins-1])
        ax.set_xticklabels(['0%', '50%', '100%'], fontsize=16)
        
       
        # ax.set_xlabel('Normalized progress (0% → 100%)', fontdict=dict(weight='bold'))
        # ax.set_ylabel('Tool', fontdict=dict(weight='bold'))
        ax.set_title(display_name, fontdict=dict(weight='bold', fontsize=16))
        
        # Colorbar for each subplot (like reference)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(model_configs), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = "paper_figures/tool_usage_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Single model example
    results_dir = "results/20251225_192048_gpt-5"
    trajectories = extract_tool_trajectories(results_dir)
    print(f"Extracted {len(trajectories)} trajectories from {results_dir}")
    
    if len(trajectories) > 0:
        heatmap_data, num_bins = normalize_trajectories(trajectories)
        plot_tool_usage_heatmap(heatmap_data, num_bins, "GPT-5", 
                               "paper_figures/tool_usage_gpt5.pdf")
    
    # Multiple models comparison (6 models to match table)
    model_configs = [
        # ("20251225_192048_gpt-5", "GPT-5"),
        # ("20251225_184241_deepseek-chat", "DeepSeek-V3.2"),
        # ("20251225_180726_qwen3-235b-a22b-instruct-2507", "Qwen3-235B-A22B-Instruct"),
        # ("20251226_011036_gpt-5", "GPT-5 (High)"),
        # ("20251226_005439_deepseek-reasoner", "DeepSeek-V3.2-Thinking"),
        # ("20251225_001120_qwen3-coder-480b-a35b-instruct", "Qwen3-Coder-480B-A35B-Instruct"),
        ("20251226_011036_gpt-5", "GPT-5"),
        ("20251225_184241_deepseek-chat", "DeepSeek-V3.2"),
        ("20251225_180726_qwen3-235b-a22b-instruct-2507", "Qwen3-235B-A22B-Instruct"),
    ]
    
    analyze_tool_usage_multiple_models("results", model_configs)
