"""
DA-Benchmark Main Evaluation Script

This script runs the DA-Benchmark evaluation on configured LLM agents.
It processes benchmark questions grouped by survey and source, executes
the agent on each block, and generates comprehensive evaluation metrics.

Usage:
    python main.py --config_file configs/your_config.yaml
"""

import argparse
import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from my_agent import MyAgentBaseline
from metric import evaluate_answers, generate_evaluation_summary, print_evaluation_summary
from utils import build_answer_entry, get_data_and_doc_paths, normalize_answer_value


def _iter_benchmark_blocks(benchmark_df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """
    Group benchmark questions by survey and source, yielding block dictionaries.
    
    Args:
        benchmark_df: DataFrame containing benchmark questions
        
    Yields:
        Dictionary containing block information including survey, source, questions,
        answer structures, data paths, and documentation paths
    """
    grouped = benchmark_df.groupby(["survey", "source"], sort=False)
    for (survey, source), group in grouped:
        data_paths, doc_paths = get_data_and_doc_paths(survey)
        yield {
            "survey": survey,
            "source": source,
            "questions": group["query"].tolist(),
            "answer_structures": group["answer_structure"].tolist(),
            "additional_infos": group["additional_info"].tolist(),
            "data_paths": data_paths,
            "doc_paths": doc_paths,
            "block_id": f"{survey}_{source}",
        }

def _load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing configuration file: {config_path.resolve()}"
        )
    with open(config_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return data


def _execute_benchmark(
    runner,
    benchmark_df: pd.DataFrame,
    results_dir: Path,
) -> Optional[pd.DataFrame]:
    if runner is None:
        return None

    # Start timing
    start_time = time.time()
    block_timings = {}
    block_metrics_dict = {}  # Store metrics for each block

    # Create timestamped directory for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f'{run_timestamp}_{runner.config["model"]["id"].split("/")[-1]}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create messages subdirectory
    messages_dir = run_dir / "messages"
    messages_dir.mkdir(exist_ok=True)
    
    # Create answers DataFrame for real-time monitoring
    answers_df = benchmark_df.copy()
    answers_df["my_answer"] = None
    
    # Add q_id column for easy matching
    grouped = benchmark_df.groupby(["survey", "source"], sort=False)
    q_ids = []
    for (survey, source), group in grouped:
        for i in range(len(group)):
            q_ids.append(f"{survey}_{source}_q{i+1}")
    answers_df["q_id"] = q_ids
    
    # Save initial empty answers DataFrame
    answers_progress_path = run_dir / "answers_progress.csv"
    answers_df.to_csv(answers_progress_path, index=False)
    print(f"Initialized answers progress file: {answers_progress_path}")
    
    # Run each block and update answers DataFrame
    for block in _iter_benchmark_blocks(benchmark_df):
        block_id = block["block_id"]
        print(f"\n{'='*80}")
        print(f"Starting block: {block_id}")
        print(f"{'='*80}")
        
        # Time this block
        block_start = time.time()
        
        # run_block returns both answers and metrics
        result = runner.run_block(block, run_timestamp=run_timestamp)
        block_answers = result.get("block_answers", {})
        block_metrics = result.get("block_metrics", {})
        
        block_elapsed = time.time() - block_start
        block_timings[block_id] = block_elapsed
        
        # Store block metrics
        block_metrics_dict[block_id] = {
            "state": block_metrics.get("state", "unknown"),
            "survey": block["survey"],
            "source": block["source"],
            "tokens": block_metrics.get("tokens", {}),
            "timing": block_metrics.get("timing", {}),
            "steps": block_metrics.get("steps", 0),
            "num_questions": block_metrics.get("num_questions", 0),
            "answers_recorded": block_metrics.get("answers_recorded", 0),
        }
        
        # Save block messages to separate file
        if "messages" in block_metrics:
            messages_file = messages_dir / f"{block_id}.json"
            
            # Convert ChatMessage objects to dict for JSON serialization
            messages_serializable = []
            for msg in block_metrics["messages"]:
                messages_serializable.append(str(msg))
            
            with open(messages_file, "w") as f:
                json.dump({
                    "block_id": block_id,
                    "survey": block["survey"],
                    "source": block["source"],
                    "messages": messages_serializable
                }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Completed block: {block_id}")
        print(f"  Time: {block_elapsed:.2f}s ({block_elapsed/60:.2f}m)")
        print(f"  Steps: {block_metrics.get('steps', 0)}")
        print(f"  Tokens: {block_metrics.get('tokens', {}).get('total_tokens', 0)}")
        print(f"{'='*80}\n")
        
        # Update CSV with the returned block answers (ensures synchronization)
        for q_id, answer_data in block_answers.items():
            mask = answers_df["q_id"] == q_id
            if mask.any():
                raw_answer = answer_data.get("raw_answer", "")
                # Parse answer (should be number or list string from validated AnswerTool)
                if isinstance(raw_answer, str):
                    try:
                        answer_val = ast.literal_eval(raw_answer)
                    except (ValueError, SyntaxError):
                        # Fallback: try to extract numbers if format is wrong
                        answer_val = normalize_answer_value(raw_answer)
                else:
                    answer_val = raw_answer
                
                # Use .at for single row assignment to avoid dimension mismatch
                idx = answers_df[mask].index[0]
                answers_df.at[idx, "my_answer"] = answer_val
        
        # Save updated progress after each block (synchronized with JSON save)
        answers_df.to_csv(answers_progress_path, index=False)
        print(f"Updated answers progress: {answers_progress_path}")
    
    # Get final artifacts
    artifacts = runner.collect_artifacts()
    recorded_answers: Dict[str, Dict[str, str]] = artifacts.get("recorded_answers", {})

    if not recorded_answers:
        print("No answers were recorded by the agent. Skipping evaluation.")
        return None

    # Save artifacts
    code_snippets = artifacts.get("code_snippets", {})
    if code_snippets:
        code_path = run_dir / "codes.json"
        with open(code_path, "w") as file:
            json.dump(code_snippets, file, indent=2)
        print(f"Saved code snippets to {code_path.resolve()}")

    notes_store = artifacts.get("notes_store", {})
    filtered_notes = {key: value for key, value in notes_store.items() if value}
    if filtered_notes:
        notes_path = run_dir / "notes.json"
        with open(notes_path, "w") as file:
            json.dump(filtered_notes, file, indent=2)
        print(f"Saved notes to {notes_path.resolve()}")

    tool_usage = artifacts.get("tool_usage_counts", {})
    if tool_usage:
        usage_path = run_dir / "tool_usage.json"
        with open(usage_path, "w") as file:
            json.dump(tool_usage, file, indent=2)
        print(f"Saved tool usage stats to {usage_path.resolve()}")

    # Calculate totals
    total_elapsed = time.time() - start_time
    total_tokens = sum(m.get("tokens", {}).get("total_tokens", 0) for m in block_metrics_dict.values())
    total_input_tokens = sum(m.get("tokens", {}).get("input_tokens", 0) for m in block_metrics_dict.values())
    total_output_tokens = sum(m.get("tokens", {}).get("output_tokens", 0) for m in block_metrics_dict.values())
    total_steps = sum(m.get("steps", 0) for m in block_metrics_dict.values())
    successful_blocks = sum(1 for m in block_metrics_dict.values() if m.get("state") == "success")
    
    # Save run summary
    run_summary = {
        "run_info": {
            "timestamp": run_timestamp,
            "model": runner.config["model"]["id"],
            "total_blocks": len(block_metrics_dict),
            "successful_blocks": successful_blocks,
            "failed_blocks": len(block_metrics_dict) - successful_blocks,
        },
        "total_metrics": {
            "tokens": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
            },
            "timing": {
                "total_seconds": round(total_elapsed, 2),
                "total_minutes": round(total_elapsed / 60, 2),
                "total_hours": round(total_elapsed / 3600, 2),
                "average_block_seconds": round(sum(block_timings.values()) / len(block_timings), 2) if block_timings else 0,
            },
            "steps": {
                "total_steps": total_steps,
                "average_steps_per_block": round(total_steps / len(block_metrics_dict), 2) if block_metrics_dict else 0,
            },
        },
    }
    
    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w") as file:
        json.dump(run_summary, file, indent=2)
    print(f"Saved run summary to {summary_path.resolve()}")
    
    # Save block metrics
    block_metrics_path = run_dir / "block_metrics.json"
    with open(block_metrics_path, "w") as file:
        json.dump(block_metrics_dict, file, indent=2)
    print(f"Saved block metrics to {block_metrics_path.resolve()}")
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETED")
    print(f"  Runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f}m / {total_elapsed/3600:.2f}h)")
    print(f"  Total Tokens: {total_tokens:,} (input: {total_input_tokens:,}, output: {total_output_tokens:,})")
    print(f"  Total Steps: {total_steps}")
    print(f"  Blocks: {successful_blocks}/{len(block_metrics_dict)} successful")
    print(f"{'='*80}\n")

    # Evaluate answers using the new simplified approach
    print("\nEvaluating answers...")
    evaluated_df = evaluate_answers(answers_df)
    
    # Save evaluated DataFrame
    evaluated_path = run_dir / "evaluated_answers.csv"
    evaluated_df.to_csv(evaluated_path, index=False)
    print(f"Saved evaluated answers to {evaluated_path.resolve()}")
    
    # Generate and save evaluation summary
    summary = generate_evaluation_summary(evaluated_df)
    summary_path = run_dir / "evaluation_summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved evaluation summary to {summary_path.resolve()}")
    
    # Print summary to console
    print_evaluation_summary(evaluated_df)
    
    return evaluated_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data analysis agent benchmarks.")
    parser.add_argument(
        '--benchmark_file', type=str, default='benchmark/benchmark.csv',
        help='Path to the benchmark CSV file.'
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/my_agent.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    # Prepare benchmark data
    benchmark_path = Path(args.benchmark_file)
    benchmark_df = pd.read_csv(benchmark_path)
    benchmark_df.fillna("NA.", inplace=True)

    # Load config and create agent
    config = _load_config(Path(args.config_file))
    print(f"\n=== Running MyAgent Benchmark ===")
    results_dir = Path("results")
    
    runner = MyAgentBaseline(config)
    _execute_benchmark(
        runner=runner,
        benchmark_df=benchmark_df,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
