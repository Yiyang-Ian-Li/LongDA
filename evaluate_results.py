"""
Standalone evaluation script for re-evaluating benchmark results from CSV.
Uses the same evaluation logic as main.py.

Usage:
    python evaluate_results.py [--csv path_to_answers_progress.csv]
    
Example:
    python evaluate_results.py --csv results/my_agent/20251202T063934Z/answers_progress.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from metric import evaluate_answers, generate_evaluation_summary, print_evaluation_summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate benchmark results from answers_progress.csv file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/my_agent/20251222T071120Z_gpt-5/answers_progress.csv",
        help="Path to the answers_progress.csv file."
    )
    args = parser.parse_args()
    
    csv_path = args.csv
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        print(f"\nUsage: python evaluate_results.py --csv <path_to_answers_progress.csv>")
        return
    
    # Load the answers progress CSV
    print(f"Loading answers from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Re-evaluate answers with current metric logic
    print("Re-evaluating answers with current metrics...")
    df = evaluate_answers(df)
    
    # Save the re-evaluated results
    output_path = Path(csv_path).parent / "evaluated_answers.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved re-evaluated results to {output_path}")
    
    # Generate and save evaluation summary
    summary = generate_evaluation_summary(df)
    summary_path = Path(csv_path).parent / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_path}\n")
    
    # Print evaluation summary using the same function as main.py
    print_evaluation_summary(df)


if __name__ == "__main__":
    main()
