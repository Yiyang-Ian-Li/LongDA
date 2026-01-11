"""Simplified evaluation metrics for benchmark answers."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils import normalize_answer_value

EPSILON = 1e-10
MATCH_TOLERANCE = 0.05  # 5% relative tolerance


def _compare_values(agent_val: Any, truth_val: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compare agent answer with ground truth and return (abs_error, pct_error, match_score).
    Handles single numbers and lists of numbers.
    Returns (None, None, None) if comparison is not possible.
    
    For lists, match_score is the fraction of elements that match within tolerance.
    """
    # Handle single number comparison
    if isinstance(agent_val, (int, float)) and isinstance(truth_val, (int, float)):
        agent_num = float(agent_val)
        truth_num = float(truth_val)
        
        if math.isnan(agent_num) or math.isnan(truth_num):
            return None, None, None
        
        abs_error = abs(agent_num - truth_num)
        pct_error = None
        if abs(truth_num) > EPSILON:
            pct_error = abs((agent_num - truth_num) / truth_num) * 100
        
        # Use relative tolerance (5% of truth value), with fallback to absolute tolerance for small values
        tolerance = max(abs(truth_num) * MATCH_TOLERANCE, 1.0)
        match_score = 1.0 if abs_error <= tolerance else 0.0
        
        return abs_error, pct_error, match_score
    
    # Handle list comparison
    if isinstance(agent_val, list) and isinstance(truth_val, list):
        if len(agent_val) != len(truth_val):
            return None, None, None
        
        abs_errors = []
        pct_errors = []
        matches = []
        
        for a_item, t_item in zip(agent_val, truth_val):
            if not isinstance(a_item, (int, float)) or not isinstance(t_item, (int, float)):
                continue
            
            a_num = float(a_item)
            t_num = float(t_item)
            
            if math.isnan(a_num) or math.isnan(t_num):
                continue
            
            abs_err = abs(a_num - t_num)
            abs_errors.append(abs_err)
            
            # Use relative tolerance (5% of truth value), with fallback to absolute tolerance for small values
            tolerance = max(abs(t_num) * MATCH_TOLERANCE, 1.0)
            matches.append(1.0 if abs_err <= tolerance else 0.0)
            
            if abs(t_num) > EPSILON:
                pct_errors.append(abs((a_num - t_num) / t_num) * 100)
        
        if not abs_errors:
            return None, None, None
        
        avg_abs = float(np.mean(abs_errors))
        avg_pct = float(np.mean(pct_errors)) if pct_errors else None
        avg_match = float(np.mean(matches))
        
        return avg_abs, avg_pct, avg_match
    
    return None, None, None


def compute_row_metrics(my_answer: Any, ground_truth: Any) -> Dict[str, Any]:
    """
    Compute metrics for a single row: coverage, match, abs_error, pct_error.
    
    Args:
        my_answer: Agent's answer (already normalized)
        ground_truth: Ground truth answer (already normalized)
    
    Returns:
        Dict with keys: coverage (0/1), match (0-1 float), abs_error (float or None), pct_error (float or None)
        For lists, match is averaged: e.g., 2 correct out of 3 = 0.67
    """
    # Check coverage
    if my_answer is None or (isinstance(my_answer, float) and math.isnan(my_answer)):
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    if ground_truth is None or (isinstance(ground_truth, float) and math.isnan(ground_truth)):
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    # Try to compare values
    abs_error, pct_error, match_score = _compare_values(my_answer, ground_truth)
    
    if abs_error is None:
        # Could not compare
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    # Successfully compared
    return {
        "coverage": 1,
        "match": match_score,
        "abs_error": abs_error,
        "pct_error": pct_error,
    }


def evaluate_answers(answers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate agent answers against ground truth and add metric columns.
    
    Args:
        answers_df: DataFrame with columns including 'answer' (ground truth), 
                   'my_answer' (agent answer), 'q_id'
    
    Returns:
        DataFrame with added columns: coverage, match, abs_error, pct_error
    """
    # Create a copy to avoid modifying the original
    df = answers_df.copy()
    
    # Initialize metric columns
    df["coverage"] = 0
    df["match"] = 0.0
    df["abs_error"] = None
    df["pct_error"] = None
    
    # Compute metrics for each row
    for idx, row in df.iterrows():
        ground_truth = normalize_answer_value(row.get("answer"))
        my_answer = row.get("my_answer")
        
        # my_answer should already be normalized when stored, but normalize again to be safe
        if my_answer is not None and not isinstance(my_answer, (int, float, list, dict)):
            my_answer = normalize_answer_value(my_answer)
        
        metrics = compute_row_metrics(my_answer, ground_truth)
        
        df.at[idx, "coverage"] = metrics["coverage"]
        df.at[idx, "match"] = metrics["match"]
        df.at[idx, "abs_error"] = metrics["abs_error"]
        df.at[idx, "pct_error"] = metrics["pct_error"]
    
    return df


def generate_evaluation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate evaluation summary statistics as a dictionary."""
    # Overall metrics
    total_questions = len(df)
    coverage_rate = df["coverage"].mean() * 100
    covered_count = df["coverage"].sum()
    
    # Match rate: sum of all match scores divided by total questions
    match_rate = df["match"].sum() / total_questions * 100
    
    covered_df = df[df["coverage"] == 1]
    avg_abs_error = covered_df["abs_error"].mean() if len(covered_df) > 0 else None
    avg_pct_error = covered_df["pct_error"].mean() if len(covered_df) > 0 else None
    
    summary = {
        "overall": {
            "total_questions": int(total_questions),
            "covered": int(covered_count),
            "coverage_rate": float(coverage_rate),
            "match_rate": float(match_rate),
            "mean_absolute_error": float(avg_abs_error) if not pd.isna(avg_abs_error) else None,
            "mean_percentage_error": float(avg_pct_error) if not pd.isna(avg_pct_error) else None,
        }
    }
    
    # Metrics by source
    if "source" in df.columns:
        summary["by_source"] = {}
        for source, group in df.groupby("source"):
            n = len(group)
            cov = group["coverage"].mean() * 100
            # Match rate: sum of all match scores divided by total questions in this source
            mat = group["match"].sum() / n * 100
            covered_group = group[group["coverage"] == 1]
            mae = covered_group["abs_error"].mean() if len(covered_group) > 0 else None
            mpe = covered_group["pct_error"].mean() if len(covered_group) > 0 else None
            
            summary["by_source"][source] = {
                "n": int(n),
                "coverage": float(cov),
                "match_rate": float(mat),
                "mean_absolute_error": float(mae) if not pd.isna(mae) else None,
                "mean_percentage_error": float(mpe) if not pd.isna(mpe) else None,
            }
    
    return summary


def print_evaluation_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the evaluation."""
    summary = generate_evaluation_summary(df)
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Overall metrics
    overall = summary["overall"]
    total = overall["total_questions"]
    covered = overall["covered"]
    cov_rate = overall["coverage_rate"]
    match_rate = overall["match_rate"]
    mae = overall["mean_absolute_error"]
    mpe = overall["mean_percentage_error"]
    
    print(f"\nOverall Metrics (n={total}):")
    print(f"  Coverage Rate: {cov_rate:.2f}% ({covered}/{total})")
    print(f"  Match Rate: {match_rate:.2f}%")
    print(f"  Mean Absolute Error: {mae:.4f}" if mae is not None else "  Mean Absolute Error: N/A")
    print(f"  Mean Percentage Error: {mpe:.2f}%" if mpe is not None else "  Mean Percentage Error: N/A")
    
    # Metrics by source
    if "by_source" in summary:
        print("\nMetrics by Source:")
        print("-" * 80)
        for source, metrics in summary["by_source"].items():
            n = metrics["n"]
            cov = metrics["coverage"]
            mat = metrics["match_rate"]
            mae = metrics["mean_absolute_error"]
            mpe = metrics["mean_percentage_error"]
            
            print(f"\n  {source} (n={n}):")
            print(f"    Coverage: {cov:.2f}%")
            print(f"    Match Rate: {mat:.2f}%")
            print(f"    MAE: {mae:.4f}" if mae is not None else "    MAE: N/A")
            print(f"    MPE: {mpe:.2f}%" if mpe is not None else "    MPE: N/A")

    print("\n" + "=" * 80)
