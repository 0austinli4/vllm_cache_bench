#!/usr/bin/env python3
"""
Compute statistics for budget overruns from verification CSV reports.

This script analyzes cases where tokens_used > tokens_allocated and computes:
- Mean Squared Error (MSE) for overruns
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Percentage Overrun
- Count and percentage of overrun cases
"""

import csv
import argparse
import math
from pathlib import Path


def compute_overrun_stats(csv_file):
    """
    Calculate statistics for budget overruns.

    Args:
        csv_file: Path to verification CSV with columns:
                 tokens_used, tokens_allocated, followed_budget

    Returns:
        Dictionary with overrun statistics
    """
    overruns = []
    total_problems = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_problems += 1

            # Skip rows without budget tracking
            if 'tokens_used' not in row or 'tokens_allocated' not in row:
                continue
            if not row['tokens_used'] or not row['tokens_allocated']:
                continue

            tokens_used = int(row['tokens_used'])
            tokens_allocated = int(row['tokens_allocated'])

            # Only consider cases where budget was exceeded
            if tokens_used > tokens_allocated:
                overrun = tokens_used - tokens_allocated
                overruns.append({
                    'problem_id': row.get('problem_id', ''),
                    'tokens_used': tokens_used,
                    'tokens_allocated': tokens_allocated,
                    'overrun': overrun,
                    'overrun_pct': (overrun / tokens_allocated) * 100 if tokens_allocated > 0 else 0
                })

    if not overruns:
        return {
            'num_overruns': 0,
            'num_total': total_problems,
            'overrun_rate': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'mean_pct_overrun': 0.0
        }

    # Calculate statistics
    num_overruns = len(overruns)

    # Mean Squared Error
    mse = sum(o['overrun'] ** 2 for o in overruns) / num_overruns

    # Root Mean Squared Error
    rmse = math.sqrt(mse)

    # Mean Absolute Error
    mae = sum(o['overrun'] for o in overruns) / num_overruns

    # Mean Percentage Overrun
    mean_pct_overrun = sum(o['overrun_pct'] for o in overruns) / num_overruns

    return {
        'num_overruns': num_overruns,
        'num_total': total_problems,
        'overrun_rate': (num_overruns / total_problems) * 100 if total_problems > 0 else 0,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mean_pct_overrun': mean_pct_overrun,
        'overruns': overruns  # Include detailed list
    }


def print_overrun_stats(stats):
    """Pretty print overrun statistics."""
    print(f"\n{'='*60}")
    print(f"BUDGET OVERRUN STATISTICS")
    print(f"{'='*60}")
    print(f"Total problems:          {stats['num_total']}")
    print(f"Budget overruns:         {stats['num_overruns']} ({stats['overrun_rate']:.1f}%)")
    print()
    print(f"Mean Absolute Error (MAE):       {stats['mae']:.2f} tokens")
    print(f"Mean Squared Error (MSE):        {stats['mse']:.2f}")
    print(f"Root Mean Squared Error (RMSE):  {stats['rmse']:.2f} tokens")
    print(f"Mean Percentage Overrun:         {stats['mean_pct_overrun']:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute budget overrun statistics from verification CSV"
    )
    parser.add_argument("--csv-file", required=True,
                        help="Path to verification CSV file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed overrun information for each problem")

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        exit(1)

    stats = compute_overrun_stats(args.csv_file)
    print_overrun_stats(stats)

    if args.verbose and stats['overruns']:
        print(f"\nDETAILED OVERRUN BREAKDOWN:")
        print(f"{'Problem ID':<15} {'Used':<10} {'Allocated':<12} {'Overrun':<10} {'Overrun %':<12}")
        print("-" * 60)
        for o in stats['overruns']:
            print(f"{o['problem_id']:<15} {o['tokens_used']:<10} {o['tokens_allocated']:<12} "
                  f"{o['overrun']:<10} {o['overrun_pct']:<12.1f}%")
