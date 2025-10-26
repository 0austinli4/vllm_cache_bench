#!/usr/bin/env python3
"""
analyze_results.py
------------------
Analyze and summarize the results from exp.json to track accuracy and token usage trends.

Usage:
  python analyze_results.py [results_dir]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(exp_json_path):
    """Load results from exp.json file."""
    with open(exp_json_path, 'r') as f:
        return json.load(f)


def analyze_results(results):
    """Analyze results and compute summary statistics."""
    if not results:
        print("No results found in exp.json")
        return

    # Group by budget mode
    by_mode = defaultdict(list)
    for result in results:
        mode = result.get('budget_mode', 'unknown')
        by_mode[mode].append(result)

    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    for mode, runs in sorted(by_mode.items()):
        print(f"\n{mode.upper()} MODE:")
        print("-" * 80)

        for i, run in enumerate(runs, 1):
            accuracy = run.get('accuracy', 0) * 100
            correct = run.get('correct', 0)
            total = run.get('total', 0)
            avg_tokens = run.get('avg_tokens_per_problem', 0)
            total_tokens = run.get('total_output_tokens', 0)
            num_prompts = run.get('num_prompts', 0)

            print(f"\nRun {i}:")
            print(f"  Prompts:           {num_prompts}")
            print(f"  Accuracy:          {correct}/{total} ({accuracy:.1f}%)")
            print(f"  Total tokens:      {total_tokens}")
            print(f"  Avg tokens/prob:   {avg_tokens:.1f}")

            if 'budgets_file' in run:
                print(f"  Budget file:       {Path(run['budgets_file']).name}")

        # Compute averages for this mode
        if runs:
            avg_accuracy = sum(r.get('accuracy', 0) for r in runs) / len(runs) * 100
            avg_tokens_per_prob = sum(r.get('avg_tokens_per_problem', 0) for r in runs) / len(runs)

            print(f"\n{mode.upper()} AVERAGES (across {len(runs)} runs):")
            print(f"  Avg accuracy:      {avg_accuracy:.1f}%")
            print(f"  Avg tokens/prob:   {avg_tokens_per_prob:.1f}")

    print("\n" + "="*80)


def compare_budgets(results):
    """Compare accuracy vs token usage across different budget levels."""
    budgeted = [r for r in results if r.get('budget_mode') == 'budgeted']
    unbounded = [r for r in results if r.get('budget_mode') == 'unbounded']

    if not budgeted or not unbounded:
        return

    print("\n" + "="*80)
    print("BUDGET IMPACT ANALYSIS")
    print("="*80)

    unbounded_acc = sum(r.get('accuracy', 0) for r in unbounded) / len(unbounded) * 100
    unbounded_tokens = sum(r.get('avg_tokens_per_problem', 0) for r in unbounded) / len(unbounded)

    print(f"\nUnbounded baseline:")
    print(f"  Accuracy:          {unbounded_acc:.1f}%")
    print(f"  Avg tokens/prob:   {unbounded_tokens:.1f}")

    for run in budgeted:
        acc = run.get('accuracy', 0) * 100
        tokens = run.get('avg_tokens_per_problem', 0)
        acc_delta = acc - unbounded_acc
        tokens_reduction = (1 - tokens / unbounded_tokens) * 100 if unbounded_tokens > 0 else 0

        budget_file = Path(run.get('budgets_file', 'unknown')).name

        print(f"\n{budget_file}:")
        print(f"  Accuracy:          {acc:.1f}% ({acc_delta:+.1f}%)")
        print(f"  Avg tokens/prob:   {tokens:.1f} ({tokens_reduction:.1f}% reduction)")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results from exp.json")
    parser.add_argument("results_dir", nargs='?', help="Path to results directory (default: auto-detect from constants)")
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Try to import DIR from constants
        try:
            from constants import DIR
            results_dir = Path(DIR)
        except ImportError:
            print("Error: Could not determine results directory. Please specify it as an argument.")
            sys.exit(1)

    exp_json = results_dir / "exp.json"

    if not exp_json.exists():
        print(f"Error: {exp_json} not found")
        sys.exit(1)

    results = load_results(exp_json)
    analyze_results(results)
    compare_budgets(results)


if __name__ == "__main__":
    main()
