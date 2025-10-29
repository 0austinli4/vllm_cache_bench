#!/usr/bin/env python3
"""
compare_runs.py
---------------
Compare results across multiple percentile runs and generate comprehensive analysis.

This script:
1. Reads verify_report.csv from each run directory (unbounded, p85, p65, p45, p25)
2. Extracts accuracy metrics for each run
3. Analyzes token budget adherence for budgeted runs
4. Generates summary report with comparison table
5. Outputs JSON file with all metrics

Expected directory structure:
    experiment_dir/
        unbounded/
            metrics/verify_report.csv
        p85/
            metrics/verify_report.csv
        p65/
            metrics/verify_report.csv
        ...

Usage:
    python compare_runs.py --experiment-dir results/model/AIME25 --output summary.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_verify_report(report_path: Path) -> Optional[Dict]:
    """
    Parse a verify_report.csv file and extract metrics.

    Returns:
        Dict with keys: total, correct, accuracy, budget_stats (if applicable), token_stats (for any run with tokens_used)
    """
    if not report_path.exists():
        return None

    total = 0
    correct = 0
    has_budget_data = False
    has_token_data = False

    # Budget tracking (for budgeted runs with allocation)
    tokens_used_total = 0
    tokens_allocated_total = 0
    budget_followed_count = 0
    budget_exceeded_count = 0
    budget_tracked_count = 0

    # Token tracking (for all runs, including unbounded)
    tokens_used_count = 0

    with open(report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Check if budget columns exist
        fieldnames = reader.fieldnames or []
        has_budget_data = 'tokens_used' in fieldnames and 'tokens_allocated' in fieldnames
        has_token_data = 'tokens_used' in fieldnames

        for row in reader:
            total += 1

            # Count correct answers
            if row.get('match', '').lower() == 'true':
                correct += 1

            # Track tokens used (for all runs, including unbounded)
            if has_token_data:
                tokens_used = row.get('tokens_used', '')
                if tokens_used:
                    try:
                        tokens_used_total += int(tokens_used)
                        tokens_used_count += 1
                    except (ValueError, TypeError):
                        pass

            # Track budget adherence if full budget data is present (budgeted runs only)
            if has_budget_data:
                tokens_used = row.get('tokens_used', '')
                tokens_allocated = row.get('tokens_allocated', '')
                followed_budget = row.get('followed_budget', '')

                if tokens_used and tokens_allocated:
                    try:
                        tokens_allocated_total += int(tokens_allocated)

                        if followed_budget != '':
                            budget_tracked_count += 1
                            if followed_budget.lower() == 'true':
                                budget_followed_count += 1
                            else:
                                budget_exceeded_count += 1
                    except (ValueError, TypeError):
                        pass

    accuracy = (correct / total * 100) if total > 0 else 0.0

    result = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }

    # Add token usage stats for any run that tracked tokens
    if has_token_data and tokens_used_count > 0:
        avg_tokens_used = tokens_used_total / tokens_used_count
        result['token_stats'] = {
            'tokens_used_total': tokens_used_total,
            'avg_tokens_used': avg_tokens_used,
            'problems_tracked': tokens_used_count
        }

    # Add budget stats for budgeted runs
    if has_budget_data and budget_tracked_count > 0:
        avg_tokens_allocated = tokens_allocated_total / budget_tracked_count
        adherence_rate = (budget_followed_count / budget_tracked_count * 100) if budget_tracked_count > 0 else 0.0
        usage_rate = (tokens_used_total / tokens_allocated_total * 100) if tokens_allocated_total > 0 else 0.0

        result['budget_stats'] = {
            'tokens_allocated_total': tokens_allocated_total,
            'avg_tokens_allocated': avg_tokens_allocated,
            'budget_followed': budget_followed_count,
            'budget_exceeded': budget_exceeded_count,
            'budget_tracked': budget_tracked_count,
            'adherence_rate': adherence_rate,
            'usage_rate': usage_rate
        }

    return result


def collect_run_data(experiment_dir: Path) -> Dict[str, Dict]:
    """
    Collect data from all runs in the experiment directory.

    Returns:
        Dict mapping run_name to metrics
    """
    runs = {}

    # Check for unbounded run
    unbounded_report = experiment_dir / 'unbounded' / 'metrics' / 'verify_report.csv'
    if not unbounded_report.exists():
        # Fall back to checking parent directory for older runs
        unbounded_parent = experiment_dir / 'unbounded'
        if unbounded_parent.is_dir():
            # Look for any verify_report*.csv in the unbounded directory
            verify_reports = list(unbounded_parent.glob('verify_report*.csv'))
            if verify_reports:
                # Use the most recently modified one
                unbounded_report = max(verify_reports, key=lambda p: p.stat().st_mtime)

    if unbounded_report.exists():
        data = parse_verify_report(unbounded_report)
        if data:
            runs['unbounded'] = data

    # Check for percentile runs (p85, p65, p45, p25, etc.)
    for run_dir in experiment_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('p') and run_dir.name[1:].isdigit():
            report = run_dir / 'metrics' / 'verify_report.csv'
            if not report.exists():
                # Fall back to checking parent directory for older runs
                verify_reports = list(run_dir.glob('verify_report*.csv'))
                if verify_reports:
                    # Use the most recently modified one
                    report = max(verify_reports, key=lambda p: p.stat().st_mtime)

            if report.exists():
                data = parse_verify_report(report)
                if data:
                    runs[run_dir.name] = data

    return runs


def print_comparison_table(runs: Dict[str, Dict]):
    """Print a formatted comparison table to console."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*100)

    # Sort runs: unbounded first, then percentiles in descending order
    def sort_key(name):
        if name == 'unbounded':
            return (0, 0)
        elif name.startswith('p'):
            try:
                return (1, -int(name[1:]))  # Negative to sort descending
            except ValueError:
                return (2, 0)
        return (3, 0)

    sorted_runs = sorted(runs.items(), key=lambda x: sort_key(x[0]))

    # Print header
    print(f"\n{'Run':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>8}  {'Budget Adherence':>18}  {'Avg Tokens':>15}")
    print("-"*100)

    # Print each run
    for run_name, data in sorted_runs:
        accuracy_str = f"{data['accuracy']:.1f}%"
        correct_total_str = f"{data['correct']}/{data['total']}"

        # Budget info
        budget_info = ""
        tokens_info = ""

        if 'budget_stats' in data:
            # Budgeted run - show budget adherence and tokens used/allocated
            bs = data['budget_stats']
            adherence_str = f"{bs['adherence_rate']:.1f}%"
            budget_info = f"{bs['budget_followed']}/{bs['budget_tracked']} ({adherence_str})"
            # Get actual tokens used from token_stats
            if 'token_stats' in data:
                ts = data['token_stats']
                tokens_info = f"{ts['avg_tokens_used']:.0f}/{bs['avg_tokens_allocated']:.0f}"
            else:
                tokens_info = f"?/{bs['avg_tokens_allocated']:.0f}"
        else:
            budget_info = "N/A"
            # Check if we have token usage data for unbounded runs
            if 'token_stats' in data:
                ts = data['token_stats']
                tokens_info = f"{ts['avg_tokens_used']:.0f} (unbounded)"
            elif run_name == 'unbounded':
                tokens_info = "unlimited"
            else:
                tokens_info = ""

        print(f"{run_name:<12} {accuracy_str:>10} {correct_total_str:>10} {data['total']:>8}  {budget_info:>18}  {tokens_info:>15}")

    print("="*100)

    # Print summary statistics
    if 'unbounded' in runs:
        baseline_accuracy = runs['unbounded']['accuracy']
        unbounded_tokens = None
        if 'token_stats' in runs['unbounded']:
            unbounded_tokens = runs['unbounded']['token_stats']['avg_tokens_used']
            print(f"\n📊 SUMMARY STATISTICS")
            print(f"   Baseline (unbounded) accuracy: {baseline_accuracy:.1f}%")
            print(f"   Baseline (unbounded) avg tokens per problem: {unbounded_tokens:.0f}\n")
        else:
            print(f"\n📊 SUMMARY STATISTICS")
            print(f"   Baseline (unbounded) accuracy: {baseline_accuracy:.1f}%")
            print(f"   ⚠️  Token usage not tracked for unbounded run\n")

        budgeted_runs = [(name, data) for name, data in sorted_runs if name != 'unbounded']

        if budgeted_runs:
            print(f"   {'Budget':<8} {'Accuracy Drop':>15} {'Token Savings':>18} {'Adherence':>12}")
            print(f"   {'-'*8} {'-'*15} {'-'*18} {'-'*12}")

            for run_name, data in budgeted_runs:
                accuracy_drop = baseline_accuracy - data['accuracy']
                accuracy_drop_str = f"{accuracy_drop:+.1f}%"

                token_savings_str = "unknown"
                adherence_str = "N/A"

                if 'budget_stats' in data:
                    bs = data['budget_stats']
                    adherence_str = f"{bs['adherence_rate']:.1f}%"

                    # Calculate actual token savings if we have both unbounded and budgeted token data
                    if unbounded_tokens and 'token_stats' in data:
                        ts = data['token_stats']
                        budgeted_tokens = ts['avg_tokens_used']
                        savings_pct = ((unbounded_tokens - budgeted_tokens) / unbounded_tokens) * 100
                        token_savings_str = f"{savings_pct:.1f}% ({budgeted_tokens:.0f}/{unbounded_tokens:.0f})"
                    else:
                        # Fallback: estimate based on percentile name
                        percentile = int(run_name[1:]) if run_name.startswith('p') else 0
                        estimated_savings = 100 - percentile
                        token_savings_str = f"~{estimated_savings}% (estimated)"

                print(f"   {run_name:<8} {accuracy_drop_str:>15} {token_savings_str:>18} {adherence_str:>12}")

    print()


def generate_summary_json(runs: Dict[str, Dict], output_path: Path):
    """Generate JSON summary file with all metrics."""
    summary = {
        'runs': runs,
        'metadata': {
            'total_runs': len(runs),
            'has_unbounded': 'unbounded' in runs
        }
    }

    # Add comparison metrics if unbounded exists
    if 'unbounded' in runs:
        baseline_accuracy = runs['unbounded']['accuracy']
        baseline_tokens = None
        if 'token_stats' in runs['unbounded']:
            baseline_tokens = runs['unbounded']['token_stats']['avg_tokens_used']
            summary['metadata']['baseline_avg_tokens'] = baseline_tokens

        comparisons = []

        for run_name, data in runs.items():
            if run_name != 'unbounded':
                comparison = {
                    'run_name': run_name,
                    'accuracy': data['accuracy'],
                    'accuracy_drop': baseline_accuracy - data['accuracy'],
                    'correct': data['correct'],
                    'total': data['total']
                }

                if 'token_stats' in data:
                    comparison['token_stats'] = data['token_stats']
                    # Calculate actual token savings if baseline available
                    if baseline_tokens:
                        budgeted_tokens = data['token_stats']['avg_tokens_used']
                        savings_pct = ((baseline_tokens - budgeted_tokens) / baseline_tokens) * 100
                        comparison['token_savings_percent'] = savings_pct

                if 'budget_stats' in data:
                    comparison['budget_stats'] = data['budget_stats']

                comparisons.append(comparison)

        summary['comparisons'] = comparisons
        summary['metadata']['baseline_accuracy'] = baseline_accuracy

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"📄 Summary JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare results across multiple percentile runs"
    )
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to experiment directory containing run subdirectories'
    )
    parser.add_argument(
        '--output',
        default='experiment_summary.json',
        help='Output path for JSON summary (default: experiment_summary.json)'
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"❌ Error: Experiment directory not found: {experiment_dir}")
        return 1

    # Collect data from all runs
    print(f"🔍 Scanning experiment directory: {experiment_dir}")
    runs = collect_run_data(experiment_dir)

    if not runs:
        print(f"❌ Error: No valid runs found in {experiment_dir}")
        print(f"   Expected subdirectories: unbounded/, p<XX>/ (e.g., p85/, p65/, p45/, p25/)")
        print(f"   Each should contain: metrics/verify_report.csv")
        return 1

    print(f"✅ Found {len(runs)} runs: {', '.join(runs.keys())}")

    # Print comparison table
    print_comparison_table(runs)

    # Generate JSON summary
    output_path = Path(args.output)
    generate_summary_json(runs, output_path)

    print(f"\n✅ Comparison complete!")

    return 0


if __name__ == '__main__':
    exit(main())
