#!/usr/bin/env python3
"""
run_experiment.py
-----------------
Master orchestration script for running complete benchmarking experiments.

This script automates the entire workflow:
1. Run dry run (unbounded) to collect baseline token usage
2. Extract token usage from dry run
3. Create budget files for each percentile (85, 65, 45, 25)
4. Run benchmarks for each percentile
5. Verify all outputs against ground truth
6. Generate comprehensive comparison report

Usage:
    # Run full experiment with default settings
    python run_experiment.py --dataset AIME25_benchmark.json

    # Run with custom number of prompts
    python run_experiment.py --dataset AIME25_benchmark.json --num-prompts 30

    # Run only specific percentiles
    python run_experiment.py --dataset AIME25_benchmark.json --percentiles 85,65,45

    # Skip dry run if already completed
    python run_experiment.py --dataset AIME25_benchmark.json --skip-dry-run

Directory structure created:
    results/{model_id}/{dataset_name}/
        ├── unbounded/          # Dry run results
        │   ├── answers/
        │   ├── metrics/
        │   └── configs/
        ├── p85/                # 85% percentile run
        ├── p65/                # 65% percentile run
        ├── p45/                # 45% percentile run
        ├── p25/                # 25% percentile run
        ├── token_usage.csv     # Per-problem token usage from dry run
        └── experiment_summary.json  # Comparison results
"""

import argparse
import subprocess
import sys
import csv
import json
from pathlib import Path
import time
import glob

# Import project constants
from constants import MODEL_ID, get_results_dir, get_dataset_base_dir


def run_command(cmd, description, check=True):
    """Run a shell command and display progress."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=check)

    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        if check:
            sys.exit(1)
    else:
        print(f"✅ {description} completed successfully")

    return result


def find_latest_stdout(answers_dir):
    """Find the most recent stdout file in the answers directory."""
    pattern = str(Path(answers_dir) / 'stdout_*.txt')
    candidates = sorted(glob.glob(pattern), key=lambda f: Path(f).stat().st_mtime)
    if candidates:
        return candidates[-1]
    return None


def create_budget_file(token_usage_csv, percentile, output_file):
    """
    Create a budget file for a specific percentile.

    Reads token_usage.csv and multiplies each problem's output_tokens by
    percentile/100 to create per-problem budgets.

    Args:
        token_usage_csv: Path to CSV with columns: id, output_tokens
        percentile: Percentile value (e.g., 85, 65, 45, 25)
        output_file: Path to output budget file (will have columns: id, token_budget)
    """
    multiplier = percentile / 100.0

    with open(token_usage_csv, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8', newline='') as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=['id', 'token_budget'])
        writer.writeheader()

        for row in reader:
            output_tokens = row.get('output_tokens')
            problem_id = row.get('id')
            if output_tokens is None or output_tokens == '' or not problem_id:
                continue

            # Multiply by percentile and round to integer
            token_budget = int(float(output_tokens) * multiplier)

            writer.writerow({
                'id': problem_id,
                'token_budget': token_budget
            })

    print(f"✅ Created budget file for p{percentile}: {output_file}")


def run_dry_run(dataset, answers, num_prompts, output_dir):
    """Run dry run (unbounded) to collect baseline token usage."""
    cmd = [
        sys.executable,
        'run.py',
        '--dry-run',
        '--dataset', dataset,
        '--answers', answers,
        '--output-dir', output_dir,
        '--num-prompts', str(num_prompts)
    ]

    run_command(cmd, "Running dry run (unbounded)")


def extract_token_usage(stdout_file, output_csv):
    """Extract per-problem token usage from dry run stdout."""
    cmd = [
        sys.executable,
        'extract_token_usage.py',
        '--stdout-file', stdout_file,
        '--out', output_csv
    ]

    run_command(cmd, "Extracting token usage from dry run")


def run_with_budget(dataset, answers, num_prompts, output_dir, budget_file):
    """Run benchmark with token budget constraints."""
    percentile = Path(output_dir).name  # e.g., 'p85'

    cmd = [
        sys.executable,
        'run.py',
        '--run-with-budgets',
        '--dataset', dataset,
        '--answers', answers,
        '--output-dir', output_dir,
        '--num-prompts', str(num_prompts),
        '--budgets-file', budget_file
    ]

    run_command(cmd, f"Running benchmark with {percentile} budget")


def verify_run(run_dir, answers_file, budget_file=None):
    """Verify outputs for a specific run."""
    answers_dir = Path(run_dir) / 'answers'
    stdout_file = find_latest_stdout(answers_dir)

    if not stdout_file:
        print(f"⚠️  No stdout file found in {answers_dir}, skipping verification")
        return None

    metrics_dir = Path(run_dir) / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    report_csv = metrics_dir / 'verify_report.csv'

    cmd = [
        sys.executable,
        'verify_outputs.py',
        '--stdout-file', stdout_file,
        '--test-jsonl', answers_file,
        '--out', str(report_csv)
    ]

    if budget_file:
        cmd.extend(['--budget-file', budget_file])

    run_command(cmd, f"Verifying outputs for {Path(run_dir).name}")

    return str(report_csv)


def generate_comparison_report(dataset_base_dir):
    """Generate comprehensive comparison report."""
    cmd = [
        sys.executable,
        'compare_runs.py',
        '--experiment-dir', dataset_base_dir,
        '--output', str(Path(dataset_base_dir) / 'experiment_summary.json')
    ]

    run_command(cmd, "Generating comparison report")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete benchmarking experiment with multiple percentiles"
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset file (e.g., AIME25_benchmark.json)'
    )
    parser.add_argument(
        '--answers',
        default=None,
        help='Path to answers file (e.g., AIME25.jsonl). Auto-detected if not provided'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=30,
        help='Number of prompts to run (default: 30)'
    )
    parser.add_argument(
        '--percentiles',
        default='85,65,45,25',
        help='Comma-separated list of percentiles to run (default: 85,65,45,25)'
    )
    parser.add_argument(
        '--skip-dry-run',
        action='store_true',
        help='Skip dry run and use existing token_usage.csv'
    )
    parser.add_argument(
        '--dry-run-only',
        action='store_true',
        help='Only run dry run, do not run budgeted experiments'
    )

    args = parser.parse_args()

    # Parse percentiles
    percentiles = [int(p.strip()) for p in args.percentiles.split(',')]

    # Infer dataset name and answers file
    dataset_path = Path(args.dataset)
    dataset_name = dataset_path.stem
    if '_benchmark' in dataset_name:
        dataset_name = dataset_name.replace('_benchmark', '')

    if args.answers:
        answers_file = args.answers
    else:
        answers_file = f"{dataset_name}.jsonl"

    # Verify files exist
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    if not Path(answers_file).exists():
        print(f"❌ Error: Answers file not found: {answers_file}")
        sys.exit(1)

    # Get base directory for this dataset
    dataset_base_dir = get_dataset_base_dir(dataset_name)

    print(f"\n{'='*80}")
    print(f"🔬 EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset:        {args.dataset}")
    print(f"Answers:        {answers_file}")
    print(f"Dataset name:   {dataset_name}")
    print(f"Num prompts:    {args.num_prompts}")
    print(f"Percentiles:    {percentiles}")
    print(f"Results dir:    {dataset_base_dir}")
    print(f"{'='*80}\n")

    # Track start time
    start_time = time.time()

    # ========================================================================
    # PHASE 1: DRY RUN (UNBOUNDED)
    # ========================================================================
    unbounded_dir = get_results_dir(dataset_name, 'unbounded')
    token_usage_csv = Path(dataset_base_dir) / 'token_usage.csv'

    if not args.skip_dry_run:
        print(f"\n{'#'*80}")
        print(f"# PHASE 1: DRY RUN (UNBOUNDED)")
        print(f"{'#'*80}\n")

        run_dry_run(args.dataset, answers_file, args.num_prompts, unbounded_dir)

        # Extract token usage
        stdout_file = find_latest_stdout(Path(unbounded_dir) / 'answers')
        if not stdout_file:
            print("❌ Error: No stdout file found after dry run")
            sys.exit(1)

        extract_token_usage(stdout_file, str(token_usage_csv))

        # Verify dry run outputs
        verify_run(unbounded_dir, answers_file)
    else:
        print(f"\n⏭️  Skipping dry run (using existing token_usage.csv)")
        if not token_usage_csv.exists():
            print(f"❌ Error: {token_usage_csv} not found. Cannot skip dry run.")
            sys.exit(1)

    if args.dry_run_only:
        print(f"\n✅ Dry run completed. Exiting (--dry-run-only specified)")
        sys.exit(0)

    # ========================================================================
    # PHASE 2: CREATE BUDGET FILES
    # ========================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 2: CREATE BUDGET FILES")
    print(f"{'#'*80}\n")

    budget_files = {}
    for percentile in percentiles:
        budget_file = Path(dataset_base_dir) / f'budgets_p{percentile}.csv'
        create_budget_file(token_usage_csv, percentile, budget_file)
        budget_files[percentile] = str(budget_file)

    # ========================================================================
    # PHASE 3: RUN BUDGETED EXPERIMENTS
    # ========================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 3: RUN BUDGETED EXPERIMENTS")
    print(f"{'#'*80}\n")

    for percentile in percentiles:
        run_dir = get_results_dir(dataset_name, f'p{percentile}')
        budget_file = budget_files[percentile]

        run_with_budget(args.dataset, answers_file, args.num_prompts, run_dir, budget_file)

        # Verify outputs
        verify_run(run_dir, answers_file, budget_file)

    # ========================================================================
    # PHASE 4: GENERATE COMPARISON REPORT
    # ========================================================================
    print(f"\n{'#'*80}")
    print(f"# PHASE 4: GENERATE COMPARISON REPORT")
    print(f"{'#'*80}\n")

    generate_comparison_report(dataset_base_dir)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60

    print(f"\n{'='*80}")
    print(f"🎉 EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Total time:     {elapsed_minutes:.1f} minutes")
    print(f"Results:        {dataset_base_dir}")
    print(f"Summary report: {Path(dataset_base_dir) / 'experiment_summary.json'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
