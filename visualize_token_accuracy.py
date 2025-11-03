#!/usr/bin/env python3
"""
Visualize the relationship between token percentiles and accuracy.
Reads experiment_summary.json and produces multiple visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

def extract_percentile_number(run_name):
    """Extract numeric percentile from run name (e.g., 'p50' -> 50)"""
    if run_name == 'unbounded':
        return 100
    return int(run_name[1:])  # Remove 'p' prefix

def load_data(json_path):
    """Load and parse experiment summary JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract data for visualization
    runs_data = []

    for run_name, run_info in data['runs'].items():
        percentile = extract_percentile_number(run_name)
        avg_tokens = run_info['token_stats']['avg_tokens_used']
        accuracy = run_info['accuracy']
        total = run_info['total']
        correct = run_info['correct']

        # Get budget stats if available
        usage_rate = None
        adherence_rate = None
        if 'budget_stats' in run_info:
            usage_rate = run_info['budget_stats']['usage_rate']
            adherence_rate = run_info['budget_stats']['adherence_rate']

        runs_data.append({
            'run_name': run_name,
            'percentile': percentile,
            'accuracy': accuracy,
            'avg_tokens': avg_tokens,
            'total': total,
            'correct': correct,
            'usage_rate': usage_rate,
            'adherence_rate': adherence_rate
        })

    # Sort by percentile
    runs_data.sort(key=lambda x: x['percentile'])

    return runs_data, data['metadata']

def create_visualizations(runs_data, metadata, output_prefix):
    """Create comprehensive visualizations"""

    # Extract arrays for plotting
    percentiles = [r['percentile'] for r in runs_data]
    accuracies = [r['accuracy'] for r in runs_data]
    avg_tokens = [r['avg_tokens'] for r in runs_data]
    corrects = [r['correct'] for r in runs_data]
    totals = [r['total'] for r in runs_data]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Token Budget vs Accuracy Analysis - MathBench-High', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy vs Token Percentile
    ax1 = axes[0, 0]
    ax1.plot(percentiles, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=metadata['baseline_accuracy'], color='r', linestyle='--',
                label=f'Baseline (unbounded): {metadata["baseline_accuracy"]:.2f}%', linewidth=2)
    ax1.set_xlabel('Token Budget Percentile', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Token Budget Percentile', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add value labels
    for i, (p, a) in enumerate(zip(percentiles, accuracies)):
        ax1.annotate(f'{a:.1f}%', (p, a), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Accuracy vs Average Tokens Used
    ax2 = axes[0, 1]
    scatter = ax2.scatter(avg_tokens, accuracies, s=200, c=percentiles,
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1.5)
    ax2.plot(avg_tokens, accuracies, linewidth=1, alpha=0.5, color='gray')
    ax2.set_xlabel('Average Tokens Used', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Average Tokens Used', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Token Budget Percentile', rotation=270, labelpad=20)

    # Add labels for each point
    for i, (tokens, acc, p) in enumerate(zip(avg_tokens, accuracies, percentiles)):
        label = f'p{p}' if p != 100 else 'unbounded'
        ax2.annotate(label, (tokens, acc), textcoords="offset points",
                    xytext=(8,-8), ha='left', fontsize=8)

    # Plot 3: Number of Correct Answers vs Token Percentile
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(percentiles)), corrects, color='#A23B72', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Token Budget Percentile', fontsize=12)
    ax3.set_ylabel('Number of Correct Answers', fontsize=12)
    ax3.set_title('Correct Answers vs Token Budget Percentile', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(percentiles)))
    ax3.set_xticklabels([f'p{p}' if p != 100 else 'unb.' for p in percentiles], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, correct, total) in enumerate(zip(bars, corrects, totals)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{correct}/{total}',
                ha='center', va='bottom', fontsize=9)

    # Plot 4: Token Efficiency (Accuracy per 100 tokens)
    ax4 = axes[1, 1]
    efficiency = [acc / (tokens / 100) for acc, tokens in zip(accuracies, avg_tokens)]
    ax4.plot(percentiles, efficiency, marker='s', linewidth=2, markersize=8, color='#F18F01')
    ax4.set_xlabel('Token Budget Percentile', fontsize=12)
    ax4.set_ylabel('Accuracy per 100 Tokens', fontsize=12)
    ax4.set_title('Token Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, (p, eff) in enumerate(zip(percentiles, efficiency)):
        ax4.annotate(f'{eff:.2f}', (p, eff), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = f"{output_prefix}_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview visualization to: {output_path}")

    # Create additional detailed plot: Accuracy drop and token savings
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Filter out unbounded for this comparison
    budgeted_runs = [r for r in runs_data if r['percentile'] != 100]
    percentiles_budgeted = [r['percentile'] for r in budgeted_runs]

    # Calculate drops and savings relative to baseline
    baseline_acc = metadata['baseline_accuracy']
    baseline_tokens = metadata['baseline_avg_tokens']

    acc_drops = [baseline_acc - r['accuracy'] for r in budgeted_runs]
    token_savings = [(baseline_tokens - r['avg_tokens']) / baseline_tokens * 100
                     for r in budgeted_runs]

    # Create dual-axis plot
    ax_acc = ax
    ax_tokens = ax.twinx()

    line1 = ax_acc.plot(percentiles_budgeted, acc_drops, marker='o', linewidth=2,
                        markersize=8, color='#E63946', label='Accuracy Drop (%)')
    line2 = ax_tokens.plot(percentiles_budgeted, token_savings, marker='s', linewidth=2,
                          markersize=8, color='#06A77D', label='Token Savings (%)')

    ax_acc.set_xlabel('Token Budget Percentile', fontsize=12)
    ax_acc.set_ylabel('Accuracy Drop from Baseline (%)', fontsize=12, color='#E63946')
    ax_tokens.set_ylabel('Token Savings from Baseline (%)', fontsize=12, color='#06A77D')
    ax_acc.tick_params(axis='y', labelcolor='#E63946')
    ax_tokens.tick_params(axis='y', labelcolor='#06A77D')

    ax_acc.set_title('Tradeoff: Accuracy Drop vs Token Savings', fontsize=14, fontweight='bold')
    ax_acc.grid(True, alpha=0.3)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_acc.legend(lines, labels, loc='upper left')

    # Add value labels
    for p, drop, saving in zip(percentiles_budgeted, acc_drops, token_savings):
        ax_acc.annotate(f'{drop:.1f}%', (p, drop), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9, color='#E63946')
        ax_tokens.annotate(f'{saving:.1f}%', (p, saving), textcoords="offset points",
                          xytext=(0,-15), ha='center', fontsize=9, color='#06A77D')

    plt.tight_layout()

    # Save figure
    output_path = f"{output_prefix}_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved tradeoff visualization to: {output_path}")

    return runs_data

def print_summary_table(runs_data, metadata):
    """Print a summary table to console"""
    print("\n" + "="*90)
    print("TOKEN BUDGET VS ACCURACY SUMMARY")
    print("="*90)
    print(f"{'Run':<12} {'Percentile':<12} {'Accuracy':<12} {'Avg Tokens':<15} {'Correct/Total':<15}")
    print("-"*90)

    for run in runs_data:
        run_name = run['run_name']
        percentile = f"{run['percentile']}%" if run['percentile'] != 100 else "unbounded"
        accuracy = f"{run['accuracy']:.2f}%"
        avg_tokens = f"{run['avg_tokens']:.1f}"
        correct_total = f"{run['correct']}/{run['total']}"

        print(f"{run_name:<12} {percentile:<12} {accuracy:<12} {avg_tokens:<15} {correct_total:<15}")

    print("="*90)
    print(f"\nBaseline (unbounded) Accuracy: {metadata['baseline_accuracy']:.2f}%")
    print(f"Baseline (unbounded) Avg Tokens: {metadata['baseline_avg_tokens']:.1f}")
    print("="*90 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize token budget vs accuracy relationship')
    parser.add_argument('--json-file', type=str,
                       default='/home/al2926/vllm_reason_bench/results/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/MathBench-High/experiment_summary.json',
                       help='Path to experiment_summary.json file')
    parser.add_argument('--output-prefix', type=str,
                       default='/home/al2926/vllm_reason_bench/results/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/MathBench-High/visualization',
                       help='Prefix for output visualization files')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.json_file}")
    runs_data, metadata = load_data(args.json_file)

    # Print summary table
    print_summary_table(runs_data, metadata)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(runs_data, metadata, args.output_prefix)

    print("\nVisualization complete!")

if __name__ == '__main__':
    main()
