#!/usr/bin/env python3
"""Compare token usage between two runs."""
import csv
import sys
from pathlib import Path

def load_token_usage(filepath):
    """Load token usage CSV into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['id']] = int(row['output_tokens'])
    return data

def main():
    concise_file = Path("/home/al2926/vllm_reason_bench/results/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/GSM8K/token_usage_be_concise.csv")
    original_file = Path("/home/al2926/vllm_reason_bench/results/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/GSM8K/token_usage_original.csv")

    concise_data = load_token_usage(concise_file)
    original_data = load_token_usage(original_file)

    # Find common IDs
    common_ids = set(concise_data.keys()) & set(original_data.keys())

    print(f"Comparing token usage between two runs")
    print(f"=" * 80)
    print(f"Common problems: {len(common_ids)}")
    print()

    # Calculate statistics
    total_concise = 0
    total_original = 0
    concise_lower = 0
    original_lower = 0
    same = 0

    differences = []

    for problem_id in sorted(common_ids, key=lambda x: int(x) if x.isdigit() else x):
        concise_tokens = concise_data[problem_id]
        original_tokens = original_data[problem_id]
        diff = concise_tokens - original_tokens
        pct_diff = (diff / original_tokens * 100) if original_tokens > 0 else 0

        differences.append({
            'id': problem_id,
            'concise': concise_tokens,
            'original': original_tokens,
            'diff': diff,
            'pct_diff': pct_diff
        })

        total_concise += concise_tokens
        total_original += original_tokens

        if concise_tokens < original_tokens:
            concise_lower += 1
        elif concise_tokens > original_tokens:
            original_lower += 1
        else:
            same += 1

    # Print summary statistics
    avg_concise = total_concise / len(common_ids)
    avg_original = total_original / len(common_ids)
    avg_diff = avg_concise - avg_original
    avg_pct_diff = (avg_diff / avg_original * 100) if avg_original > 0 else 0

    print(f"Summary Statistics:")
    print(f"-" * 80)
    print(f"Total tokens (concise):    {total_concise:,}")
    print(f"Total tokens (original):   {total_original:,}")
    print(f"Difference:                {total_concise - total_original:+,} ({((total_concise - total_original)/total_original*100):+.1f}%)")
    print()
    print(f"Average tokens (concise):  {avg_concise:.1f}")
    print(f"Average tokens (original): {avg_original:.1f}")
    print(f"Average difference:        {avg_diff:+.1f} ({avg_pct_diff:+.1f}%)")
    print()
    print(f"Concise used fewer:        {concise_lower} problems ({concise_lower/len(common_ids)*100:.1f}%)")
    print(f"Original used fewer:       {original_lower} problems ({original_lower/len(common_ids)*100:.1f}%)")
    print(f"Same token count:          {same} problems ({same/len(common_ids)*100:.1f}%)")
    print()

    # Show top 10 differences (where concise saved most)
    differences.sort(key=lambda x: x['diff'])
    print(f"Top 10 problems where 'be concise' saved most tokens:")
    print(f"-" * 80)
    print(f"{'ID':<10} {'Concise':>10} {'Original':>10} {'Diff':>10} {'% Change':>10}")
    print(f"-" * 80)
    for item in differences[:10]:
        print(f"{item['id']:<10} {item['concise']:>10,} {item['original']:>10,} {item['diff']:>10,} {item['pct_diff']:>9.1f}%")
    print()

    # Show top 10 where concise used more
    print(f"Top 10 problems where 'be concise' used MORE tokens:")
    print(f"-" * 80)
    print(f"{'ID':<10} {'Concise':>10} {'Original':>10} {'Diff':>10} {'% Change':>10}")
    print(f"-" * 80)
    for item in differences[-10:]:
        print(f"{item['id']:<10} {item['concise']:>10,} {item['original']:>10,} {item['diff']:>10,} {item['pct_diff']:>9.1f}%")

if __name__ == '__main__':
    main()
