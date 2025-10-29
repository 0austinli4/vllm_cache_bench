#!/usr/bin/env python3
"""
extract_token_usage.py
----------------------
Parse the stdout file from a dry run and extract per-problem token usage.

Output CSV format:
    id, output_tokens

This CSV is used by the orchestration script to create per-problem budgets
for different percentiles.

Usage:
    python extract_token_usage.py --stdout-file <path> --out <output.csv>
"""

import argparse
import csv
import re
from pathlib import Path


def parse_stdout_blocks(text: str):
    """
    Parse stdout file and extract token usage per problem.

    Supports both formats:
    - Regular: === Request Output ===
    - Full: === Full Request Output ===

    Returns:
        List of dicts with 'id' and 'output_tokens'
    """
    # Try to split by regular format first
    blocks = [b.strip() for b in re.split(r"=+ Request Output =+", text) if b.strip()]

    # If no blocks found, try full output format
    if len(blocks) <= 1:
        blocks = [b.strip() for b in re.split(r"=+ Full Request Output =+", text) if b.strip()]

    records = []

    for block in blocks:
        problem_id = None
        output_tokens = None

        # Extract ID from output
        m = re.search(r"ID:\s*(\S+)", block)
        if m:
            problem_id = m.group(1)
            if problem_id == "None":
                problem_id = None

        # Fallback to Index if ID not found (backwards compatibility)
        if problem_id is None:
            m = re.search(r"Index:\s*(\d+)", block)
            if m:
                problem_id = m.group(1)

        # Extract actual tokens used (this is the output length)
        m = re.search(r"Actual tokens used:\s*(\d+)", block)
        if m:
            try:
                output_tokens = int(m.group(1))
            except ValueError:
                output_tokens = None

        # Only include records with valid token counts and ID
        if output_tokens is not None and problem_id is not None:
            records.append({
                'id': problem_id,
                'output_tokens': output_tokens
            })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-problem token usage from dry run stdout"
    )
    parser.add_argument(
        "--stdout-file",
        required=True,
        help="Path to stdout file from dry run"
    )
    parser.add_argument(
        "--out",
        default="token_usage.csv",
        help="Output CSV file (default: token_usage.csv)"
    )

    args = parser.parse_args()

    stdout_path = Path(args.stdout_file)
    if not stdout_path.exists():
        raise SystemExit(f"Error: stdout file not found: {stdout_path}")

    # Parse stdout file
    text = stdout_path.read_text(encoding='utf-8')
    records = parse_stdout_blocks(text)

    if not records:
        raise SystemExit("Error: No token usage data found in stdout file")

    # Write CSV
    out_path = Path(args.out)
    with out_path.open('w', encoding='utf-8', newline='') as csvf:
        fieldnames = ['id', 'output_tokens']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            writer.writerow({
                'id': rec['id'],
                'output_tokens': rec['output_tokens']
            })

    print(f"✅ Extracted token usage for {len(records)} problems")
    print(f"📄 Output written to: {out_path}")

    # Print statistics
    total_tokens = sum(rec['output_tokens'] for rec in records)
    avg_tokens = total_tokens / len(records) if records else 0
    max_tokens = max(rec['output_tokens'] for rec in records) if records else 0
    min_tokens = min(rec['output_tokens'] for rec in records) if records else 0

    print(f"\n📊 Token Usage Statistics:")
    print(f"  Total problems:     {len(records)}")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  Average per problem: {avg_tokens:.1f}")
    print(f"  Min:                {min_tokens:,}")
    print(f"  Max:                {max_tokens:,}")


if __name__ == '__main__':
    main()
