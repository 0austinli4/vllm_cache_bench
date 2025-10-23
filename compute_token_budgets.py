#!/usr/bin/env python3
"""
compute_token_budgets.py

Parse the stdout file produced by `backend_request_func` (same marker format as the verifier),
measure the number of tokens in each generated output using the project's tokenizer, and
produce budget suggestions (e.g., 90%, 75%, 50% of the observed output tokens).

Outputs a CSV with columns:
  index, unique_id, prompt, generated_text, output_tokens, budget_90, budget_75, budget_50

"""
from pathlib import Path
import argparse
import csv
import re
import json
import importlib.util
import sys


def parse_stdout_blocks(text: str):
    blocks = [b.strip() for b in re.split(r"=+ Request Output =+", text) if b.strip()]
    records = []
    for b in blocks:
        # Extract common keys that backend_request_func writes to stdout.
        # Prefer explicit 'Actual tokens used' when present to avoid re-tokenizing.
        prompt = ""
        generated = ""
        prompt_len = None
        actual_tokens = None
        latency = None
        ttft = None
        success = None
        timestamp = None
        unique_id = None

        m = re.search(r"Prompt:\s*(.*?)\n(?:Generated text:|Actual tokens used:|Prompt length:|Budget estimated:|$)", b, flags=re.S)
        if m:
            prompt = m.group(1).strip().replace('\n', ' ')

        # First prefer any explicit \boxed{...} answer commonly used in LaTeX-style outputs.
        # This lets the script pick the concise final answer without re-tokenizing surrounding text.
        m_box = re.search(r"\\boxed\s*\{(.+?)\}", b, flags=re.S)
        if m_box:
            # normalize whitespace/newlines inside the box
            generated = m_box.group(1).strip().replace('\n', ' ')
        else:
            m = re.search(r"Generated text:\s*(.*?)\n(?:Actual tokens used:|Prompt length:|Budget estimated:|$)", b, flags=re.S)
            if m:
                generated = m.group(1).strip()

        # prompt length (may be labeled Prompt length or Prompt length:)
        m = re.search(r"Prompt length:\s*(\d+)", b)
        if m:
            try:
                prompt_len = int(m.group(1))
            except Exception:
                prompt_len = None

        m = re.search(r"Actual tokens used:\s*(\d+)", b)
        if m:
            try:
                actual_tokens = int(m.group(1))
            except Exception:
                actual_tokens = None

        # Additional metadata fields that may appear in newer outputs
        m = re.search(r"Latency:\s*([0-9.]+)s", b)
        if m:
            try:
                latency = float(m.group(1))
            except Exception:
                latency = None

        m = re.search(r"TTFT:\s*([0-9.]+)s", b)
        if m:
            try:
                ttft = float(m.group(1))
            except Exception:
                ttft = None

        m = re.search(r"Success:\s*(True|False)", b)
        if m:
            success = True if m.group(1) == 'True' else False

        m = re.search(r"Timestamp:\s*(\d+\.?\d*)", b)
        if m:
            try:
                timestamp = float(m.group(1))
            except Exception:
                timestamp = None

        # Some logs include a unique id or index label
        m = re.search(r"Unique[_ ]?id:\s*(\S+)", b)
        if m:
            unique_id = m.group(1)

        records.append({
            "prompt": prompt,
            "generated_text": generated,
            "prompt_len": prompt_len,
            "actual_tokens": actual_tokens,
            "latency": latency,
            "ttft": ttft,
            "success": success,
            "timestamp": timestamp,
            "unique_id": unique_id,
            "raw": b,
        })

    return records


def compute_budgets(token_count: int, proportions=(0.9, 0.75, 0.5)):
    return {f"{int(p*100)}%": max(0, int(token_count * p)) for p in proportions}


def main():
    parser = argparse.ArgumentParser(description="Compute token budgets from stdout log using project's tokenizer")
    parser.add_argument("--stdout-file", required=True, help="Path to stdout file written by backend_request_func")
    parser.add_argument("--test-jsonl", default="test.jsonl", help="Optional test.jsonl to include unique_id mapping")
    parser.add_argument("--out", default="token_budgets.csv", help="CSV output file")
    args = parser.parse_args()

    stdout_path = Path(args.stdout_file)
    if not stdout_path.exists():
        raise SystemExit(f"stdout file not found: {stdout_path}")

    text = stdout_path.read_text(encoding='utf-8')
    records = parse_stdout_blocks(text)

    # Optionally load test.jsonl unique_id mapping in order
    test_map = []
    test_path = Path(args.test_jsonl)
    if test_path.exists():
        with test_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    test_map.append(obj.get('unique_id'))
                except Exception:
                    test_map.append(None)

    out_path = Path(args.out)
    with out_path.open('w', encoding='utf-8', newline='') as csvf:
        # Do NOT include generated_text in the metrics CSV to avoid large or problematic content.
        fieldnames = ['index', 'unique_id', 'prompt', 'actual_tokens', 'output_tokens', 'budget_90', 'budget_75', 'budget_50']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(records):
            unique_id = test_map[i] if i < len(test_map) else ''
            gen = rec.get('generated_text', '')
            actual = rec.get('actual_tokens')
            if actual is not None:
                tok_count = actual
            else:
                # token_measure expects str and returns token count
                try:
                    tok_count = token_measure(gen)
                except Exception:
                    # fallback to simple whitespace estimate if tokenizer fails
                    tok_count = len(gen.split())

            budgets = compute_budgets(tok_count)
            writer.writerow({
                'index': i,
                'unique_id': unique_id,
                'prompt': rec.get('prompt',''),
                'actual_tokens': actual if actual is not None else '',
                'output_tokens': tok_count,
                'budget_90': budgets['90%'],
                'budget_75': budgets['75%'],
                'budget_50': budgets['50%'],
            })

    print(f"Wrote token budgets for {len(records)} records to {out_path}")


if __name__ == '__main__':
    main()
