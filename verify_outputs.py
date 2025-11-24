#!/usr/bin/env python3
"""
verify_outputs.py
-----------------

Compare LLM-generated answers to expected answers in test.jsonl.
Optionally track token budget adherence.

Usage:
  python verify_outputs.py --stdout-file outputs.txt --test-jsonl test.jsonl --out results.csv
  python verify_outputs.py --stdout-file outputs.txt --test-jsonl test.jsonl --budget-file budgets.csv --out results.csv

Expected format:
- test.jsonl: one JSON per line, each with an "answer" field.
- stdout file: each line (or block) contains a model output answer, one per test case.
- budget file (optional): CSV with columns: id, token_budget

Output CSV columns:
- problem_id, expected, generated, match, note, tokens_used
- If --budget-file provided: tokens_allocated, followed_budget
"""

import argparse
import csv
import json
import math
import re
from fractions import Fraction
from pathlib import Path

def load_jsonl(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries

def load_model_outputs(path):
    """
    Extract answers from the model's output file.
    Parses the === Request Output === format from benchmark_serving.py stdout.
    Returns list of (generated_text, problem_id, tokens_used, thinking_tokens, has_budget) tuples.
    - problem_id: ID from dataset
    - thinking_tokens: tokens used in thinking phase (multi-phase only), None otherwise
    - has_budget: True if this entry is from a budgeted run, False for unbounded runs
    """
    outputs = []
    text = Path(path).read_text(encoding="utf-8")

    # Split by request output blocks
    blocks = re.split(r"=+ Request Output =+", text)

    for block in blocks:
        if not block.strip():
            continue

        # Extract ID from output
        id_match = re.search(r"ID:\s*(.+)", block)
        if id_match:
            problem_id = id_match.group(1).strip()
            if problem_id == "None":
                problem_id = None
        else:
            # Try to find Index: field as fallback (for backwards compatibility)
            index_match = re.search(r"Index:\s*(\d+)", block)
            problem_id = index_match.group(1).strip() if index_match else None

        # Extract generated text
        gen_match = re.search(r"Generated text:\s*(.+?)(?=\nActual tokens|$)", block, re.DOTALL)
        generated_text = gen_match.group(1).strip() if gen_match else ""

        # Extract actual tokens used (total for multi-phase)
        tokens_match = re.search(r"Actual tokens used:\s*(\d+)", block)
        tokens_used = int(tokens_match.group(1)) if tokens_match else None

        # Extract thinking tokens used (only present in multi-phase generation)
        thinking_tokens_match = re.search(r"Thinking tokens used:\s*(\d+)", block)
        thinking_tokens = int(thinking_tokens_match.group(1)) if thinking_tokens_match else None

        # Extract token budget (present in budgeted runs, absent in unbounded runs)
        budget_match = re.search(r"Token budget:\s*(\d+)", block)
        has_budget = budget_match is not None

        outputs.append((generated_text, problem_id, tokens_used, thinking_tokens, has_budget))

    return outputs

def load_budget_file(path):
    """
    Load token budget file.
    Returns dict mapping problem_id (str) to token_budget (int).
    """
    budgets = {}
    if not path:
        return budgets

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Use 'id' column from budget file
                problem_id = str(row.get('id', '')).strip()
                if not problem_id:
                    continue
                budget = int(row['token_budget'])
                budgets[problem_id] = budget
            except (ValueError, KeyError):
                continue
    return budgets

def normalize(s):
    s = s.strip()
    s = s.replace("$", "")
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\", "")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()

    # Clean up malformed extractions from multi-phase generation
    # Examples: "A}" -> "A", "B}'" -> "B", "C}'," -> "C"
    # These occur when answer prompt ends with '\boxed{' and model generates closing braces
    s = re.sub(r'^([A-D])\}[\'",]*$', r'\1', s)  # Single letter with trailing junk
    s = re.sub(r'^([A-D])\}$', r'\1', s)  # Just closing brace
    s = re.sub(r'[}\'"]+$', '', s)  # Remove trailing braces/quotes from any answer

    return s.strip()

def try_numeric(s):
    """Try to interpret as number or fraction or sqrt."""
    if not s:
        return None
    s = s.strip()
    try:
        return float(Fraction(s))
    except Exception:
        pass
    # \frac{a}{b}
    m = re.match(r"frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            return None
    # sqrt(...)
    m = re.match(r"([+-]?\d+(?:/\d+)?)?\s*sqrt\{?([0-9.+-]+)\}?", s)
    if m:
        coef = m.group(1)
        coef = float(Fraction(coef)) if coef else 1.0
        return coef * math.sqrt(float(m.group(2)))
    # fallback: first number in string
    m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

def answers_match(expected, got):
    e, g = normalize(expected), normalize(got)

    # Exact match after normalization
    if e == g:
        return True, "exact"

    # Multiple choice: check if both are single letters (A/B/C/D)
    e_clean = e.strip().upper()
    g_clean = g.strip().upper()
    if len(e_clean) == 1 and len(g_clean) == 1 and e_clean.isalpha() and g_clean.isalpha():
        if e_clean == g_clean:
            return True, f"multiple choice ({e_clean})"
        else:
            return False, f"multiple choice mismatch ({e_clean} vs {g_clean})"

    # Numeric comparison
    ev, gv = try_numeric(e), try_numeric(g)
    if ev is not None and gv is not None:
        if math.isclose(ev, gv, rel_tol=1e-6, abs_tol=1e-6):
            return True, f"numeric close ({ev} ~ {gv})"
        else:
            return False, f"numeric differ ({ev} vs {gv})"

    # Token overlap fallback
    e_tokens, g_tokens = set(e.lower().split()), set(g.lower().split())
    overlap = e_tokens & g_tokens
    if overlap and len(overlap) >= 0.6 * min(len(e_tokens), len(g_tokens)):
        return True, f"token overlap {len(overlap)}"

    return False, "mismatch"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout-file", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-file", default=None, help="Optional: Path to budget CSV file (e.g., budgets_for_client_50.csv)")
    args = ap.parse_args()

    # Infer results directory from stdout-file path
    stdout_path = Path(args.stdout_file)
    if args.out is None:
        # Extract results directory (e.g., results/9216db5781bf21249d130ec9da846c4624c16137)
        # stdout_file is typically: results/{model}/answers/stdout_*.txt
        if "results" in stdout_path.parts:
            results_idx = stdout_path.parts.index("results")
            if len(stdout_path.parts) > results_idx + 1:
                results_dir = Path(*stdout_path.parts[:results_idx+2])
                metrics_dir = results_dir / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                args.out = str(metrics_dir / "verify_report.csv")
            else:
                args.out = "verify_report.csv"
        else:
            args.out = "verify_report.csv"

    tests = load_jsonl(args.test_jsonl)
    outputs = load_model_outputs(args.stdout_file)
    budgets = load_budget_file(args.budget_file) if args.budget_file else {}

    rows = []
    matches = 0

    # Create a map of problem ID to expected answer
    # JSONL files may have "index" or "id" field
    test_map = {}
    for test in tests:
        # Try "index" first (for JSONL format), then "id" (for JSON format)
        problem_id = str(test.get("index") or test.get("id", ""))
        if problem_id:
            test_map[problem_id] = test.get("answer", "").strip()

    for i, (generated_text, problem_id, tokens_used, thinking_tokens, has_budget) in enumerate(outputs):
        # If budget file is provided, only process entries that have budget info
        # (skip unbounded runs that may be mixed in the same stdout file)
        if budgets and not has_budget:
            continue

        # Look up expected answer by problem_id
        exp = ""
        if problem_id is not None:
            exp = test_map.get(problem_id, "")
        elif i < len(tests):
            # Fallback: use sequential index if no ID available
            fallback_id = str(tests[i].get("index") or tests[i].get("id", i))
            exp = tests[i].get("answer", "").strip()
            problem_id = fallback_id

        # Extract answer from generated text (look for \boxed{})
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", generated_text)
        if boxed_match:
            got = boxed_match.group(1).strip()
        else:
            # Fallback: use the generated text itself (may be truncated)
            got = generated_text[:100].strip()

        ok, note = answers_match(exp, got)
        if ok:
            matches += 1

        # Build base row
        row = {
            "problem_id": problem_id,
            "expected": exp,
            "generated": got[:200],  # Truncate for readability
            "match": ok,
            "note": note,
        }

        # Always include tokens_used if available
        row["tokens_used"] = tokens_used if tokens_used is not None else ""

        # Add token budget tracking only if budgets exist
        if budgets:
            tokens_allocated = budgets.get(problem_id)
            followed_budget = None

            # For budget adherence, use thinking_tokens if available (multi-phase),
            # otherwise use total tokens_used (single-phase)
            tokens_for_budget_check = thinking_tokens if thinking_tokens is not None else tokens_used

            if tokens_for_budget_check is not None and tokens_allocated is not None:
                followed_budget = tokens_for_budget_check <= tokens_allocated

            row["tokens_allocated"] = tokens_allocated if tokens_allocated is not None else ""
            row["followed_budget"] = followed_budget if followed_budget is not None else ""

        rows.append(row)

    # Determine fieldnames dynamically based on whether budget data is present
    fieldnames = ["problem_id", "expected", "generated", "match", "note", "tokens_used"]
    if budgets:
        fieldnames.extend(["tokens_allocated", "followed_budget"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Calculate total from actual rows processed (after filtering)
    total = len(rows)
    pct = 100 * matches / total if total else 0
    print(f"✅ Compared {total} items: {matches} correct ({pct:.1f}%)")

    # Calculate and print token usage statistics (always, if tokens_used data is available)
    tokens_used_values = [row.get("tokens_used") for row in rows if isinstance(row.get("tokens_used"), int)]
    if tokens_used_values:
        total_tokens_used = sum(tokens_used_values)
        avg_tokens_used = total_tokens_used / len(tokens_used_values)
        min_tokens = min(tokens_used_values)
        max_tokens = max(tokens_used_values)

        print(f"\n📊 Token Usage Statistics:")
        print(f"  Total tokens used: {total_tokens_used:,}")
        print(f"  Avg tokens per problem: {avg_tokens_used:.0f}")
        print(f"  Min tokens: {min_tokens:,}")
        print(f"  Max tokens: {max_tokens:,}")

    # Print budget adherence statistics if budget data was provided
    if budgets:
        budget_followed = sum(1 for row in rows if row.get("followed_budget") is True)
        budget_exceeded = sum(1 for row in rows if row.get("followed_budget") is False)
        budget_tracked = budget_followed + budget_exceeded

        # Calculate budget-specific token statistics
        total_tokens_allocated = sum(row.get("tokens_allocated") for row in rows if isinstance(row.get("tokens_allocated"), int))
        avg_tokens_allocated = total_tokens_allocated / budget_tracked if budget_tracked > 0 else 0

        if budget_tracked > 0:
            budget_pct = 100 * budget_followed / budget_tracked
            print(f"\n📊 Token Budget Adherence:")
            print(f"  Budget adherence: {budget_followed}/{budget_tracked} ({budget_pct:.1f}%) within budget")
            print(f"  Total tokens allocated: {total_tokens_allocated:,}")
            print(f"  Avg tokens allocated per problem: {avg_tokens_allocated:.0f}")
            if tokens_used_values and total_tokens_allocated > 0:
                total_tokens_used = sum(tokens_used_values)
                usage_pct = 100 * total_tokens_used / total_tokens_allocated
                print(f"  Overall token usage: {usage_pct:.1f}% of allocated budget")

    print(f"\n📄 Report written to {args.out}")

if __name__ == "__main__":
    main()
