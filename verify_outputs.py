#!/usr/bin/env python3
"""
verify_outputs.py
-----------------

Compare LLM-generated answers to expected answers in test.jsonl.

Usage:
  python verify_outputs.py --stdout-file outputs.txt --test-jsonl test.jsonl --out results.csv

Expected format:
- test.jsonl: one JSON per line, each with an "answer" field.
- stdout file: each line (or block) contains a model output answer, one per test case.
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
    Returns list of (generated_text, unique_id) tuples.
    """
    outputs = []
    text = Path(path).read_text(encoding="utf-8")

    # Split by request output blocks
    blocks = re.split(r"=+ Request Output =+", text)

    for block in blocks:
        if not block.strip():
            continue

        # Extract unique_id or index
        unique_id_match = re.search(r"Unique id:\s*(.+)", block)
        if unique_id_match:
            unique_id = unique_id_match.group(1).strip()
            if unique_id == "None":
                unique_id = None
        else:
            # Try to find Index: field as fallback
            index_match = re.search(r"Index:\s*(\d+)", block)
            unique_id = index_match.group(1).strip() if index_match else None

        # Extract generated text
        gen_match = re.search(r"Generated text:\s*(.+?)(?=\nActual tokens|$)", block, re.DOTALL)
        if gen_match:
            generated_text = gen_match.group(1).strip()
            outputs.append((generated_text, unique_id))

    return outputs

def normalize(s):
    s = s.strip()
    s = s.replace("$", "")
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\", "")
    s = re.sub(r"\s+", " ", s)
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
    if e == g:
        return True, "exact"

    ev, gv = try_numeric(e), try_numeric(g)
    if ev is not None and gv is not None:
        if math.isclose(ev, gv, rel_tol=1e-6, abs_tol=1e-6):
            return True, f"numeric close ({ev} ~ {gv})"
        else:
            return False, f"numeric differ ({ev} vs {gv})"

    # token overlap fallback
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

    rows = []
    total = len(outputs)
    matches = 0

    # Create a map of problem index to expected answer
    test_map = {}
    for test in tests:
        idx = test.get("index")
        if idx is not None:
            test_map[idx] = test.get("answer", "").strip()

    for i, (generated_text, unique_id) in enumerate(outputs):
        # Try to match by unique_id first, otherwise use sequential index
        if unique_id is not None and unique_id.isdigit():
            problem_idx = int(unique_id)
            exp = test_map.get(problem_idx, "")
        elif i < len(tests):
            problem_idx = tests[i].get("index", i)
            exp = tests[i].get("answer", "").strip()
        else:
            problem_idx = i
            exp = ""

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

        rows.append({
            "problem_index": problem_idx,
            "expected": exp,
            "generated": got[:200],  # Truncate for readability
            "match": ok,
            "note": note
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_index", "expected", "generated", "match", "note"])
        writer.writeheader()
        writer.writerows(rows)

    pct = 100 * matches / total if total else 0
    print(f"✅ Compared {total} items: {matches} correct ({pct:.1f}%)")
    print(f"📄 Report written to {args.out}")

if __name__ == "__main__":
    main()
