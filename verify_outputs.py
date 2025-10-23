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
    This version looks for text inside \\boxed{} first; if not found, uses the last number-like token.
    """
    outputs = []
    text = Path(path).read_text(encoding="utf-8")

    # Try to split on boxed answers if available
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        outputs = [b.strip() for b in boxed]
    else:
        # fallback: take non-empty lines that look like answers
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            outputs.append(line)
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
    ap.add_argument("--out", default="verify_report.csv")
    args = ap.parse_args()

    tests = load_jsonl(args.test_jsonl)
    outputs = load_model_outputs(args.stdout_file)

    rows = []
    total = min(len(tests), len(outputs))
    matches = 0

    for i in range(total):
        exp = tests[i].get("answer", "").strip()
        got = outputs[i].strip()
        ok, note = answers_match(exp, got)
        if ok:
            matches += 1
        rows.append({
            "index": i,
            "expected": exp,
            "generated": got,
            "match": ok,
            "note": note
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "expected", "generated", "match", "note"])
        writer.writeheader()
        writer.writerows(rows)

    pct = 100 * matches / total if total else 0
    print(f"✅ Compared {total} items: {matches} correct ({pct:.1f}%)")
    print(f"📄 Report written to {args.out}")

if __name__ == "__main__":
    main()
