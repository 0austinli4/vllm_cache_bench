#!/bin/bash
# Rerun verification for all MathBench-High percentiles with corrected extraction

module load anaconda3/2024.6
conda activate dynasor

BASE_DIR="results/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d/MathBench-High"

for perc in p40 p50 p60 p70 p80 p90 p95; do
  echo "=== Verifying $perc ==="

  STDOUT=$(ls -t "$BASE_DIR/$perc/answers/stdout_"*.txt 2>/dev/null | grep -v "full_output" | head -n1)

  if [[ -n "$STDOUT" ]]; then
    python verify_outputs.py \
      --stdout-file "$STDOUT" \
      --test-jsonl datasets/MathBench-High.jsonl \
      --budget-file "$BASE_DIR/budgets_$perc.csv" \
      --out "$BASE_DIR/$perc/metrics/verify_report.csv"
    echo ""
  else
    echo "No stdout file found for $perc"
    echo ""
  fi
done

echo "=== Summary ==="
echo ""
for perc in p40 p50 p60 p70 p80 p90 p95; do
  if [[ -f "$BASE_DIR/$perc/metrics/verify_report.csv" ]]; then
    correct=$(grep -c ",True," "$BASE_DIR/$perc/metrics/verify_report.csv")
    total=$(( $(wc -l < "$BASE_DIR/$perc/metrics/verify_report.csv") - 1 ))
    accuracy=$(echo "scale=1; 100 * $correct / $total" | bc)
    echo "$perc: $correct/$total ($accuracy%)"
  fi
done
