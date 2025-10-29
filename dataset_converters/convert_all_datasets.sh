#!/bin/bash
# convert_all_datasets.sh
# =======================
# Convert all datasets in the datasets/ directory to benchmark format.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="/home/al2926/vllm_reason_bench/datasets"

cd "$SCRIPT_DIR"

echo "========================================================================"
echo "CONVERTING ALL DATASETS"
echo "========================================================================"
echo ""

# GSM8K
if [ -f "${DATASETS_DIR}/gsm8k_test.json" ]; then
    echo "📊 Converting GSM8K..."
    python convert_dataset.py \
        --input "${DATASETS_DIR}/gsm8k_test.json" \
        --output-dir "${DATASETS_DIR}" \
        --dataset-type gsm8k \
        --output-name GSM8K
    echo ""
fi

# MathBench Arithmetic
if [ -f "${DATASETS_DIR}/mathbench_arithmetic.jsonl" ]; then
    echo "📊 Converting MathBench-Arithmetic..."
    python convert_dataset.py \
        --input "${DATASETS_DIR}/mathbench_arithmetic.jsonl" \
        --output-dir "${DATASETS_DIR}" \
        --dataset-type mathbench_arithmetic \
        --output-name MathBench-Arithmetic
    echo ""
fi

# MathBench College
if [ -f "${DATASETS_DIR}/mathbench_college.jsonl" ]; then
    echo "📊 Converting MathBench-College..."
    python convert_dataset.py \
        --input "${DATASETS_DIR}/mathbench_college.jsonl" \
        --output-dir "${DATASETS_DIR}" \
        --dataset-type mathbench_college \
        --output-name MathBench-College
    echo ""
fi

# MathBench Middle
if [ -f "${DATASETS_DIR}/mathbench_middle.jsonl" ]; then
    echo "📊 Converting MathBench-Middle..."
    python convert_dataset.py \
        --input "${DATASETS_DIR}/mathbench_middle.jsonl" \
        --output-dir "${DATASETS_DIR}" \
        --dataset-type mathbench_middle \
        --output-name MathBench-Middle
    echo ""
fi

# MathBench High
if [ -f "${DATASETS_DIR}/mathbench_high.jsonl" ]; then
    echo "📊 Converting MathBench-High..."
    python convert_dataset.py \
        --input "${DATASETS_DIR}/mathbench_high.jsonl" \
        --output-dir "${DATASETS_DIR}" \
        --dataset-type mathbench_high \
        --output-name MathBench-High
    echo ""
fi

echo "========================================================================"
echo "✅ ALL DATASETS CONVERTED"
echo "========================================================================"
echo ""
echo "Available datasets for benchmarking:"
echo "  - GSM8K_benchmark.json"
echo "  - MathBench-Arithmetic_benchmark.json"
echo "  - MathBench-College_benchmark.json"
echo "  - MathBench-Middle_benchmark.json"
echo "  - MathBench-High_benchmark.json"
echo ""
echo "Run experiments with:"
echo "  ./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100"
echo "========================================================================"
