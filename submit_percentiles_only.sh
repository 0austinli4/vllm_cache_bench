#!/bin/bash
# submit_percentiles_only.sh
# ==========================
# Submit ONLY percentile jobs to SLURM (skips dry run).
# Use this when you already have budget files created from a previous dry run.
#
# This script:
# 1. Verifies that budget files exist for requested percentiles
# 2. Submits percentile jobs in parallel (no dependencies)
#
# Usage:
#   ./submit_percentiles_only.sh --dataset AIME25_benchmark.json --percentiles "75,60,45,30"
#   ./submit_percentiles_only.sh --dataset MathBench-College_benchmark.json --percentiles "90,75,60"

set -e

# Default values
DATASET=""
ANSWERS=""
NUM_PROMPTS=30
PERCENTILES="85,65,45,25"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --answers)
            ANSWERS="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --percentiles)
            PERCENTILES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dataset <file> [--answers <file>] [--num-prompts <n>] [--percentiles <list>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: $0 --dataset <file> [--answers <file>] [--num-prompts <n>] [--percentiles <list>]"
    exit 1
fi

# Remove spaces and convert commas to colons for sbatch --export compatibility
# sbatch --export uses comma as delimiter, so "75,50,25" would be parsed incorrectly
# Convert to "75:50:25" for safe passing, then convert back in the job script
PERCENTILES=$(echo "$PERCENTILES" | tr -d ' ' | tr ',' ':')

# Extract dataset name for directory organization
DATASET_NAME=$(basename "$DATASET" .json | sed 's/_benchmark//')
DATASET_DIR=$(dirname "$DATASET")

# Infer answers file if not provided
if [[ -z "$ANSWERS" ]]; then
    ANSWERS="${DATASET_DIR}/${DATASET_NAME}.jsonl"
fi

# Verify files exist
if [[ ! -f "$DATASET" ]]; then
    echo "Error: Dataset file not found: $DATASET"
    exit 1
fi

if [[ ! -f "$ANSWERS" ]]; then
    echo "Error: Answers file not found: $ANSWERS"
    exit 1
fi

# Get model ID from constants.py
MODEL_ID=$(python3 -c "from constants import MODEL_ID; print(MODEL_ID)")
RESULTS_BASE="results/${MODEL_ID}/${DATASET_NAME}"

# Create log directory
mkdir -p "${RESULTS_BASE}/slurm_logs"

echo "========================================================================"
echo "PERCENTILE-ONLY SUBMISSION"
echo "========================================================================"
echo "Dataset:        $DATASET"
echo "Answers:        $ANSWERS"
echo "Dataset name:   $DATASET_NAME"
echo "Num prompts:    $NUM_PROMPTS"
echo "Percentiles:    $(echo "$PERCENTILES" | tr ':' ',')"
echo "Results dir:    $RESULTS_BASE"
echo "Logs dir:       ${RESULTS_BASE}/slurm_logs"
echo "========================================================================"
echo ""

# ========================================================================
# STEP 1: Verify budget files exist
# ========================================================================
echo "🔍 Verifying budget files exist..."

MISSING_BUDGETS=()
IFS=':' read -ra PERC_ARRAY <<< "$PERCENTILES"

for PERC in "${PERC_ARRAY[@]}"; do
    PERC=$(echo "$PERC" | xargs)  # Trim whitespace
    BUDGET_FILE="${RESULTS_BASE}/budgets_p${PERC}.csv"

    if [[ ! -f "$BUDGET_FILE" ]]; then
        MISSING_BUDGETS+=("p${PERC}")
        echo "  ❌ Missing: $BUDGET_FILE"
    else
        echo "  ✅ Found: $BUDGET_FILE"
    fi
done

if [[ ${#MISSING_BUDGETS[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Error: Budget files missing for percentiles: ${MISSING_BUDGETS[*]}"
    echo ""
    echo "You need to run the dry run first to generate budget files:"
    echo "  ./submit_experiment.sh --dataset $DATASET --num-prompts $NUM_PROMPTS"
    echo ""
    echo "Or run the dry run locally:"
    echo "  python run_experiment.py --dataset $DATASET --dry-run-only"
    exit 1
fi

echo ""
echo "✅ All budget files found"
echo ""

# ========================================================================
# STEP 2: Submit percentile jobs (no dependencies)
# ========================================================================
echo "🚀 Submitting percentile jobs (no dry run dependency)..."

PERCENTILE_JOBS=()
IFS=':' read -ra PERC_ARRAY <<< "$PERCENTILES"

for PERC in "${PERC_ARRAY[@]}"; do
    PERC=$(echo "$PERC" | xargs)  # Trim whitespace

    PERC_JOB=$(sbatch --parsable \
        --job-name="p${PERC}_${DATASET_NAME}" \
        --output="${RESULTS_BASE}/slurm_logs/p${PERC}-%j.out" \
        --error="${RESULTS_BASE}/slurm_logs/p${PERC}-%j.err" \
        --export=ALL,DATASET="$DATASET",ANSWERS="$ANSWERS",NUM_PROMPTS="$NUM_PROMPTS",PERCENTILE="$PERC",RUN_TYPE="percentile" \
        job_single_run.slurm)

    PERCENTILE_JOBS+=($PERC_JOB)
    echo "✅ p${PERC} job submitted: Job ID $PERC_JOB"
done

echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo "========================================================================"
echo "✅ ALL JOBS SUBMITTED SUCCESSFULLY"
echo "========================================================================"
echo "Percentile jobs:"
echo "  ${PERCENTILE_JOBS[*]} (running in parallel, no dependencies)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${RESULTS_BASE}/slurm_logs/p${PERC_ARRAY[0]}*.out"
echo ""
echo "Results will be in:"
echo "  ${RESULTS_BASE}/"
echo ""
echo "After all jobs complete, run this command to generate comparison report:"
echo "  python compare_runs.py --experiment-dir ${RESULTS_BASE} --output ${RESULTS_BASE}/experiment_summary.json"
echo "========================================================================"
