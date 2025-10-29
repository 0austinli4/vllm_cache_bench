#!/bin/bash
# submit_experiment.sh
# ====================
# Submit complete benchmarking experiment to SLURM with parallel execution.
#
# This script:
# 1. Submits dry run job
# 2. After dry run completes, submits all percentile jobs in parallel
# 3. After all jobs complete, generates comparison report
#
# Usage:
#   ./submit_experiment.sh --dataset AIME25_benchmark.json
#   ./submit_experiment.sh --dataset AIME25_benchmark.json --num-prompts 30
#   ./submit_experiment.sh --dataset AIME25_benchmark.json --percentiles "85,65,45,25"

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
echo "EXPERIMENT SUBMISSION"
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
# STEP 1: Submit dry run job
# ========================================================================
echo "🚀 Submitting dry run job..."

DRY_RUN_JOB=$(sbatch --parsable \
    --job-name="dryrun_${DATASET_NAME}" \
    --output="${RESULTS_BASE}/slurm_logs/dryrun-%j.out" \
    --error="${RESULTS_BASE}/slurm_logs/dryrun-%j.err" \
    --export=ALL,DATASET="$DATASET",ANSWERS="$ANSWERS",NUM_PROMPTS="$NUM_PROMPTS",PERCENTILES="$PERCENTILES",RUN_TYPE="dry-run" \
    job_single_run.slurm)

echo "✅ Dry run job submitted: Job ID $DRY_RUN_JOB"
echo "   Logs: ${RESULTS_BASE}/slurm_logs/dryrun-${DRY_RUN_JOB}.{out,err}"
echo ""

# ========================================================================
# STEP 2: Submit percentile jobs (dependent on dry run completion)
# ========================================================================
echo "🚀 Submitting percentile jobs (will start after dry run completes)..."

PERCENTILE_JOBS=()
IFS=':' read -ra PERC_ARRAY <<< "$PERCENTILES"

for PERC in "${PERC_ARRAY[@]}"; do
    PERC=$(echo "$PERC" | xargs)  # Trim whitespace

    PERC_JOB=$(sbatch --parsable \
        --job-name="p${PERC}_${DATASET_NAME}" \
        --output="${RESULTS_BASE}/slurm_logs/p${PERC}-%j.out" \
        --error="${RESULTS_BASE}/slurm_logs/p${PERC}-%j.err" \
        --dependency=afterok:$DRY_RUN_JOB \
        --export=ALL,DATASET="$DATASET",ANSWERS="$ANSWERS",NUM_PROMPTS="$NUM_PROMPTS",PERCENTILE="$PERC",RUN_TYPE="percentile" \
        job_single_run.slurm)

    PERCENTILE_JOBS+=($PERC_JOB)
    echo "✅ p${PERC} job submitted: Job ID $PERC_JOB (depends on $DRY_RUN_JOB)"
done

echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo "========================================================================"
echo "✅ ALL JOBS SUBMITTED SUCCESSFULLY"
echo "========================================================================"
echo "Job dependency chain:"
echo "  1. Dry run:      $DRY_RUN_JOB"
echo "  2. Percentiles:  ${PERCENTILE_JOBS[*]} (parallel, after dry run)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${RESULTS_BASE}/slurm_logs/dryrun-${DRY_RUN_JOB}.out"
echo ""
echo "Results will be in:"
echo "  ${RESULTS_BASE}/"
echo ""
echo "After all jobs complete, run this command to generate comparison report:"
echo "  python compare_runs.py --experiment-dir ${RESULTS_BASE} --output ${RESULTS_BASE}/experiment_summary.json"
echo "========================================================================"
