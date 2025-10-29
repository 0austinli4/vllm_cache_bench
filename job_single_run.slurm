#!/bin/bash
#SBATCH --job-name=vllm_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --time=00:59:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=al2926@princeton.edu

# Load modules and activate environment
module purge
module load anaconda3/2024.6
conda activate dynasor

# Export required environment variables
export OPENAI_HARMONY_LOCAL_PATH=/scratch/gpfs/WLLOYD/al2926/hf_models/openai/gpt-oss-20b
export TIKTOKEN_ENCODINGS_BASE=/scratch/gpfs/WLLOYD/al2926/hf_models/openai/encodings

# Change to the correct directory
cd /home/al2926/vllm_reason_bench

# Extract model ID and dataset name for directory structure
MODEL_ID=$(python3 -c "from constants import MODEL_ID; print(MODEL_ID)")

# For compare jobs, DATASET_NAME is already exported; for other jobs, extract from DATASET
if [[ -z "$DATASET_NAME" ]]; then
    DATASET_NAME=$(basename "$DATASET" .json | sed 's/_benchmark//')
fi

RESULTS_BASE="results/${MODEL_ID}/${DATASET_NAME}"

echo "========================================================================"
echo "JOB INFORMATION"
echo "========================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Node:           $SLURM_NODELIST"
echo "Run Type:       $RUN_TYPE"
echo "Dataset:        $DATASET"
echo "Answers:        $ANSWERS"
echo "Num Prompts:    $NUM_PROMPTS"
if [[ "$RUN_TYPE" == "percentile" ]]; then
    echo "Percentile:     $PERCENTILE"
fi
echo "Results Base:   $RESULTS_BASE"
echo "========================================================================"
echo ""

# ========================================================================
# Execute based on RUN_TYPE
# ========================================================================

if [[ "$RUN_TYPE" == "dry-run" ]]; then
    # ====================================================================
    # DRY RUN (UNBOUNDED)
    # ====================================================================
    echo "🚀 Starting dry run (unbounded)..."

    OUTPUT_DIR="${RESULTS_BASE}/unbounded"

    python run.py \
        --dry-run \
        --dataset "$DATASET" \
        --answers "$ANSWERS" \
        --output-dir "$OUTPUT_DIR" \
        --num-prompts "$NUM_PROMPTS"

    echo ""
    echo "✅ Dry run completed"

    # Extract token usage
    echo ""
    echo "🔍 Extracting token usage..."

    # Find the most recent stdout file, excluding full_output files
    STDOUT_FILE=$(ls -t "${OUTPUT_DIR}/answers/stdout_"*.txt 2>/dev/null | grep -v "full_output" | head -n1)

    if [[ -z "$STDOUT_FILE" ]]; then
        echo "❌ Error: No stdout file found in ${OUTPUT_DIR}/answers/"
        exit 1
    fi

    python extract_token_usage.py \
        --stdout-file "$STDOUT_FILE" \
        --out "${RESULTS_BASE}/token_usage.csv"

    echo ""
    echo "✅ Token usage extracted"

    # Create budget files for all percentiles
    echo ""
    echo "📊 Creating budget files..."

    # Use PERCENTILES env var if provided, otherwise default to 85:65:45:25
    # Note: PERCENTILES uses colons as delimiters (converted from commas by submit script)
    # because sbatch --export uses commas as delimiters
    PERCENTILES_TO_CREATE="${PERCENTILES:-85:65:45:25}"

    # Convert colons back to commas for display and processing
    PERCENTILES_DISPLAY=$(echo "$PERCENTILES_TO_CREATE" | tr ':' ',')
    echo "Creating budgets for percentiles: $PERCENTILES_DISPLAY"

    IFS=':' read -ra PERC_ARRAY <<< "$PERCENTILES_TO_CREATE"
    for PERC in "${PERC_ARRAY[@]}"; do
        PERC=$(echo "$PERC" | xargs)  # Trim whitespace
        python3 << EOF
import csv
import sys
from pathlib import Path

token_usage_csv = '${RESULTS_BASE}/token_usage.csv'
percentile = ${PERC}
output_file = '${RESULTS_BASE}/budgets_p${PERC}.csv'
multiplier = percentile / 100.0

with open(token_usage_csv, 'r', encoding='utf-8') as inf, \
     open(output_file, 'w', encoding='utf-8', newline='') as outf:
    reader = csv.DictReader(inf)
    writer = csv.DictWriter(outf, fieldnames=['id', 'token_budget'])
    writer.writeheader()

    for row in reader:
        output_tokens = row.get('output_tokens')
        problem_id = row.get('id')
        if output_tokens is None or output_tokens == '' or not problem_id:
            continue

        token_budget = int(float(output_tokens) * multiplier)

        writer.writerow({
            'id': problem_id,
            'token_budget': token_budget
        })

print(f'✅ Created budget file: {output_file}')
EOF
    done

    echo ""
    echo "✅ All budget files created"

elif [[ "$RUN_TYPE" == "percentile" ]]; then
    # ====================================================================
    # PERCENTILE RUN (BUDGETED)
    # ====================================================================
    echo "🚀 Starting p${PERCENTILE} run (budgeted)..."

    OUTPUT_DIR="${RESULTS_BASE}/p${PERCENTILE}"
    BUDGET_FILE="${RESULTS_BASE}/budgets_p${PERCENTILE}.csv"

    if [[ ! -f "$BUDGET_FILE" ]]; then
        echo "❌ Error: Budget file not found: $BUDGET_FILE"
        exit 1
    fi

    python run.py \
        --run-with-budgets \
        --dataset "$DATASET" \
        --answers "$ANSWERS" \
        --output-dir "$OUTPUT_DIR" \
        --num-prompts "$NUM_PROMPTS" \
        --budgets-file "$BUDGET_FILE"

    echo ""
    echo "✅ p${PERCENTILE} run completed"

    # Verify outputs with budget tracking
    echo ""
    echo "🔍 Verifying outputs..."

    # Find the most recent stdout file, excluding full_output files
    STDOUT_FILE=$(ls -t "${OUTPUT_DIR}/answers/stdout_"*.txt 2>/dev/null | grep -v "full_output" | head -n1)

    if [[ -z "$STDOUT_FILE" ]]; then
        echo "❌ Error: No stdout file found in ${OUTPUT_DIR}/answers/"
        exit 1
    fi

    python verify_outputs.py \
        --stdout-file "$STDOUT_FILE" \
        --test-jsonl "$ANSWERS" \
        --budget-file "$BUDGET_FILE" \
        --out "${OUTPUT_DIR}/metrics/verify_report.csv"

    echo ""
    echo "✅ Verification completed"

elif [[ "$RUN_TYPE" == "compare" ]]; then
    # ====================================================================
    # COMPARISON REPORT
    # ====================================================================
    echo "🚀 Generating comparison report..."

    python compare_runs.py \
        --experiment-dir "$RESULTS_BASE" \
        --output "${RESULTS_BASE}/experiment_summary.json"

    echo ""
    echo "✅ Comparison report generated"

else
    echo "❌ Error: Unknown RUN_TYPE: $RUN_TYPE"
    echo "Valid types: dry-run, percentile, compare"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ JOB COMPLETED SUCCESSFULLY"
echo "========================================================================"
