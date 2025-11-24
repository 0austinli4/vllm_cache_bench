#!/bin/bash
#SBATCH --job-name=meta_pred
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --time=00:30:00
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
DATASET_NAME=$(basename "$DATASET" .json | sed 's/_benchmark//')

RESULTS_BASE="results/${MODEL_ID}/${DATASET_NAME}"

echo "========================================================================"
echo "META-PREDICTION EXPERIMENT"
echo "========================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURM_NODELIST"
echo "Dataset:        $DATASET"
echo "Num Prompts:    $NUM_PROMPTS"
echo "Baseline:       $BASELINE_REPORT"
echo "Results Dir:    ${RESULTS_BASE}/meta_prediction"
echo "========================================================================"
echo ""

# ========================================================================
# STEP 1: Run meta-prediction benchmark
# ========================================================================
echo "🚀 Running meta-prediction benchmark..."

python run_meta_prediction.py \
    --dataset "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --output-dir "${RESULTS_BASE}/meta_prediction"

echo ""
echo "✅ Meta-prediction benchmark completed"

# ========================================================================
# STEP 2: Verify predictions against baseline
# ========================================================================
echo ""
echo "🔍 Verifying meta-predictions against baseline..."

# Find the most recent stdout file
STDOUT_FILE=$(ls -t "${RESULTS_BASE}/meta_prediction/answers/stdout_"*.txt 2>/dev/null | grep -v "full_output" | head -n1)

if [[ -z "$STDOUT_FILE" ]]; then
    echo "❌ Error: No stdout file found in ${RESULTS_BASE}/meta_prediction/answers/"
    exit 1
fi

# If baseline report not provided, try to find it
if [[ -z "$BASELINE_REPORT" ]]; then
    # Try metrics subdirectory first (new structure)
    BASELINE_REPORT=$(ls -t "${RESULTS_BASE}/unbounded/metrics/verify_report_"*.csv 2>/dev/null | head -n1)

    # If not found, try unbounded directory directly (old structure)
    if [[ -z "$BASELINE_REPORT" ]]; then
        BASELINE_REPORT=$(ls -t "${RESULTS_BASE}/unbounded/verify_report_"*.csv 2>/dev/null | head -n1)
    fi

    if [[ -z "$BASELINE_REPORT" ]]; then
        echo "❌ Error: No baseline report found. Please run unbounded experiment first."
        exit 1
    fi
    echo "Using baseline report: $BASELINE_REPORT"
fi

python verify_meta_predictions.py \
    --meta-outputs "$STDOUT_FILE" \
    --baseline-report "$BASELINE_REPORT" \
    --out "${RESULTS_BASE}/meta_prediction/metrics/meta_prediction_report.csv"

echo ""
echo "✅ Verification completed"

echo ""
echo "========================================================================"
echo "✅ META-PREDICTION EXPERIMENT COMPLETED"
echo "========================================================================"
echo "Results:"
echo "  Meta-prediction outputs: ${RESULTS_BASE}/meta_prediction/answers/"
echo "  Evaluation report:       ${RESULTS_BASE}/meta_prediction/metrics/meta_prediction_report.csv"
echo "========================================================================"
