# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment Setup

**IMPORTANT**: Before running ANY Python scripts in this repository, you MUST activate the correct environment:

```bash
module load anaconda3/2024.6
conda activate dynasor
```

This applies to ALL Python commands including any script execution.

**Always use these commands before running Python code.**

## Project Overview

This is an automated benchmarking tool for evaluating vLLM (Virtual Large Language Model) performance on mathematical reasoning tasks under token budget constraints. The project tests model accuracy across multiple budget percentiles and analyzes the tradeoff between token usage and accuracy.

## Quick Start

### Running a Complete Experiment

The easiest way to run a full experiment is using the orchestration script:

```bash
# Run full experiment (dry run + all percentiles) on SLURM
./submit_experiment.sh --dataset AIME25_benchmark.json --num-prompts 30
```

This automatically:
1. Runs dry run (unbounded) to collect baseline token usage
2. Extracts per-problem token usage
3. Creates budget files for percentiles: 85%, 65%, 45%, 25%
4. Runs benchmarks for each percentile (parallel on SLURM)
5. Verifies all outputs against ground truth
6. Generates comprehensive comparison report

### Running Meta-Prediction Experiment

Test if the LLM can predict its own performance (answer correctness and response length):

```bash
# First, run the unbounded experiment to establish baseline
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100

# Then run meta-prediction experiment
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

See [META_PREDICTION_README.md](META_PREDICTION_README.md) for detailed documentation.

## Available Datasets

The following datasets have been converted and are ready for benchmarking:

| Dataset | Problems | Type | Answer Format |
|---------|----------|------|---------------|
| **AIME25** | 30 | Math competition | Numeric |
| **GSM8K** | 7,473 | Grade school math | Numeric (extracted from solution) |
| **MathBench-Arithmetic** | 300 | Elementary arithmetic | Numeric |
| **MathBench-College** | 150 | College-level math | Multiple choice (A/B/C/D) |
| **MathBench-Middle** | 150 | Middle school math | Multiple choice (A/B/C/D) |
| **MathBench-High** | 150 | High school math | Multiple choice (A/B/C/D) |

### Example Usage

```bash
# Run experiment on GSM8K (full dataset)
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 7473

# Run experiment on MathBench-College
./submit_experiment.sh --dataset datasets/MathBench-College_benchmark.json --num-prompts 150

# Run experiment on subset of GSM8K
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

### Adding New Datasets

To convert a new dataset to benchmark format:

```bash
cd dataset_converters

# Convert a single dataset
python convert_dataset.py \
    --input /path/to/dataset.jsonl \
    --output-dir ../datasets/ \
    --dataset-type <type>

# Or convert all datasets at once
./convert_all_datasets.sh
```

See `dataset_converters/dataset_registry.py` for supported dataset types and formats.

## New Directory Structure

Results are organized by model and dataset:

```
results/
  └── {model_id}/
      └── {dataset_name}/
          ├── unbounded/              # Dry run (no budget constraints)
          │   ├── answers/            # Model outputs
          │   ├── metrics/            # Verification reports
          │   │   └── verify_report.csv
          │   └── configs/            # Run configurations
          ├── p85/                    # 85% percentile budget run
          │   ├── answers/
          │   ├── metrics/
          │   │   └── verify_report.csv
          │   └── configs/
          ├── p65/                    # 65% percentile budget run
          ├── p45/                    # 45% percentile budget run
          ├── p25/                    # 25% percentile budget run
          ├── token_usage.csv         # Per-problem token usage from dry run
          ├── budgets_p85.csv         # Budget file for 85% run
          ├── budgets_p65.csv         # Budget file for 65% run
          ├── budgets_p45.csv         # Budget file for 45% run
          ├── budgets_p25.csv         # Budget file for 25% run
          ├── experiment_summary.json # Aggregate comparison metrics
          └── slurm_logs/            # SLURM job logs (if using SLURM)
              ├── dryrun-{job_id}.out
              ├── p85-{job_id}.out
              └── ...
```

## Core Workflow Components

### 1. Master Orchestration Scripts

#### `run_experiment.py`
Python orchestration script for complete experiments. Runs sequentially.

```bash
# Run full experiment
python run_experiment.py --dataset AIME25_benchmark.json --num-prompts 30

# Run only dry run
python run_experiment.py --dataset AIME25_benchmark.json --dry-run-only

# Skip dry run if already completed
python run_experiment.py --dataset AIME25_benchmark.json --skip-dry-run

# Custom percentiles
python run_experiment.py --dataset AIME25_benchmark.json --percentiles "90,70,50,30"
```

#### `submit_experiment.sh`
SLURM wrapper for parallel execution. Submits separate jobs for each run.

```bash
# Submit experiment to SLURM (parallel execution)
./submit_experiment.sh --dataset AIME25_benchmark.json --num-prompts 30

# Custom percentiles
./submit_experiment.sh --dataset AIME25_benchmark.json --percentiles "85,65,45,25"

# Monitor progress
squeue -u $USER
tail -f results/{model_id}/{dataset_name}/slurm_logs/dryrun-{job_id}.out
```

**Job dependency chain:**
1. Dry run job runs first
2. All percentile jobs run in parallel after dry run completes
3. Comparison job runs after all percentile jobs complete

### 2. Individual Run Script

#### `run.py`
Low-level script for running individual benchmarks. Called by orchestration scripts.

```bash
# Dry run (unbounded)
python run.py --dry-run --dataset AIME25_benchmark.json --num-prompts 30 --output-dir results/{model}/AIME25/unbounded

# Budgeted run
python run.py --run-with-budgets --dataset AIME25_benchmark.json --num-prompts 30 --output-dir results/{model}/AIME25/p85 --budgets-file results/{model}/AIME25/budgets_p85.csv
```

**Parameters:**
- `--dataset`: Path to dataset file (e.g., AIME25_benchmark.json)
- `--answers`: Path to answers file (auto-detected if not provided)
- `--output-dir`: Directory for results
- `--num-prompts`: Number of problems to run
- `--dry-run`: Run without budget constraints
- `--run-with-budgets`: Run with budget constraints
- `--budgets-file`: Path to budget CSV file

### 3. Token Budget Management

#### `extract_token_usage.py`
Extracts per-problem token usage from dry run outputs.

```bash
python extract_token_usage.py --stdout-file results/{model}/AIME25/unbounded/answers/stdout_*.txt --out token_usage.csv
```

**Output format (token_usage.csv):**
```csv
id,output_tokens
1,1523
2,2104
...
```

#### Budget File Creation
Budget files are automatically created by the orchestration scripts. Each percentile gets its own budget file where:

```
token_budget_for_problem_i = percentile * token_usage_from_dry_run_for_problem_i
```

For example, if dry run used 1000 tokens for problem 5:
- p85 budget: 850 tokens
- p65 budget: 650 tokens
- p45 budget: 450 tokens
- p25 budget: 250 tokens

Budget files use this format:
```csv
id,token_budget
1,1294
2,1788
...
```

### 4. Verification and Analysis

#### `verify_outputs.py`
Verifies model outputs against ground truth and tracks budget adherence.

```bash
# Basic verification
python verify_outputs.py --stdout-file outputs.txt --test-jsonl AIME25.jsonl --out report.csv

# With budget tracking
python verify_outputs.py --stdout-file outputs.txt --test-jsonl AIME25.jsonl --budget-file budgets.csv --out report.csv
```

**Output columns:**
- `problem_id`: Problem ID from dataset
- `expected`: Ground truth answer
- `generated`: Model's answer
- `match`: True/False correctness
- `note`: Matching method used
- `tokens_used`: Actual tokens used (if budget tracking enabled)
- `tokens_allocated`: Budget allocated (if budget tracking enabled)
- `followed_budget`: Whether budget was followed (if budget tracking enabled)

#### `compare_runs.py`
Generates comprehensive comparison across all runs.

```bash
python compare_runs.py --experiment-dir results/{model_id}/AIME25 --output experiment_summary.json
```

**Generates:**
- Console table comparing all runs
- JSON file with detailed metrics
- Accuracy comparison vs baseline
- Budget adherence statistics
- Token usage analysis

**Example output:**
```
====================================================================================================
EXPERIMENT RESULTS COMPARISON
====================================================================================================

Run          Accuracy    Correct    Total  Budget Adherence      Avg Tokens
----------------------------------------------------------------------------------------------------
unbounded      23.3%        7/30       30  N/A                    unlimited
p85            20.0%        6/30       30  28/30 (93.3%)          1234/1450
p65            16.7%        5/30       30  29/30 (96.7%)          945/1000
p45            13.3%        4/30       30  30/30 (100.0%)         656/710
p25            10.0%        3/30       30  30/30 (100.0%)         364/400
====================================================================================================

📊 SUMMARY STATISTICS
   Baseline (unbounded) accuracy: 23.3%

   Budget   Accuracy Drop   Token Savings    Adherence
   -------- --------------- --------------- ------------
   p85            -3.3%            ~15%        93.3%
   p65            -6.6%            ~35%        96.7%
   p45           -10.0%            ~55%       100.0%
   p25           -13.3%            ~75%       100.0%
```

## Advanced Usage

### Running Individual Components

If you need to run components separately:

```bash
# 1. Run dry run
python run.py --dry-run --dataset AIME25_benchmark.json --num-prompts 30 --output-dir results/{model}/AIME25/unbounded

# 2. Extract token usage
python extract_token_usage.py --stdout-file results/{model}/AIME25/unbounded/answers/stdout_*.txt --out results/{model}/AIME25/token_usage.csv

# 3. Create budget file for p85
python -c "
import csv
multiplier = 0.85
with open('results/{model}/AIME25/token_usage.csv', 'r') as inf, open('results/{model}/AIME25/budgets_p85.csv', 'w', newline='') as outf:
    reader = csv.DictReader(inf)
    writer = csv.DictWriter(outf, fieldnames=['id', 'token_budget'])
    writer.writeheader()
    for row in reader:
        token_budget = int(float(row['output_tokens']) * multiplier)
        writer.writerow({'id': row['id'], 'token_budget': token_budget})
"

# 4. Run with budget
python run.py --run-with-budgets --dataset AIME25_benchmark.json --num-prompts 30 --output-dir results/{model}/AIME25/p85 --budgets-file results/{model}/AIME25/budgets_p85.csv

# 5. Verify outputs
python verify_outputs.py --stdout-file results/{model}/AIME25/p85/answers/stdout_*.txt --test-jsonl AIME25.jsonl --budget-file results/{model}/AIME25/budgets_p85.csv --out results/{model}/AIME25/p85/metrics/verify_report.csv
```

### Working with Multiple Datasets

The system supports multiple datasets. Each dataset gets its own directory:

```bash
# Run experiment on AIME25
./submit_experiment.sh --dataset AIME25_benchmark.json --num-prompts 30

# Run experiment on another dataset
./submit_experiment.sh --dataset MATH500_benchmark.json --num-prompts 100

# Directory structure:
# results/{model_id}/
#   ├── AIME25/
#   │   ├── unbounded/
#   │   ├── p85/
#   │   └── ...
#   └── MATH500/
#       ├── unbounded/
#       ├── p85/
#       └── ...
```

### Custom Percentiles

You can specify custom percentiles:

```bash
# Run with custom percentiles
./submit_experiment.sh --dataset AIME25_benchmark.json --percentiles "90,75,50,25,10"

# Or with run_experiment.py
python run_experiment.py --dataset AIME25_benchmark.json --percentiles "90,75,50,25,10"
```

## Configuration

### Model Configuration (`constants.py`)

```python
MODEL = "/path/to/model"  # Path to vLLM model
MODEL_ID = "model_identifier"  # Extracted from path
```

### Server Configuration (`run.py`)

Server settings control vLLM behavior:
- GPU memory utilization: 80%
- Port: 8000
- Block size: 16
- Max batched tokens: 16384

### SLURM Configuration (`job_single_run.slurm`)

Default SLURM settings:
- Time limit: 1:30:00
- Memory: 40G per CPU
- GPUs: 1 per job
- Email notifications: enabled

Modify these in `job_single_run.slurm` as needed.

## Key Metrics

### Accuracy Metrics
- **Accuracy**: Percentage of correct answers
- **Correct/Total**: Number of problems solved correctly
- **Accuracy Drop**: Difference from unbounded baseline

### Token Budget Metrics
- **Budget Adherence Rate**: Percentage of problems staying within budget
- **Token Usage Rate**: Actual tokens used vs allocated
- **Average Tokens Used**: Mean tokens per problem
- **Average Tokens Allocated**: Mean budget per problem

### Output Files
- **verify_report.csv**: Per-problem verification results
- **experiment_summary.json**: Aggregate metrics across all runs
- **token_usage.csv**: Per-problem token usage from dry run
- **budgets_p{N}.csv**: Token budgets for percentile N

## Meta-Prediction Experiment

The meta-prediction experiment tests the LLM's ability to predict its own performance - a form of meta-cognition.

### What It Tests

1. **Answer Correctness Prediction**: Can the model predict if it will answer a question correctly?
2. **Response Length Prediction**: Can the model estimate how many tokens its response will use?

### Workflow

```bash
# Step 1: Run unbounded experiment to establish baseline
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100

# Step 2: Run meta-prediction experiment
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

### How It Works

Each question is transformed into a meta-prediction prompt:

```
Evaluate whether you can answer a given question and estimate the length of your response.

Provide your assessment in the following format:
Can answer: \boxed{[yes]/[no]}
Estimated length: \boxed{[X tokens]}

Evaluate the following question:
<question>
[ORIGINAL QUESTION]
</question>

Can answer: \boxed{
```

The model's predictions are then compared against actual results from the unbounded run.

### Evaluation Metrics

**Correctness Prediction:**
- Accuracy: % of correct predictions about answer correctness
- Precision (predicted "yes"): When model predicts it can answer, how often is it correct?
- Precision (predicted "no"): When model predicts it cannot answer, how often is it wrong?

**Length Prediction:**
- MAE (Mean Absolute Error): Average token difference
- MAPE (Mean Absolute Percentage Error): Average percentage error
- RMSE (Root Mean Squared Error): Penalizes large errors

### Directory Structure

```
results/{model_id}/{dataset_name}/
└── meta_prediction/
    ├── answers/                      # Raw predictions
    ├── metrics/
    │   └── meta_prediction_report.csv  # Evaluation results
    └── {dataset}_meta_prediction.json  # Transformed dataset
```

### Detailed Documentation

See [META_PREDICTION_README.md](META_PREDICTION_README.md) for:
- Complete usage examples
- Output format specifications
- Analysis techniques
- Troubleshooting guide

## Troubleshooting

### Common Issues

1. **No stdout files found**
   - Ensure benchmark ran successfully
   - Check that vLLM server started properly
   - Look in `{output_dir}/answers/` for stdout files

2. **Budget file not found**
   - Ensure dry run completed successfully
   - Check that `extract_token_usage.py` ran
   - Verify budget files exist in dataset base directory

3. **SLURM jobs not starting**
   - Check job dependencies: `squeue -u $USER`
   - Review logs in `results/{model}/{dataset}/slurm_logs/`
   - Verify dry run completed successfully

4. **Server fails to start**
   - Check GPU availability: `nvidia-smi`
   - Review server logs in results directory
   - Ensure no other vLLM servers are running on the same port

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View real-time logs
tail -f results/{model}/{dataset}/slurm_logs/dryrun-{job_id}.out

# Check all logs
ls -lt results/{model}/{dataset}/slurm_logs/
```

## Migration Notes

### Changes from Old System

1. **Directory structure**: Results now organized by model → dataset → run
2. **Automated workflow**: Single command runs entire experiment
3. **Parallel execution**: SLURM jobs run percentiles in parallel
4. **Per-problem budgets**: Token budgets computed per problem, not global percentiles
5. **Budget adherence tracking**: Automatically tracked in verification reports
6. **Comprehensive comparison**: Single script compares all runs

### Old Results

Old results have been moved to `results_legacy/` directory. They follow the old structure and are not compatible with the new workflow.

## Important Notes

- **Environment**: Always activate the dynasor conda environment before running scripts
- **Server lifecycle**: Each run automatically kills existing servers and starts fresh
- **Token budgets**: Computed per-problem based on dry run usage (percentile * dry_run_tokens)
- **Parallel execution**: SLURM submission enables parallel runs; local execution is sequential
- **Budget adherence**: Automatically verified for all budgeted runs
- **Answer format**: Expected in LaTeX `\\boxed{}` notation for math problems
- **Shell scripts**: ALL shell scripts (*.sh) MUST use Unix line endings (LF), not Windows line endings (CRLF). If creating or modifying shell scripts, ensure they use Unix line endings to avoid "bad interpreter" errors
- **SLURM percentile passing**: Percentiles are converted from comma-delimited (user input: "75,50,25") to colon-delimited ("75:50:25") when passing through sbatch --export, because sbatch uses commas as delimiters. The job script converts them back. NEVER use commas directly in sbatch --export for list values
