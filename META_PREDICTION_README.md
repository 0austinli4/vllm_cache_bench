# Meta-Prediction Experiment

## Overview

This experiment tests whether the LLM can accurately predict:
1. **Answer Correctness**: Can it determine if it will answer a question correctly?
2. **Response Length**: Can it estimate how many tokens its response will be?

This is a meta-cognitive evaluation - we're asking the model to predict its own performance before actually solving the problems.

## Quick Start

### Prerequisites

1. You must have already run the **unbounded (dry run)** experiment for your dataset to establish baseline actual results
2. The unbounded run should have generated a `verify_report_*.csv` file with actual correctness and token usage

### Running the Experiment

#### Option 1: SLURM Submission (Recommended)

```bash
# Run meta-prediction experiment on GSM8K (100 problems)
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100

# Run on AIME25 (30 problems)
./submit_meta_prediction.sh --dataset datasets/AIME25_benchmark.json --num-prompts 30

# Specify baseline report explicitly
./submit_meta_prediction.sh \
    --dataset datasets/GSM8K_benchmark.json \
    --num-prompts 100 \
    --baseline results/{model_id}/GSM8K/unbounded/metrics/verify_report_1762099424.csv
```

#### Option 2: Local Execution

```bash
# Activate environment
module load anaconda3/2024.6
conda activate dynasor

# Run meta-prediction benchmark
python run_meta_prediction.py \
    --dataset datasets/GSM8K_benchmark.json \
    --num-prompts 100

# After completion, verify against baseline
python verify_meta_predictions.py \
    --meta-outputs results/{model_id}/GSM8K/meta_prediction/answers/stdout_*.txt \
    --baseline-report results/{model_id}/GSM8K/unbounded/metrics/verify_report_*.csv \
    --out results/{model_id}/GSM8K/meta_prediction/metrics/meta_prediction_report.csv
```

## How It Works

### Step 1: Prompt Transformation

Each question is wrapped in a meta-prediction prompt:

```
Evaluate whether you can answer a given question and estimate the length of your response.

Provide your assessment in the following format:
Can answer: \boxed{[yes]/[no]}
Estimated length: \boxed{[X tokens]}

Examples:
Question: What is the derivative of x^2 + 3x?
Can answer: \boxed{yes}
Estimated length: \boxed{50 tokens}

Question: Solve Fermat's Last Theorem
Can answer: \boxed{no}
Estimated length: \boxed{N/A}

Evaluate the following question:
<question>
[ORIGINAL QUESTION HERE]
</question>

Can answer: \boxed{
```

### Step 2: Collect Predictions

The model generates responses like:
- `Can answer: \boxed{yes}` or `Can answer: \boxed{no}`
- `Estimated length: \boxed{150 tokens}` or `Estimated length: \boxed{N/A}`

### Step 3: Compare Against Baseline

We compare the model's predictions against the actual results from the unbounded run:

**Correctness Prediction:**
- If model predicted "yes" → check if it actually solved the problem correctly
- If model predicted "no" → check if it actually got it wrong

**Length Prediction:**
- Compare predicted token count vs actual token count
- Calculate error metrics: MAE, MAPE, RMSE

## Output Files

### Directory Structure

```
results/{model_id}/{dataset_name}/
└── meta_prediction/
    ├── answers/
    │   └── stdout_*.txt              # Raw model outputs
    ├── configs/
    │   └── config_*.json             # Run configuration
    ├── metrics/
    │   └── meta_prediction_report.csv # Evaluation results
    └── {dataset_name}_meta_prediction.json  # Transformed dataset
```

### Meta-Prediction Report CSV

The report contains the following columns:

| Column | Description |
|--------|-------------|
| `problem_id` | Problem ID from dataset |
| `predicted_can_answer` | Model's prediction: 'yes' or 'no' |
| `actual_correct` | Whether model actually got it correct (from baseline) |
| `correctness_prediction_correct` | True if prediction matches reality |
| `predicted_length` | Predicted token count |
| `actual_length` | Actual token count (from baseline) |
| `length_error` | Difference: predicted - actual |
| `length_relative_error` | Absolute percentage error |

## Evaluation Metrics

### Correctness Prediction Metrics

- **Accuracy**: Percentage of correct predictions about answer correctness
- **Precision (predicted "yes")**: When model predicts it can answer, how often is it actually correct?
- **Precision (predicted "no")**: When model predicts it cannot answer, how often is it actually incorrect?

### Length Prediction Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference in tokens
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Root of mean squared errors (penalizes large errors)

## Example Output

```
====================================================================
📊 META-PREDICTION EVALUATION RESULTS
====================================================================
Total problems evaluated: 100

--- CORRECTNESS PREDICTION ---
Accuracy: 68/100 (68.0%)
  Predicted 'yes' (can answer): 85 times
    Actually correct: 62 (72.9%)
  Predicted 'no' (cannot answer): 15 times
    Actually incorrect: 6 (40.0%)

--- LENGTH PREDICTION ---
Valid length predictions: 85
Mean Absolute Error (MAE): 127.3 tokens
Mean Absolute Percentage Error (MAPE): 42.5%
Root Mean Squared Error (RMSE): 189.7 tokens
====================================================================
```

## Use Cases

### 1. Self-Awareness Assessment
Understand how well the model can predict its own performance - a form of meta-cognition.

### 2. Confidence Calibration
Check if the model is over-confident (predicts "yes" but fails) or under-confident (predicts "no" but would succeed).

### 3. Token Budget Planning
Evaluate if the model can help with dynamic token budget allocation by predicting response lengths.

### 4. Problem Difficulty Estimation
Problems where the model predicts "no" might be inherently more difficult.

## Comparing Against Baseline

The meta-prediction experiment requires a baseline unbounded run. Make sure you have:

1. **Run the unbounded experiment first:**
   ```bash
   ./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
   ```

2. **Wait for completion and verify the baseline exists:**
   ```bash
   ls results/{model_id}/GSM8K/unbounded/metrics/verify_report_*.csv
   ```

3. **Then run meta-prediction:**
   ```bash
   ./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
   ```

## Troubleshooting

### "No baseline report found"

**Problem**: The verification step cannot find the baseline unbounded results.

**Solution**:
- Ensure you've run the unbounded experiment first
- Check that `results/{model_id}/{dataset_name}/unbounded/metrics/verify_report_*.csv` exists
- Specify baseline explicitly with `--baseline` flag

### "No valid predictions found"

**Problem**: The model didn't follow the expected output format.

**Solution**:
- Check the raw outputs in `answers/stdout_*.txt`
- Ensure the model is generating `\boxed{yes}` or `\boxed{no}` for correctness
- Ensure the model is generating `\boxed{X tokens}` for length

### Predictions are all "yes" or all "no"

**Problem**: The model might be gaming the system or doesn't understand the task.

**Solution**:
- Check the prompt template in `run_meta_prediction.py`
- Consider adding more examples to the prompt
- Try with different temperature settings

## Advanced Usage

### Custom Prompt Template

Edit the `META_PREDICTION_PROMPT_TEMPLATE` in `run_meta_prediction.py`:

```python
META_PREDICTION_PROMPT_TEMPLATE = """[Your custom prompt here]

Question: {question}

Can answer: \\boxed{{"""
```

### Running on a Subset

Test on a small subset first:

```bash
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 10
```

### Analyzing Results

Load the CSV report for detailed analysis:

```python
import pandas as pd

df = pd.read_csv('results/.../meta_prediction/metrics/meta_prediction_report.csv')

# Check prediction accuracy by actual difficulty
df['actual_difficulty'] = df['actual_correct'].map({True: 'easy', False: 'hard'})
print(df.groupby('actual_difficulty')['correctness_prediction_correct'].mean())

# Check length prediction accuracy by question length
df['length_bucket'] = pd.cut(df['actual_length'], bins=[0, 200, 400, 600, 1000])
print(df.groupby('length_bucket')['length_relative_error'].mean())
```

## Integration with Existing Workflow

The meta-prediction experiment is designed to work alongside your existing experiments:

```
results/{model_id}/{dataset_name}/
├── unbounded/              # Standard unbounded run (baseline)
├── p85/                    # 85% budget run
├── p65/                    # 65% budget run
├── p45/                    # 45% budget run
├── p25/                    # 25% budget run
└── meta_prediction/        # Meta-prediction experiment (NEW)
```

All experiments share the same dataset but test different aspects of model performance.

## Future Extensions

Potential enhancements to the meta-prediction experiment:

1. **Multi-turn prediction**: Ask model to refine its predictions after seeing examples
2. **Difficulty ranking**: Ask model to rank problems by difficulty
3. **Strategy selection**: Ask model to predict which reasoning strategy would work best
4. **Budget-aware prediction**: Ask model to predict performance under token constraints
