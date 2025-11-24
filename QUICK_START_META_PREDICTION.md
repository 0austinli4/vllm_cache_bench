# Quick Start: Meta-Prediction Experiment

## What is this?

Tests if the LLM can predict:
1. ✅ Will it answer correctly? (yes/no)
2. 📏 How many tokens will it use? (number)

## Prerequisites

You must have already run the **unbounded experiment** to get baseline results:

```bash
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

Wait for it to complete and verify you have:
```bash
ls results/*/GSM8K/unbounded/metrics/verify_report_*.csv
```

## Run Meta-Prediction

### Option 1: SLURM (Recommended)

```bash
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

Monitor progress:
```bash
squeue -u $USER
tail -f results/*/GSM8K/slurm_logs/meta_prediction-*.out
```

### Option 2: Local (for testing)

```bash
# Activate environment
module load anaconda3/2024.6
conda activate dynasor

# Run meta-prediction
python run_meta_prediction.py \
    --dataset datasets/GSM8K_benchmark.json \
    --num-prompts 10  # Start small for testing

# Verify predictions
python verify_meta_predictions.py \
    --meta-outputs results/*/GSM8K/meta_prediction/answers/stdout_*.txt \
    --baseline-report results/*/GSM8K/unbounded/metrics/verify_report_*.csv \
    --out results/*/GSM8K/meta_prediction/metrics/meta_prediction_report.csv
```

## Understanding Results

The verification script will print something like:

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

### What do these numbers mean?

**Correctness Prediction:**
- **Accuracy 68%**: Model correctly predicted whether it would get the answer right 68% of the time
- **Predicted 'yes' → Actually correct 72.9%**: When confident, it's right 73% of the time
- **Predicted 'no' → Actually incorrect 40%**: When not confident, it's still often wrong in its prediction

**Length Prediction:**
- **MAE 127.3 tokens**: On average, off by ~127 tokens per prediction
- **MAPE 42.5%**: On average, predictions are ~42% off from actual
- **RMSE 189.7 tokens**: Some predictions are way off (high variance)

## Example Output Files

### Meta-Prediction Report CSV

```csv
problem_id,predicted_can_answer,actual_correct,correctness_prediction_correct,predicted_length,actual_length,length_error,length_relative_error
4531,yes,True,True,200,188,12,6.38%
6018,yes,True,True,250,261,-11,4.21%
4121,no,False,True,,,
3209,yes,False,False,150,149,1,0.67%
```

## Common Use Cases

### 1. Test on Small Sample First

```bash
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 10
```

### 2. Compare Multiple Datasets

```bash
# Run on GSM8K
./submit_meta_prediction.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100

# Run on AIME25
./submit_meta_prediction.sh --dataset datasets/AIME25_benchmark.json --num-prompts 30

# Compare: is the model better at predicting performance on easy vs hard problems?
```

### 3. Analyze Results in Python

```python
import pandas as pd

# Load results
df = pd.read_csv('results/.../meta_prediction/metrics/meta_prediction_report.csv')

# Confidence calibration: is the model overconfident?
overconfident = df[(df['predicted_can_answer'] == 'yes') & (df['actual_correct'] == False)]
print(f"Overconfident: {len(overconfident)} / {len(df[df['predicted_can_answer'] == 'yes'])}")

# Length prediction accuracy by actual difficulty
df['got_it_right'] = df['actual_correct']
print(df.groupby('got_it_right')['length_relative_error'].describe())
```

## Troubleshooting

### "No baseline report found"

Run the unbounded experiment first:
```bash
./submit_experiment.sh --dataset datasets/GSM8K_benchmark.json --num-prompts 100
```

### "No valid predictions found"

Check the raw outputs to see if the model is following the format:
```bash
tail -n 50 results/*/GSM8K/meta_prediction/answers/stdout_*.txt
```

Look for: `\boxed{yes}` or `\boxed{no}` and `\boxed{150 tokens}`

### Job fails immediately

Check the SLURM log:
```bash
ls -lt results/*/GSM8K/slurm_logs/
tail -f results/*/GSM8K/slurm_logs/meta_prediction-*.out
```

## Next Steps

- Read [META_PREDICTION_README.md](META_PREDICTION_README.md) for detailed documentation
- See [CLAUDE.md](CLAUDE.md) for full project documentation
- Customize the prompt in `run_meta_prediction.py` for different experiments

## Questions?

The meta-prediction experiment is designed to:
1. Understand model self-awareness
2. Identify overconfident predictions
3. Enable dynamic token budget allocation
4. Assess problem difficulty estimation

Think of it as asking the model: "Before you solve this, can you tell me if you'll get it right and how long it will take?"
