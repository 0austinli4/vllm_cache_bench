# Dataset Converters

This directory contains tools for converting various math benchmark datasets into the format required by the vLLM benchmarking workflow.

## Available Tools

### `convert_dataset.py`
Universal converter script that handles multiple dataset formats.

**Usage:**
```bash
python convert_dataset.py \
    --input <input_file> \
    --output-dir <output_directory> \
    [--dataset-type <type>] \
    [--output-name <name>] \
    [--sample-size <n>]
```

**Arguments:**
- `--input`: Path to input dataset file (JSON or JSONL)
- `--output-dir`: Output directory for converted files (default: `datasets/`)
- `--dataset-type`: Dataset type (auto-detected if not specified)
- `--output-name`: Output filename prefix (auto-generated if not specified)
- `--sample-size`: Limit to first N problems (optional, for testing)

**Examples:**
```bash
# Convert GSM8K (auto-detect type)
python convert_dataset.py \
    --input ../datasets/gsm8k_test.json \
    --output-dir ../datasets/

# Convert with explicit type
python convert_dataset.py \
    --input ../datasets/mathbench_college.jsonl \
    --output-dir ../datasets/ \
    --dataset-type mathbench_college

# Convert subset for testing
python convert_dataset.py \
    --input ../datasets/gsm8k_test.json \
    --output-dir ../datasets/ \
    --sample-size 100
```

### `convert_all_datasets.sh`
Convenience script to convert all datasets at once.

**Usage:**
```bash
./convert_all_datasets.sh
```

This automatically converts:
- GSM8K (7,473 problems)
- MathBench-Arithmetic (300 problems)
- MathBench-College (150 problems)
- MathBench-Middle (150 problems)
- MathBench-High (150 problems)

### `dataset_registry.py`
Central registry for dataset configurations.

Defines:
- Answer extraction logic for each dataset type
- Prompt formatting (especially for multiple choice)
- Dataset metadata

**Supported Dataset Types:**
- `gsm8k`: Grade school math with solution and final answer
- `mathbench_arithmetic`: Elementary arithmetic with numeric answers
- `mathbench_college`: College-level multiple choice
- `mathbench_middle`: Middle school multiple choice
- `mathbench_high`: High school multiple choice
- `aime25`: Math competition (already in correct format)

## Output Formats

Each conversion produces two files:

### 1. Benchmark File (`*_benchmark.json`)
Used by vLLM serving for inference.

```json
[
  {
    "id": "1",
    "conversations": [
      {"from": "human", "value": "question text"},
      {"from": "gpt", "value": "Solution will be generated"}
    ]
  }
]
```

### 2. Answer File (`*.jsonl`)
Used for answer verification.

```jsonl
{"index": 1, "problem": "question text", "answer": "final answer"}
{"index": 2, "problem": "question text", "answer": "final answer"}
```

## Dataset-Specific Formats

### GSM8K
**Input format:**
```json
{
  "question": "problem text",
  "answer": "solution text\n#### 42"
}
```

**Answer extraction:** Extract number after `####`

### MathBench Arithmetic
**Input format:**
```json
{
  "question": "problem text",
  "answer": "numeric answer",
  "topic": "topic name"
}
```

**Answer extraction:** Direct numeric value

### MathBench Multiple Choice
**Input format:**
```json
{
  "question": "problem text",
  "options": ["option A", "option B", "option C", "option D"],
  "answer": "A",
  "topic": "topic name"
}
```

**Answer extraction:** Letter (A/B/C/D)

**Prompt formatting:** Options are listed as:
```
Question text
Choose from the following options:
A. option A
B. option B
C. option C
D. option D

Provide your answer as a single letter (A, B, C, or D) in \boxed{}.
```

## Adding New Dataset Types

To add support for a new dataset type:

1. **Define extraction function** in `dataset_registry.py`:
```python
def extract_my_dataset_answer(answer_text: str) -> str:
    # Custom logic to extract final answer
    return final_answer
```

2. **Register the dataset** in `dataset_registry.py`:
```python
DATASETS = {
    'my_dataset': DatasetConfig(
        name='My Dataset',
        answer_format='custom_format',
        extract_answer=extract_my_dataset_answer,
        format_prompt=None  # Optional custom prompt formatter
    ),
    # ... other datasets
}
```

3. **Add to conversion script** (optional):
Edit `convert_all_datasets.sh` to include your dataset in batch conversions.

## Verification

The converted datasets are automatically compatible with `verify_outputs.py`, which supports:
- Numeric answers (exact and approximate matching)
- Multiple choice answers (letter matching)
- LaTeX `\boxed{}` notation extraction
- Token overlap fallback for complex answers

## Testing

Before running a full experiment, test conversions with sample sizes:

```bash
# Test GSM8K conversion with 10 problems
python convert_dataset.py \
    --input ../datasets/gsm8k_test.json \
    --output-dir ../datasets/ \
    --sample-size 10 \
    --output-name GSM8K-test

# Run quick benchmark test
cd ..
./submit_experiment.sh \
    --dataset datasets/GSM8K-test_benchmark.json \
    --num-prompts 10
```
