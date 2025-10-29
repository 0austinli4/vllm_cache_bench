#!/usr/bin/env python3
"""
convert_dataset.py
------------------
Universal dataset converter that transforms various math benchmark datasets
into the format required by the vLLM benchmarking workflow.

Supports:
- GSM8K (question + solution with #### answer)
- MathBench Arithmetic (question + direct numeric answer)
- MathBench Multiple Choice (question + options + letter answer)

Output formats:
1. {dataset}_benchmark.json - For vLLM serving (benchmark format)
2. {dataset}.jsonl - For answer verification

Usage:
    python convert_dataset.py --input datasets/gsm8k_test.json --output-dir datasets/
    python convert_dataset.py --input datasets/mathbench_college.jsonl --output-dir datasets/
"""

import argparse
import json
import sys
from pathlib import Path
from dataset_registry import get_dataset_config, infer_dataset_type


def load_json_or_jsonl(filepath: Path):
    """Load either JSON or JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # JSON array
            return json.load(f)
        else:
            # JSONL
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data


def convert_to_benchmark_format(data: list, dataset_config, sample_size=None) -> list:
    """
    Convert dataset to vLLM benchmark format.

    Format:
    [
      {
        "id": "1",
        "conversations": [
          {"from": "human", "value": "question"},
          {"from": "gpt", "value": "Solution will be generated"}
        ]
      }
    ]
    """
    benchmark = []

    # Limit sample size if specified
    if sample_size:
        data = data[:sample_size]

    for i, item in enumerate(data, 1):
        # Extract question
        question = item.get('question', '')

        # For multiple choice, format with options
        if dataset_config.answer_format == 'multiple_choice':
            options = item.get('options', [])
            prompt = dataset_config.format_prompt(question, options)
        else:
            prompt = question

        benchmark_item = {
            "id": str(i),
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": "Solution will be generated"}
            ]
        }
        benchmark.append(benchmark_item)

    return benchmark


def convert_to_answer_format(data: list, dataset_config, sample_size=None) -> list:
    """
    Convert dataset to answer verification format.

    Format (JSONL):
    {"index": 1, "problem": "question", "answer": "final_answer"}
    """
    answers = []

    # Limit sample size if specified
    if sample_size:
        data = data[:sample_size]

    for i, item in enumerate(data, 1):
        question = item.get('question', '')
        raw_answer = item.get('answer', '')

        # Extract final answer using dataset-specific logic
        final_answer = dataset_config.extract_answer(raw_answer)

        answer_item = {
            "index": i,
            "problem": question,
            "answer": final_answer
        }
        answers.append(answer_item)

    return answers


def main():
    parser = argparse.ArgumentParser(
        description="Convert math dataset to vLLM benchmark format"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input dataset file (JSON or JSONL)'
    )
    parser.add_argument(
        '--output-dir',
        default='datasets/',
        help='Output directory (default: datasets/)'
    )
    parser.add_argument(
        '--dataset-type',
        default=None,
        help='Dataset type (auto-detected if not specified). Options: gsm8k, mathbench_arithmetic, mathbench_college, etc.'
    )
    parser.add_argument(
        '--output-name',
        default=None,
        help='Output filename prefix (auto-detected if not specified)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Limit to first N problems (optional)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify input exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    # Infer dataset type if not specified
    if args.dataset_type:
        dataset_type = args.dataset_type
    else:
        dataset_type = infer_dataset_type(input_path.name)
        if not dataset_type:
            print(f"❌ Error: Could not infer dataset type from filename: {input_path.name}")
            print("   Please specify --dataset-type")
            return 1

    print(f"🔍 Dataset type: {dataset_type}")

    # Get dataset configuration
    try:
        dataset_config = get_dataset_config(dataset_type)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    print(f"📋 Dataset: {dataset_config.name}")
    print(f"📝 Answer format: {dataset_config.answer_format}")

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        # Auto-generate from dataset type
        output_name = dataset_type.upper().replace('_', '-')

    # Load input data
    print(f"\n📂 Loading: {input_path}")
    data = load_json_or_jsonl(input_path)
    print(f"✅ Loaded {len(data)} problems")

    if args.sample_size:
        print(f"🎯 Limiting to first {args.sample_size} problems")

    # Convert to benchmark format
    print(f"\n🔄 Converting to benchmark format...")
    benchmark_data = convert_to_benchmark_format(data, dataset_config, args.sample_size)

    benchmark_file = output_dir / f"{output_name}_benchmark.json"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Benchmark file: {benchmark_file}")

    # Convert to answer format
    print(f"\n🔄 Converting to answer format...")
    answer_data = convert_to_answer_format(data, dataset_config, args.sample_size)

    answer_file = output_dir / f"{output_name}.jsonl"
    with open(answer_file, 'w', encoding='utf-8') as f:
        for item in answer_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Answer file: {answer_file}")

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Dataset:       {dataset_config.name}")
    print(f"Problems:      {len(benchmark_data)}")
    print(f"Benchmark:     {benchmark_file}")
    print(f"Answers:       {answer_file}")
    print(f"\nReady to run experiment:")
    print(f"  ./submit_experiment.sh --dataset {benchmark_file.name} --num-prompts {len(benchmark_data)}")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
