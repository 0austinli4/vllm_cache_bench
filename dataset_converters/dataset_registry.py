"""
dataset_registry.py
-------------------
Central registry for dataset configurations and answer extraction logic.

Each dataset type has:
- answer_format: How answers are formatted in the original dataset
- extract_answer: Function to extract the final answer
- format_prompt: Optional function to format the question as a prompt
"""

import re
from typing import Dict, Any, Callable, Optional


class DatasetConfig:
    """Configuration for a dataset type."""

    def __init__(
        self,
        name: str,
        answer_format: str,
        extract_answer: Callable[[str], str],
        format_prompt: Optional[Callable[[str], str]] = None
    ):
        self.name = name
        self.answer_format = answer_format
        self.extract_answer = extract_answer
        self.format_prompt = format_prompt or (lambda x: x)


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract final answer from GSM8K format.
    Format: "solution text\n#### 42"
    """
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole answer
    return answer_text.strip()


def extract_numeric_answer(answer_text: str) -> str:
    """
    Extract numeric answer (already in final form).
    """
    return str(answer_text).strip()


def extract_multiple_choice_answer(answer_letter: str) -> str:
    """
    Extract multiple choice answer letter.
    Format: "A", "B", "C", or "D"
    """
    return str(answer_letter).strip().upper()


def format_multiple_choice_prompt(question: str, options: list) -> str:
    """
    Format multiple choice question with options.
    """
    prompt = question.strip()
    if not prompt.endswith('?'):
        prompt += '\n'
    else:
        prompt += ' '

    prompt += "Choose from the following options:\n"
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D
        prompt += f"{letter}. {option}\n"

    prompt += "\nProvide your answer as a single letter (A, B, C, or D) in \\boxed{{}}."
    return prompt


# Dataset registry
DATASETS = {
    'gsm8k': DatasetConfig(
        name='GSM8K',
        answer_format='solution_with_final',  # "solution\n#### answer"
        extract_answer=extract_gsm8k_answer
    ),
    'gsm8k_zero': DatasetConfig(
        name='GSM8K-Zero',
        answer_format='solution_with_final',
        extract_answer=extract_gsm8k_answer
    ),
    'mathbench_arithmetic': DatasetConfig(
        name='MathBench-Arithmetic',
        answer_format='numeric',  # Direct numeric answer
        extract_answer=extract_numeric_answer
    ),
    'mathbench_college': DatasetConfig(
        name='MathBench-College',
        answer_format='multiple_choice',  # Letter A/B/C/D
        extract_answer=extract_multiple_choice_answer,
        format_prompt=format_multiple_choice_prompt
    ),
    'mathbench_middle': DatasetConfig(
        name='MathBench-Middle',
        answer_format='multiple_choice',
        extract_answer=extract_multiple_choice_answer,
        format_prompt=format_multiple_choice_prompt
    ),
    'mathbench_high': DatasetConfig(
        name='MathBench-High',
        answer_format='multiple_choice',
        extract_answer=extract_multiple_choice_answer,
        format_prompt=format_multiple_choice_prompt
    ),
    'aime25': DatasetConfig(
        name='AIME25',
        answer_format='numeric',  # Already in correct format
        extract_answer=extract_numeric_answer
    )
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'mathbench_college')

    Returns:
        DatasetConfig object

    Raises:
        ValueError if dataset is not registered
    """
    # Normalize dataset name
    dataset_key = dataset_name.lower().replace('-', '_')

    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {', '.join(DATASETS.keys())}"
        )

    return DATASETS[dataset_key]


def infer_dataset_type(filename: str) -> Optional[str]:
    """
    Infer dataset type from filename.

    Args:
        filename: Dataset filename

    Returns:
        Dataset key or None if cannot infer
    """
    filename_lower = filename.lower()

    if 'gsm8k' in filename_lower:
        if 'zero' in filename_lower:
            return 'gsm8k_zero'
        return 'gsm8k'
    elif 'mathbench' in filename_lower:
        if 'arithmetic' in filename_lower:
            return 'mathbench_arithmetic'
        elif 'college' in filename_lower:
            return 'mathbench_college'
        elif 'middle' in filename_lower:
            return 'mathbench_middle'
        elif 'high' in filename_lower:
            return 'mathbench_high'
    elif 'aime' in filename_lower:
        return 'aime25'

    return None
