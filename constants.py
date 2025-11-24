import os
from pathlib import Path

# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Qwen/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Xin-Rui/models--Xin-Rui--BudgetThinker-1.5b/snapshots/b7304c41480eca765e2f77b5a6641133c64c1ea7"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Salesforce/models--Salesforce--E1-Math-1.5B/snapshots/3fa7501acb8aec762347e70be9a04a6324c886fc"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Salesforce/models--Salesforce--E1-Math-7B/snapshots/d5c4a114d31f5a485603704143291b38ac65d052"
MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Extract model ID from path
MODEL_ID = MODEL.split('/')[-1]

def get_results_dir(dataset_name, run_name="unbounded"):
    """
    Get results directory for a specific dataset and run.

    Args:
        dataset_name: Name of the dataset (e.g., 'AIME25', 'GSM8K')
        run_name: Name of the run (e.g., 'unbounded', 'p85', 'p65', 'p45', 'p25')

    Returns:
        Path to the results directory
    """
    results_dir = Path(f"results/{MODEL_ID}/{dataset_name}/{run_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (results_dir / "answers").mkdir(exist_ok=True)
    (results_dir / "configs").mkdir(exist_ok=True)
    (results_dir / "metrics").mkdir(exist_ok=True)

    return str(results_dir)

def get_dataset_base_dir(dataset_name):
    """
    Get base directory for a dataset (contains all runs).

    Args:
        dataset_name: Name of the dataset

    Returns:
        Path to the dataset directory
    """
    dataset_dir = Path(f"results/{MODEL_ID}/{dataset_name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return str(dataset_dir)

# Legacy: Default DIR for backwards compatibility (will be overridden in run.py)
DIR = f"results/{MODEL_ID}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

# --num-scheduler-steps 1
VLLM_SERVER_CMD_TEMPLATE = (
    "VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO vllm serve {model} --disable-log-requests "
    "--max_num_seqs 512 --disable_custom_all_reduce "
    "--enable-chunked-prefill --enable-prefix-caching "
    "{args}"
)
#  --quantization=fp8 
# --max-model-len 4096 
# --max-model-len 4096

    # "--enable-reasoning --reasoning-parser deepseek_r1 "
    
CLIENT_CMD_TEMPLATE = (
    f"python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    f"--model {MODEL} --dataset-name sharegpt --endpoint /v1/completions "
    "--dataset-path {} --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {} "
    "--percentile-metrics ttft,tpot,itl,e2el --stdout-file {}"
)

SERVER_READY_PATTERN = r"startup complete"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"!error!"
RAISE_PATTERN = r"raise"