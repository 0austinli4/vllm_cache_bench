import os

# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/meta-llama/Llama-3.2-3B-Instruct"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/meta-llama/Llama-3.1-8B-Instruct"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/openai/gpt-oss-20b"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/mistralai/Mistral-7B-v0.1"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Qwen/Qwen2.5-1.5B-Instruct"
# MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL = "/scratch/gpfs/WLLOYD/al2926/hf_models/Qwen/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

os.makedirs(f'{DIR}/configs', exist_ok=True)
os.makedirs(f'{DIR}/metrics', exist_ok=True)

# --num-scheduler-steps 1
VLLM_SERVER_CMD_TEMPLATE = (
    "VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO vllm serve {model} --disable-log-requests "
    "--max_num_seqs 512 --max-model-len 16384 --disable_custom_all_reduce "
    "--enable-chunked-prefill --enable-prefix-caching --quantization=fp8 "
    "{args}"
)

CLIENT_CMD_TEMPLATE = (
    f"python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    f"--save-result --model {MODEL} --endpoint /v1/completions "
    "--dataset-path {} --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {} "
    "--percentile-metrics ttft,tpot,itl,e2el --stdout-file {}"
)

SERVER_READY_PATTERN = r"startup complete"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"!error!"
RAISE_PATTERN = r"raise"