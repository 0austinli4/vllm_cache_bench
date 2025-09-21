from constants import DIR, CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN, SGLANG_SERVER_CMD_TEMPLATE
import json
import subprocess
import time
import asyncio
from utils import kill_server
import os
import re
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process, launch_server_cmd
import sys
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import os
from tqdm import tqdm

server_configs = []
i = 0
for alg in ['lru']:
    for size in [0.02]:
        server_configs.append({
            'host': 'localhost', 
            'eviction_algorithm': alg,
            'size': size,
            'port': 8000 + i, 
            'cuda_devices': f'CUDA_VISIBLE_DEVICES={i}',
            'args': (
                f"--host localhost "
                f"--port {8000 + i}"
            ),
            'total_tokens': 0
        })
        i += 1

dataset = 'sharegpt'
dataset_file = '/scratch/gpfs/al2926/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json'
dataset_tay = '/home/al2926/vllm_cache_bench/sglang_tay.json'

client_configs = [
    {
        'num_prompts': 10, 
        'request_rate': 0.08,
    }
]

def run_server(server_config):
    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd

    # Launch the server
    # --mem-fraction-static 0.5 
    # --max-running-requests 2 
    # --max-total-tokens 1000000
    # --mem-fraction-static {server_config[size]} 
    print("RUNNING WITH TOKEN NUMBER: ", server_config['total_tokens'], server_config['alg'])
    # --max-total-tokens {server_config['total_tokens']}
    server_process, port = launch_server_cmd(
        f"{sys.executable} -m sglang.launch_server --model-path /scratch/gpfs/al2926/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --host 0.0.0.0 --chunked-prefill-size -1 --decode-log-interval 500 --enable-metrics --disable-outlines-disk-cache"
    )

    # sys.executable
    print("Server running on port", port)
    server_config['port'] = port
    log_file_name = f"{LOG_FILE}_{server_config['port']}_{server_config['eviction_algorithm']}.log"
    
    with open(log_file_name, "w") as log_file:
        log_file.write(f"Log file created for port {server_config['port']} with eviction algorithm {server_config['eviction_algorithm']}.\n")
    
    # Wait for the server to be ready
    wait_for_server(f"http://localhost:{port}")

    return log_file_name

def wait_for_server_ready(log_file_name, timeout=150):
    """Wait until the server is ready or a timeout occurs."""
    for _ in range(timeout):
        if os.path.exists(log_file_name):
            with open(log_file_name, "r") as log_file:
                log_data = log_file.read()
                if re.search(SERVER_READY_PATTERN, log_data):
                    print("Server is ready.")
                    return True
                if re.search(CUDA_OOM_PATTERN, log_data) or re.search(ERROR_PATTERN, log_data) or re.search(RAISE_PATTERN, log_data):
                    print("Server encountered an error.")
                    return False
        time.sleep(1)
    print("Server startup timed out.")
    return False

def get_file_name(server_config):
    # Use a simple timestamp for a unique result filename.
    return str(int(time.time()))

async def run_client(client_config, server_config):
    num_prompts = client_config['num_prompts']
    request_rate = client_config['request_rate']
    prefix = get_file_name(server_config)
    result_filename = f"{prefix}.json"
    
    # Extract host/port directly from the dictionary.
    host = server_config["host"]
    port = server_config['port']
    
    client_cmd = CLIENT_CMD_TEMPLATE.format(
        dataset_tay, dataset, host, port, result_filename, num_prompts, request_rate
    )
    print("Running client command:", client_cmd)
    
    # Use asyncio's subprocess shell to run the client command asynchronously.
    process = await asyncio.create_subprocess_shell(
        client_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print("Client stdout:", stdout.decode())
    print("Client stderr:", stderr.decode())
    hit_ratios = []

    ## do it by std out of stats
    for line in stdout.decode().split("\n"):
        if 'cache_hit_rate' in line and len(line.split()) > 0 and line.split()[0] != "#":
            hit_ratios.append(line.split()[-1])
    
    # with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
    #     json.dump([client_config, server_config], fp)
    print('\n' + client_cmd + '\n')
    result = {'hit_ratios': hit_ratios}
    for k in ['eviction_algorithm', 'size']:
        result[k] = server_config[k]
    for k in client_config.keys():
        result[k] = client_config[k]
    return result

async def start_server(server_config):
    # Launch the server and wait for it to be ready.
    log_file_name = await asyncio.to_thread(run_server, server_config)
    is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
    return is_ready, log_file_name

async def start_exp(server_config, client_configs):
    print("Starting server configuration")
    is_ready, log_file_name = await start_server(server_config)
    
    # results = []
    # for client_config in client_configs:
    #     result = await run_client(client_config, server_config)
    #     results.append(result)

    # # Save results to `exp.json` in append mode
    # exp_file = f"{DIR}/lru_results.json"
    
    # # Load existing data if the file exists
    # if os.path.exists(exp_file):
    #     with open(exp_file, "r") as fp:
    #         try:
    #             existing_data = json.load(fp)
    #             if not isinstance(existing_data, list):
    #                 existing_data = []  # Reset if data is corrupted
    #         except json.JSONDecodeError:
    #             existing_data = []  # Reset if file is empty or corrupted
    # else:
    #     existing_data = []

    # # Append new results
    # existing_data.extend(results)

    # Write back to the file
    # with open(exp_file, "w") as fp:
    #     json.dump(existing_data, fp, indent=4)

    # print(f"Saved results to {exp_file}")

async def main():
    # Stop any running server on this node.
    kill_server(server_configs[0]['host'])
    tasks = [start_exp(server_config, client_configs) for server_config in server_configs]
    await asyncio.gather(*tasks)

# execute the rest of the script using myparam
if __name__ == "__main__":
    alg = sys.argv[1]
    asyncio.run(main())
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # /scratch/gpfs/al2926/hf_models
    model_path = "/scratch/gpfs/al2926/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))