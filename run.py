from constants import DIR, CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN
import json
import subprocess
import time
import asyncio
import argparse
from utils import kill_server
import os
import re
import aiohttp
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN
import sys
from pathlib import Path
import glob
import csv

server_configs = []
i = 0
for alg in ['lru']:
    for size in [0.8]:
        server_configs.append({
            'host': 'localhost',
            'cuda_devices': f'CUDA_VISIBLE_DEVICES={i}',
            'eviction_algorithm': alg,
            'port': 8000+i,
            'size': size,
            'args': (
                f"--gpu_memory_utilization {size} "
                f"--pipeline-parallel-size 1 --port {8000+i} "
                f"--block_size=16 --max-num-batched-tokens 4096"
            )
        })
        i += 1
        
dataset_file = '~/vllm_cache_bench/AIME25_benchmark.json'
client_configs = [
    {
        'num_prompts': 20,
        'request_rate': 4,
    },
]

def run_server(server_config):
    """Start the server with specified parallel sizes."""
    log_file_name = f"{LOG_FILE}_{server_config['port']}_{server_config['eviction_algorithm']}.log"
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_config['args'])
    print('\n', server_cmd, '\n')

    # Combine CUDA device settings with the actual server command
    full_cmd = f"{server_config['cuda_devices']} {server_cmd}"

    with open(log_file_name, "w") as log_file:
        process = subprocess.Popen(full_cmd, shell=True, stdout=log_file, stderr=log_file)

    return log_file_name

def wait_for_server_ready(log_file_name, timeout=600):
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


async def get_queue_metrics(host, port):
    """Fetch queue time metrics from vLLM's Prometheus endpoint."""
    metrics_url = f"http://{host}:{port}/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    queue_metrics = {}
                    
                    # Parse relevant queue metrics
                    for line in metrics_text.split('\n'):
                        if 'time_in_queue_requests' in line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                metric_name = parts[0]
                                metric_value = float(parts[1])
                                queue_metrics[metric_name] = metric_value
                        elif 'request_queue_time_seconds' in line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                metric_name = parts[0]
                                metric_value = float(parts[1])
                                queue_metrics[metric_name] = metric_value
                    
                    return queue_metrics
                else:
                    print(f"Failed to fetch metrics: {response.status}")
                    return {}
    except Exception as e:
        print(f"Error fetching queue metrics: {e}")
        return {}

async def run_client(client_config, server_config, args=None, budgets_file=None):
    num_prompts = client_config['num_prompts']
    request_rate = client_config['request_rate']
    prefix = get_file_name(server_config)
    result_filename = f"{prefix}.json"
    
    # Extract host directly from the dictionary.
    host = server_config["host"]
    
    # Extract the port from the 'args' string.
    port_match = re.search(r'--port\s+(\d+)', server_config["args"])
    if not port_match:
        raise ValueError("Port not found in server configuration 'args'.")
    port = port_match.group(1)
    
    # Get baseline queue metrics before benchmark
    baseline_queue_metrics = await get_queue_metrics(host, port)
    
    # Create stdout file path for this experiment
    stdout_file = f"{DIR}/answers/stdout_{result_filename}.txt"
    
    client_cmd = CLIENT_CMD_TEMPLATE.format(
        dataset_file, host, port, result_filename, num_prompts, request_rate, stdout_file
    )
    # If a budgets file is provided, append it to the client command so
    # the benchmark client will attach per-request token budgets.
    if budgets_file:
        client_cmd = client_cmd + f" --budgets-file {budgets_file}"
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
    
    # Get final queue metrics after benchmark
    final_queue_metrics = await get_queue_metrics(host, port)
    
    hit_ratios = []
    for line in stdout.decode().split("\n"):
        if 'gpu_prefix_cache_hit_rate' in line:
            hit_ratios.append(line.split()[-1])
    
    with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
        json.dump([client_config, server_config], fp)
    print('\n' + client_cmd + '\n')
    result = {'hit_ratios': hit_ratios}
    
    # Calculate queue metrics differences (metrics from this benchmark run)
    queue_metrics_diff = {}
    for key in final_queue_metrics:
        if key in baseline_queue_metrics:
            queue_metrics_diff[key] = final_queue_metrics[key] - baseline_queue_metrics[key]
        else:
            queue_metrics_diff[key] = final_queue_metrics[key]
    
    result['queue_metrics'] = queue_metrics_diff
    
    for k in ['eviction_algorithm', 'size']:
        result[k] = server_config[k]
    for k in client_config.keys():
        result[k] = client_config[k]
    return result

async def start_server(server_config):
    # Launch the server and wait for it to be ready.
    log_file_name = await asyncio.to_thread(run_server, server_config)
    print("wait_for_server_ready:", log_file_name)
    is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
    return is_ready

async def start_exp(server_config, client_configs, args=None, budgets_file=None):
    print("Starting server configuration:", server_config)
    await start_server(server_config)
    
    results = []
    for client_config in client_configs:
        result = await run_client(client_config, server_config, args=args, budgets_file=budgets_file)
        results.append(result)

    # Save results to `exp.json` in append mode
    exp_file = f"{DIR}/exp.json"
    
    # Load existing data if the file exists
    if os.path.exists(exp_file):
        with open(exp_file, "r") as fp:
            try:
                existing_data = json.load(fp)
                if not isinstance(existing_data, list):
                    existing_data = []  # Reset if data is corrupted
            except json.JSONDecodeError:
                existing_data = []  # Reset if file is empty or corrupted
    else:
        existing_data = []

    # Append new results
    existing_data.extend(results)

    # Write back to the file
    with open(exp_file, "w") as fp:
        json.dump(existing_data, fp, indent=4)

    print(f"Saved results to {exp_file}")

async def main(args):
    # Stop any running server first
    kill_server(server_configs[0]['host'])

    budgets_file_to_pass = None

    # === DRY RUN ===
    if args.dry_run:
        print("\n=== DRY RUN: Running benchmark without token budgets ===")
        tasks = [
            start_exp(server_config, client_configs, args=args, budgets_file=None)
            for server_config in server_configs
        ]
        await asyncio.gather(*tasks)
        print("Dry run complete. You can now compute token budgets with --compute-budgets.")
        return

    # === COMPUTE BUDGETS ===
    if args.compute_budgets:
        answers_dir = Path(DIR) / 'answers'
        pattern = str(answers_dir / 'stdout_*.txt')
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not candidates:
            raise SystemExit(f"No prior stdout files found in {answers_dir}; cannot compute budgets")
        latest_stdout = candidates[-1]
        print(f"Computing token budgets from latest stdout: {latest_stdout}")
        token_budgets_path = str(Path(DIR) / 'token_budgets.csv')

        ret = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / 'compute_token_budgets.py'),
            '--stdout-file', latest_stdout,
            '--out', token_budgets_path,
            '--test-jsonl', str(Path(__file__).parent / 'test.jsonl')
        ])
        if ret.returncode != 0:
            raise SystemExit('compute_token_budgets.py failed')

        # Convert to minimal budgets csv expected by benchmark client
        percentile_col = f'budget_{args.budgets_percentile}'
        budgets_csv = Path(DIR) / f'budgets_for_client_{args.budgets_percentile}.csv'
        with open(token_budgets_path, 'r', encoding='utf-8') as inf, open(budgets_csv, 'w', encoding='utf-8', newline='') as outf:
            reader = csv.DictReader(inf)
            writer = csv.DictWriter(outf, fieldnames=['index', 'unique_id', 'token_budget'])
            writer.writeheader()
            for row in reader:
                tok = row.get(percentile_col)
                if tok is None or tok == '':
                    continue
                writer.writerow({
                    'index': row.get('index', ''),
                    'unique_id': row.get('unique_id', ''),
                    'token_budget': tok
                })
        print(f"Budgets file written to {budgets_csv}")
        return

    # === RUN WITH BUDGETS ===
    if args.run_with_budgets or args.budgets_file:
        budgets_file_to_pass = args.budgets_file
        if not budgets_file_to_pass:
            budgets_file_to_pass = str(Path(DIR) / f'budgets_for_client_{args.budgets_percentile}.csv')
        print(f"\n=== RUN WITH BUDGETS: Using {budgets_file_to_pass} ===")
        tasks = [
            start_exp(server_config, client_configs, args=args, budgets_file=budgets_file_to_pass)
            for server_config in server_configs
        ]
        await asyncio.gather(*tasks)

    # === VERIFY OUTPUTS ===
    if args.verify_after:
        answers_dir = Path(DIR) / 'answers'
        pattern = str(answers_dir / 'stdout_*.txt')
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not candidates:
            print(f"No stdout files found in {answers_dir} to verify")
            return
        latest_stdout = candidates[-1]
        print(f"Running verification against {latest_stdout}")
        report_path = Path(DIR) / 'verify_report.csv'
        ret = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / 'verify_outputs.py'),
            '--stdout-file', latest_stdout,
            '--test-jsonl', str(Path(__file__).parent / 'test.jsonl'),
            '--out', str(report_path)
        ])
        if ret.returncode != 0:
            print('verify_outputs.py failed')
        else:
            print(f'Verification report written to {report_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vllm cache benchmark in dry-run, budgeted, or verify mode"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without token budgets (collects raw token usage)")
    parser.add_argument("--compute-budgets", action="store_true",
                        help="Compute token budgets from the last dry run's stdout")
    parser.add_argument("--run-with-budgets", action="store_true",
                        help="Run again using computed or given token budgets")
    parser.add_argument("--budgets-file", default="",
                        help="Path to a precomputed budgets CSV to use in budgeted run")
    parser.add_argument("--budgets-percentile", type=int, default=50,
                        help="Percentile column to use when converting token budgets (e.g. 50 for budget_50)")
    parser.add_argument("--verify-after", action="store_true",
                        help="Verify latest outputs using verify_outputs.py")

    if "SLURM_JOB_ID" in os.environ:
        parser.set_defaults(compute_budgets=True, verify_after=True)

    args = parser.parse_args()
    asyncio.run(main(args))
    
