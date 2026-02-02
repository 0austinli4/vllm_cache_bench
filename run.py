from constants import CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN
import json
import subprocess
import time
import asyncio
import argparse
from utils import kill_server
import os
import re
import sys
from pathlib import Path
import glob
import csv

# Global variables - will be set based on command-line arguments
DIR = None
dataset_file = None
answers_file = None

server_configs = []
i = 0
for size in [0.8]:
    server_configs.append({
        'host': 'localhost',
        'cuda_devices': f'CUDA_VISIBLE_DEVICES={i}',
        'port': 8000+i,
        'gpu_memory_utilization': size,
        'args': (
            f"--gpu_memory_utilization {size} "
            f"--pipeline-parallel-size 1 --port {8000+i} "
            f"--block_size=16 --max-num-batched-tokens 16384"
        )
    })
    i += 1

client_configs = [
    {
        'num_prompts': 10,
        'request_rate': 0.2,
    },
]

def run_server(server_config):
    """Start the server with specified parallel sizes."""
    from constants import MODEL

    # Include SLURM job ID in log filename for parallel job isolation
    slurm_job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID') or 'local'
    log_file_name = f"{LOG_FILE}_{server_config['port']}_{slurm_job_id}.log"

    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(model=MODEL, args=server_config['args'])
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
    # If running under SLURM, append the job ID to ensure uniqueness across concurrent jobs
    timestamp = str(int(time.time()))
    slurm_job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')

    if slurm_job_id:
        return f"{timestamp}_{slurm_job_id}"
    else:
        return timestamp



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

    # Create stdout file path for this experiment
    stdout_file = f"{DIR}/answers/stdout_{result_filename}.txt"

    # Clear the stdout file if it exists to prevent appending from previous runs
    if os.path.exists(stdout_file):
        os.remove(stdout_file)
        print(f"Cleared existing stdout file: {stdout_file}")

    client_cmd = CLIENT_CMD_TEMPLATE.format(
        dataset_file, host, port, result_filename, num_prompts, request_rate, stdout_file
    )

    # Add budgets file if provided
    if budgets_file:
        client_cmd = client_cmd + f" --budgets-file {budgets_file}"

    print("Running client command:", client_cmd)

    # Prepend CUDA device settings to client command to match server environment
    full_client_cmd = f"{server_config['cuda_devices']} {client_cmd}"

    # Use asyncio's subprocess shell to run the client command asynchronously.
    process = await asyncio.create_subprocess_shell(
        full_client_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print("Client stdout:", stdout.decode())
    print("Client stderr:", stderr.decode())

    # Save config for reference
    with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
        json.dump([client_config, server_config], fp)
    print('\n' + client_cmd + '\n')

    # Read the benchmark result JSON file that was created by benchmark_serving.py
    benchmark_result_path = f"{DIR}/{result_filename}"
    benchmark_data = {}
    if os.path.exists(benchmark_result_path):
        with open(benchmark_result_path, 'r') as fp:
            benchmark_data = json.load(fp)

    # Extract token usage metrics
    output_lens = benchmark_data.get('output_lens', [])
    total_output_tokens = sum(output_lens)
    avg_tokens_per_problem = total_output_tokens / len(output_lens) if output_lens else 0

    # Compute accuracy by comparing generated outputs to expected answers
    from verify_outputs import load_jsonl, answers_match

    generated_texts = benchmark_data.get('generated_texts', [])
    # Load ground truth answers from answers_file (test.jsonl)
    test_data = load_jsonl(answers_file)

    correct = 0
    total = min(len(generated_texts), len(test_data), num_prompts)

    for i in range(total):
        expected = test_data[i].get('answer', '')
        generated = generated_texts[i] if i < len(generated_texts) else ''
        is_match, _ = answers_match(expected, generated)
        if is_match:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    # Build result dictionary with relevant metrics
    result = {
        'num_prompts': num_prompts,
        'request_rate': request_rate,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'total_output_tokens': total_output_tokens,
        'avg_tokens_per_problem': avg_tokens_per_problem,
        'output_lens': output_lens,  # Per-problem token counts
        'budget_mode': 'budgeted' if budgets_file else 'unbounded',
    }

    if budgets_file:
        result['budgets_file'] = budgets_file

    return result

async def start_server(server_config):
    # Launch the server and wait for it to be ready.
    log_file_name = await asyncio.to_thread(run_server, server_config)
    print("wait_for_server_ready:", log_file_name)
    is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
    return is_ready

async def verify_latest_run(use_unique_id=True, budget_file=None):
    """
    Verify the latest benchmark run's outputs against ground truth.
    Returns tuple: (correct_count, total_count, report_path, overrun_stats)

    Args:
        use_unique_id: If True, include timestamp in report filename
        budget_file: Optional path to budget CSV file for budget-aware verification
    """
    answers_dir = Path(DIR) / 'answers'
    pattern = str(answers_dir / 'stdout_*.txt')
    # Exclude full_output files - they have different format that verify_outputs.py can't parse
    candidates = [f for f in sorted(glob.glob(pattern), key=os.path.getmtime)
                  if 'full_output' not in f]
    if not candidates:
        print(f"⚠️  No stdout files found in {answers_dir} to verify")
        return 0, 0, None, None

    latest_stdout = candidates[-1]
    timestamp = int(time.time())
    report_path = Path(DIR) / f'verify_report_{timestamp}.csv'

    print(f"\n{'='*60}")
    print(f"🔍 VERIFYING OUTPUTS")
    print(f"{'='*60}")
    print(f"Stdout file:  {latest_stdout}")
    print(f"Ground truth: {answers_file}")
    if budget_file:
        print(f"Budget file:  {budget_file}")
    print(f"Report:       {report_path}")
    print()

    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'verify_outputs.py'),
        '--stdout-file', latest_stdout,
        '--test-jsonl', str(Path(__file__).parent / answers_file),
        '--out', str(report_path)
    ]

    # Add budget file if provided
    if budget_file:
        cmd.extend(['--budget-file', str(budget_file)])

    ret = subprocess.run(cmd, capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"❌ Verification failed!")
        print(ret.stderr)
        return 0, 0, None, None

    # Parse verification output to get counts
    output = ret.stdout
    print(output)

    # Extract correct/total from output like "✅ Compared 31 items: 6 correct (19.4%)"
    match = re.search(r'Compared (\d+) items: (\d+) correct', output)
    if match:
        total = int(match.group(1))
        correct = int(match.group(2))
    else:
        total, correct = 0, 0

    # Compute budget overrun statistics if budget file was used
    overrun_stats = None
    if budget_file and report_path:
        from compute_budget_overrun_stats import compute_overrun_stats
        overrun_stats = compute_overrun_stats(str(report_path))

        # Print overrun statistics
        if overrun_stats['num_overruns'] > 0:
            print(f"\n{'='*60}")
            print(f"📊 BUDGET OVERRUN STATISTICS")
            print(f"{'='*60}")
            print(f"Total problems:          {overrun_stats['num_total']}")
            print(f"Budget overruns:         {overrun_stats['num_overruns']} ({overrun_stats['overrun_rate']:.1f}%)")
            print()
            print(f"Mean Absolute Error (MAE):       {overrun_stats['mae']:.2f} tokens")
            print(f"Mean Squared Error (MSE):        {overrun_stats['mse']:.2f}")
            print(f"Root Mean Squared Error (RMSE):  {overrun_stats['rmse']:.2f} tokens")
            print(f"Mean Percentage Overrun:         {overrun_stats['mean_pct_overrun']:.1f}%")
            print(f"{'='*60}\n")

    return correct, total, str(report_path), overrun_stats

async def start_exp(server_config, client_configs, args=None, budgets_file=None):
    print("Starting server configuration:", server_config)
    await start_server(server_config)

    results = []
    for client_config in client_configs:
        result = await run_client(client_config, server_config, args=args, budgets_file=budgets_file)
        results.append(result)

        # Print summary for this run
        print("\n" + "="*60)
        print(f"RESULTS SUMMARY")
        print("="*60)
        print(f"Accuracy:              {result['correct']}/{result['total']} ({result['accuracy']*100:.1f}%)")
        print(f"Total output tokens:   {result['total_output_tokens']}")
        print(f"Avg tokens/problem:    {result['avg_tokens_per_problem']:.1f}")
        print(f"Mode:                  {result['budget_mode']}")
        if budgets_file:
            print(f"Budget file:           {result.get('budgets_file', 'N/A')}")
        print("="*60 + "\n")

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

    # Return the last result for verification purposes
    return results[-1] if results else None

async def main(args):
    budgets_file_to_pass = None
    ran_experiment = False  # Track if we ran any experiments
    used_budget_file = None  # Track budget file used for verification

    # === DRY RUN ===
    if args.dry_run:
        # Stop any running server before starting experiments
        kill_server(server_configs[0]['host'])
        print("\n=== DRY RUN: Running benchmark without token budgets ===")
        tasks = [
            start_exp(server_config, client_configs, args=args, budgets_file=None)
            for server_config in server_configs
        ]
        await asyncio.gather(*tasks)
        ran_experiment = True
        print("Dry run complete. You can now compute token budgets with --compute-budgets.")

    # === CREATE BUDGETS FROM TOKENS FILE (NEW SIMPLIFIED METHOD) ===
    if args.create_budgets_from_tokens:
        if not args.tokens_file:
            raise SystemExit("Error: --tokens-file is required when using --create-budgets-from-tokens")

        tokens_file = Path(args.tokens_file)
        if not tokens_file.exists():
            raise SystemExit(f"Error: Tokens file not found: {tokens_file}")

        # Convert multiplier to percentile name (e.g., 0.5 -> 50)
        percentile_name = int(args.budget_multiplier * 100)
        budgets_csv = Path(DIR) / f'budgets_for_client_{percentile_name}.csv'

        print(f"\n=== CREATING BUDGETS FROM TOKENS FILE ===")
        print(f"Input file:   {tokens_file}")
        print(f"Multiplier:   {args.budget_multiplier} ({percentile_name}%)")
        print(f"Output file:  {budgets_csv}")

        with open(tokens_file, 'r', encoding='utf-8') as inf, \
             open(budgets_csv, 'w', encoding='utf-8', newline='') as outf:
            reader = csv.DictReader(inf)
            writer = csv.DictWriter(outf, fieldnames=['id', 'token_budget'])
            writer.writeheader()

            for row in reader:
                output_tokens = row.get('output_tokens')
                problem_id = row.get('id')
                if output_tokens is None or output_tokens == '' or not problem_id:
                    continue

                # Multiply by budget multiplier and round to integer
                token_budget = int(float(output_tokens) * args.budget_multiplier)

                writer.writerow({
                    'id': problem_id,
                    'token_budget': token_budget
                })

        print(f"✅ Budgets file created: {budgets_csv}")
        print(f"\nTo run with these budgets:")
        print(f"  python run.py --run-with-budgets --budgets-percentile {percentile_name}")
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
            writer = csv.DictWriter(outf, fieldnames=['id', 'token_budget'])
            writer.writeheader()
            for row in reader:
                tok = row.get(percentile_col)
                problem_id = row.get('id')
                if tok is None or tok == '' or not problem_id:
                    continue
                writer.writerow({
                    'id': problem_id,
                    'token_budget': tok
                })
        print(f"Budgets file written to {budgets_csv}")
        return

    # === RUN WITH BUDGETS ===
    if args.run_with_budgets or args.budgets_file:
        # Stop any running server before starting experiments
        kill_server(server_configs[0]['host'])
        budgets_file_to_pass = args.budgets_file
        if not budgets_file_to_pass:
            budgets_file_to_pass = str(Path(DIR) / f'budgets_for_client_{args.budgets_percentile}.csv')
        print(f"\n=== RUN WITH BUDGETS: Using {budgets_file_to_pass} ===")
        tasks = [
            start_exp(server_config, client_configs, args=args, budgets_file=budgets_file_to_pass)
            for server_config in server_configs
        ]
        await asyncio.gather(*tasks)
        ran_experiment = True
        used_budget_file = budgets_file_to_pass  # Track for verification

    # === ALWAYS VERIFY OUTPUTS AFTER ANY EXPERIMENT ===
    if ran_experiment:
        correct, total, report_path, overrun_stats = await verify_latest_run(budget_file=used_budget_file)

        # Print final verification summary
        if total > 0:
            accuracy_pct = (correct / total) * 100
            print(f"\n{'='*60}")
            print(f"📊 FINAL VERIFICATION RESULTS")
            print(f"{'='*60}")
            print(f"✅ Correct:   {correct}/{total} ({accuracy_pct:.1f}%)")
            print(f"❌ Incorrect: {total - correct}/{total} ({100 - accuracy_pct:.1f}%)")
            if report_path:
                print(f"📄 Report:    {report_path}")
            print(f"{'='*60}\n")

    # === MANUAL VERIFY (if --verify-after flag is used without running experiments) ===
    elif args.verify_after:
        correct, total, report_path, overrun_stats = await verify_latest_run()
        if total > 0:
            accuracy_pct = (correct / total) * 100
            print(f"\n{'='*60}")
            print(f"📊 VERIFICATION RESULTS")
            print(f"{'='*60}")
            print(f"✅ Correct:   {correct}/{total} ({accuracy_pct:.1f}%)")
            print(f"❌ Incorrect: {total - correct}/{total} ({100 - accuracy_pct:.1f}%)")
            if report_path:
                print(f"📄 Report:    {report_path}")
            print(f"{'='*60}\n")
        
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cache_dir = "/scratch/gpfs/WLLOYD/al2926/hf_models/daman1209arora"
    # daman1209arora/alpha_0.2_DeepSeek-R1-Distill-Qwen-7B
    # daman1209arora/alpha_0.4_DeepSeek-R1-Distill-Qwen-7B
    tokenizer = AutoTokenizer.from_pretrained(
        "daman1209arora/alpha_0.4_DeepSeek-R1-Distill-Qwen-7B",
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "daman1209arora/alpha_0.4_DeepSeek-R1-Distill-Qwen-7B",
        cache_dir=cache_dir,
        trust_remote_code=True
    )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run vllm cache benchmark in dry-run, budgeted, or verify mode"
#     )
#     # New parameters for dataset and output directory
#     parser.add_argument("--dataset", default="AIME25_benchmark.json",
#                         help="Path to dataset file (e.g., AIME25_benchmark.json)")
#     parser.add_argument("--answers", default=None,
#                         help="Path to answers file (e.g., AIME25.jsonl). If not provided, inferred from dataset name")
#     parser.add_argument("--output-dir", default=None,
#                         help="Output directory for results. If not provided, uses default structure")
#     parser.add_argument("--num-prompts", type=int, default=30,
#                         help="Number of prompts to run (default: 30)")

#     # Existing parameters
#     parser.add_argument("--dry-run", action="store_true",
#                         help="Run without token budgets (collects raw token usage)")
#     parser.add_argument("--compute-budgets", action="store_true",
#                         help="Compute token budgets from the last dry run's stdout")
#     parser.add_argument("--create-budgets-from-tokens", action="store_true",
#                         help="Create budget file by multiplying output_tokens from a CSV file")
#     parser.add_argument("--tokens-file", default="",
#                         help="Path to CSV file with output_tokens column (for --create-budgets-from-tokens)")
#     parser.add_argument("--budget-multiplier", type=float, default=0.5,
#                         help="Multiplier for output_tokens (e.g., 0.5 for 50%%, 0.75 for 75%%). Default: 0.5")
#     parser.add_argument("--run-with-budgets", action="store_true",
#                         help="Run again using computed or given token budgets")
#     parser.add_argument("--budgets-file", default="",
#                         help="Path to a precomputed budgets CSV to use in budgeted run")
#     parser.add_argument("--budgets-percentile", type=int, default=50,
#                         help="Percentile column to use when converting token budgets (e.g. 50 for budget_50)")
#     parser.add_argument("--verify-after", action="store_true",
#                         help="Verify latest outputs using verify_outputs.py")

#     args = parser.parse_args()

#     # Set global variables based on arguments
#     dataset_file = args.dataset

#     # Infer answers file if not provided
#     if args.answers:
#         answers_file = args.answers
#     else:
#         # Infer from dataset name (e.g., AIME25_benchmark.json -> AIME25.jsonl)
#         dataset_name = Path(dataset_file).stem
#         if '_benchmark' in dataset_name:
#             dataset_name = dataset_name.replace('_benchmark', '')
#         answers_file = f"{dataset_name}.jsonl"

#     # Set output directory
#     if args.output_dir:
#         DIR = args.output_dir
#     else:
#         # Use default: results/{model_id}/{dataset_name}/unbounded
#         from constants import get_results_dir
#         dataset_name = Path(dataset_file).stem.replace('_benchmark', '')
#         DIR = get_results_dir(dataset_name, "unbounded")

#     # Create output directories
#     Path(DIR).mkdir(parents=True, exist_ok=True)
#     for subdir in ['answers', 'configs', 'metrics']:
#         (Path(DIR) / subdir).mkdir(exist_ok=True)

#     # Update client configs with num_prompts
#     client_configs[0]['num_prompts'] = args.num_prompts

#     asyncio.run(main(args))
    
