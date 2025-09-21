import os
import psutil
import socket
import subprocess
import math
import time
from typing import Optional
import requests
from tqdm import tqdm
def restart_ray():
    """Stop and start Ray on the head node."""
    subprocess.run("ray stop --force", shell=True, check=True)
    subprocess.run(f"VLLM_PP_LAYER_PARTITION={os.environ['VLLM_PP_LAYER_PARTITION']} RAY_DEDUP_LOGS=0 ray start --head", shell=True, check=True)


def restart_ray_remote(hostname):
    """Stop and start Ray on a remote node."""
    head_node_ip = socket.gethostbyname(socket.gethostname())
    stop_command = f"ssh {hostname} 'ray stop --force'"
    start_command = (
        f"ssh {hostname} 'VLLM_PP_LAYER_PARTITION={os.environ['VLLM_PP_LAYER_PARTITION']} "
        f"RAY_DEDUP_LOGS=0 ray start --address=\"{head_node_ip}:6379\"'"
    )
    subprocess.run(stop_command, shell=True, check=True)
    subprocess.run(start_command, shell=True, check=True)


def set_pp_layers(pp, num_layers, original_pp_partition):
    """Set pipeline-parallel layers for the model."""
    if pp <= 2 or original_pp_partition:
        os.environ["VLLM_PP_LAYER_PARTITION"] = ""
        return
    else:
        layer_per_stage = math.ceil(num_layers / pp)
        first_last = num_layers - (pp - 2) * layer_per_stage
        first = first_last // 2
        last = first_last - first
        layer_partition = [first] + [layer_per_stage] * (pp - 2) + [last]
        os.environ["VLLM_PP_LAYER_PARTITION"] = ','.join(map(str, layer_partition))
    restart_ray()
    restart_ray_remote("node2")


def is_port_in_use(port, host=None):
    """
    Check if a port is in use locally or on a remote node.
    :param port: Port number to check.
    :param node: Node to check (None for local).
    :return: True if port is in use, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ssh", host, f"lsof -i:{port}"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.returncode == 0  # lsof returns 0 if the port is in use
    except subprocess.CalledProcessError as e:
        print(f"Failed to check port on {host}: {e}")
        return False

def kill_server(host):
    """Kill any running server process."""
    try:
        subprocess.run(["ssh", host, "pkill -f python3"])
        # subprocess.run(["ssh", host, "pkill -f /opt/conda/bin/python3.12"])
        # subprocess.run(["ssh", host, "pkill -f nsys"])
        print("Killed any running 'vllm serve' process.")
    except subprocess.CalledProcessError as e:
        print("No 'vllm serve' processes were running:", e)

    time.sleep(5)


# def check_if_found():
#     dataset_path = '/scratch/gpfs/al2926/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json'

#     if not os.path.isfile(dataset_path) and dataset_path == "":
#         print("NOT FOUND")
#         dataset_path = download_and_cache_file(SHAREGPT_URL)
#     else:
#         print("FOUND")


def download_and_cache_file(
    url: str = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", 
    filename: Optional[str] = None
):
    """Download and cache a file from a URL with progress bar."""
    # Set default cache directory to HuggingFace cache
    cache_dir = "/scratch/gpfs/al2926/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)

    # If no filename provided, use the last part of the URL
    if filename is None:
        filename = os.path.join(cache_dir, url.split("/")[-1])

    print(f"Downloading from {url} to {filename}")

    try:
        # Stream the response to show the progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Total size of the file in bytes
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192  # Download in chunks of 8KB for better performance

        # Use tqdm to display the progress bar
        with open(filename, "wb") as f, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)

        print(f"Download complete. File saved to {filename}")
        return filename

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

import re

def decode_tree(tree_output, tokenizer):
    """
    Given a tree_output string and a tokenizer, print the tree with decoded text,
    and remove timestamp prefixes.
    """
    # Regex to match lines with token ids
    token_line_re = re.compile(r"(\[len=\d+\]) \[([0-9, ]+)\](?:\.\.\.)? hits=\d+ ref=\d+")
    # Regex to match and remove timestamps at the start of each line
    timestamp_re = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [^\]]+\] ?")

    for line in tree_output.strip().splitlines():
        # Remove timestamp prefix if present
        line_wo_ts = timestamp_re.sub("", line)
        m = token_line_re.search(line_wo_ts)
        if m:
            token_ids_str = m.group(2)
            token_ids = [int(x) for x in token_ids_str.split(",")]
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            # Replace the token id list with the decoded text
            new_line = token_line_re.sub(r"\1 \"" + decoded + "\"", line_wo_ts)
            print(new_line)
        else:
            print(line_wo_ts)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    # Load the tokenizer for Qwen2.5-0.5B
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    tree_output = """
[2025-04-26 01:06:59 TP0] [len=0] [] hits=12 ref=1
[2025-04-26 01:06:59 TP0] └── [len=30] [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=21 ref=1
[2025-04-26 01:06:59 TP0]     └── [len=2] [9707, 0] hits=19 ref=1
[2025-04-26 01:06:59 TP0]         └── [len=12] [2585, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=17 ref=1
[2025-04-26 01:06:59 TP0]             └── [len=2] [9707, 0] hits=15 ref=1
[2025-04-26 01:06:59 TP0]                 └── [len=12] [2585, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=13 ref=1
[2025-04-26 01:06:59 TP0]                     └── [len=2] [9707, 0] hits=11 ref=1
[2025-04-26 01:06:59 TP0]                         └── [len=12] [2585, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=9 ref=1
[2025-04-26 01:06:59 TP0]                             └── [len=2] [9707, 0] hits=7 ref=1
[2025-04-26 01:06:59 TP0]                                 └── [len=12] [2585, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=5 ref=1
[2025-04-26 01:06:59 TP0]                                     └── [len=2] [9707, 0] hits=3 ref=1
[2025-04-26 01:06:59 TP0]                                         └── [len=12] [2585, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198]... hits=1 ref=1
    """


    # this was never inserted into the cache [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 6335, 16261, 525, 12482, 5135, 30, 151645, 198, 151644, 77091, 198, 40, 2776, 14589, 11, 714, 358, 2776, 1207, 16948, 11, 537, 264, 4462, 315, 54364, 14817, 13, 358, 1513, 944, 614, 1931, 7246, 1995, 476, 2615, 311, 6335, 16261, 13, 1416, 498, 2299, 3330, 369, 1995, 911, 14487, 6335, 16261, 11, 358, 4172, 6934, 13295, 279, 3946, 3910, 476, 3590, 3687, 6816, 315, 279, 18890, 498, 2299, 8014, 304, 11, 438, 807, 3545, 1736, 8837, 323, 44876, 389, 862, 15409, 13]

    token_ids_list = [
        [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 6335, 16261, 525, 12482, 5135, 30, 151645, 198, 151644, 77091, 198],
        [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 6335, 16261, 525, 12482, 5135, 30, 151645, 198, 151644, 77091, 198, 40, 2776, 14589, 11, 714, 358, 2776, 1207, 16948, 11, 537, 264, 4462, 315, 54364, 14817, 13, 358, 1513, 944, 614, 1931, 7246, 1995, 476, 2615, 311, 6335, 16261, 13, 1416, 498, 2299, 3330, 369, 1995, 911, 14487, 6335, 16261, 11, 358, 4172, 6934, 13295, 279, 3946, 3910, 476, 3590, 3687, 6816, 315, 279, 18890, 498, 2299, 8014, 304, 11, 438, 807, 3545, 1736, 8837, 323, 44876, 389, 862, 1736, 8837, 323, 44876, 389, 862, 19511, 15409, 13],
        [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 6335, 16261, 525, 12482, 5135, 30, 151645, 198, 151644, 77091, 198, 40, 2776, 14589, 11, 714, 358, 2776, 1207, 16948, 11, 537, 264, 4462, 315, 54364, 14817, 13, 358, 1513, 944, 614, 1931, 7246, 1995, 476, 2615, 311, 6335, 16261, 13, 1416, 498, 2299, 3330, 369, 1995, 911, 14487, 6335, 16261, 11, 358, 4172, 6934, 13295, 279, 3946, 3910, 476, 3590, 3687, 6816, 315, 279, 18890, 498, 2299, 8014, 304, 11, 438, 807, 3545, 3410, 8837, 323, 44876, 13], 
        [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 6335, 16261, 525, 12482, 5135, 30, 151645, 198, 151644, 77091, 198, 40, 2776, 14589, 11, 714, 358, 2776, 1207, 16948, 11, 537, 264, 4462, 315, 54364, 14817, 13, 358, 1513, 944, 614, 1931, 7246, 1995, 476, 2615, 311, 6335, 16261, 13, 1416, 498, 2299, 3330, 369, 1995, 911, 14487, 6335, 16261, 11, 358, 4172, 6934, 13295, 279, 3946, 3910, 476, 3590, 3687, 6816, 315, 279, 18890, 498, 2299, 8014, 304, 11, 438, 807, 3545, 1736, 8837, 323, 44876, 389, 862, 15409, 13],
    ]


    print(len(token_ids_list[0]))

    for token_ids in token_ids_list:
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(len(token_ids), "Decoded text:", decoded_text)
        print(
            "\n\n\n"
        )
    # decode_tree(tree_output, tokenizer)