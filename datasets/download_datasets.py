# from datasets import load_dataset
# import os

# # Directory to save datasets
# OUTPUT_DIR = "/home/al2926/datasets"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Public datasets with test splits
# datasets = {
#     "gsm8k_main": ("openai/gsm8k", "main"),
#     "mathbench_arithmetic": ("mathbench/mathbench-arithmetic", None),
#     "mathbench_middle": ("mathbench/mathbench-middle", None),
#     "mathbench_high": ("mathbench/mathbench-high", None),
#     "mathbench_college": ("mathbench/mathbench-college", None)
# }

# for name, (repo, config) in datasets.items():
#     print(f"Downloading {name} test set...")
#     if config:
#         ds = load_dataset(repo, config, split="test")
#     else:
#         ds = load_dataset(repo, split="test")
    
#     out_file = os.path.join(OUTPUT_DIR, f"{name}_test.jsonl")
#     ds.to_json(out_file, orient="records", lines=True)
#     print(f"Saved to {out_file}")

# print("\nAll public datasets downloaded successfully!")
import pandas as pd

# Load the parquet file
df = pd.read_parquet("/home/al2926/vllm_reason_bench/datasets/train-00000-of-00001.parquet")

# Convert to JSON
df.to_json("gmsk_test.json", orient="records", lines=True)