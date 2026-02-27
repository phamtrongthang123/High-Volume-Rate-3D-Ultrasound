"""
00_download_data.py — Download CAMUS dataset before running on HPC.

Downloads the CAMUS sample dataset from HuggingFace via ZEA's Dataset API.
Run this on a node with internet before submitting the reconstruction job.
"""

import env_setup  # noqa: F401 — must be first

from zea.data import Dataset

print("Downloading CAMUS sample dataset from HuggingFace...")
print("(timeout set to 300s for slow networks)")
dataset = Dataset("hf://zeahub/camus-sample/val", key="image")

# Iterate to trigger full download and cache
count = 0
for i in range(len(dataset)):
    _ = dataset[i]
    count += 1

dataset.close()
print(f"Downloaded {count} samples.")
print("Dataset cached locally. Ready for offline use.")
