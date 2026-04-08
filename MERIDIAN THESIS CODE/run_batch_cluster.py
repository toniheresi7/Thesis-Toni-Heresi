# run_batch_cluster.py
# Cluster-specific entry point for run_batch.py.
# Forces FinBERT to use CUDA (RTX 6000 Ada on haskell) instead of MPS/CPU.
# Usage: python run_batch_cluster.py [same args as run_batch.py]

import sys
import os
sys.path.insert(0, "src")

# Patch device to CUDA before run_batch imports torch
import torch
import run_batch as _rb

_rb._DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
_rb._BATCH_SIZE = 256 if _rb._DEVICE == "cuda" else 8

print(f"[cluster] FinBERT device: {_rb._DEVICE}  batch_size: {_rb._BATCH_SIZE}")
if _rb._DEVICE == "cuda":
    print(f"[cluster] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[cluster] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Re-initialise the HuggingFace pipeline with the correct device
from transformers import pipeline as _hf_pipeline
_rb._PIPE = _hf_pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=True,
    truncation=True,
    max_length=512,
    device=_rb._DEVICE,
    batch_size=_rb._BATCH_SIZE,
)

if __name__ == "__main__":
    _rb.main()
