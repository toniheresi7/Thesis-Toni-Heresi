#!/bin/bash
#SBATCH --job-name=stocktwits
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log

set -e

source $HOME/miniforge3/bin/activate myproject
cd ~/social_signal_platform

echo "=========================================="
echo "  StockTwits RoBERTa Pipeline (Session 1)"
echo "  Started: $(date)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi

# Step 1: Run pipeline with StockTwits RoBERTa model
echo ""
echo "========== STEP 1: Run StockTwits pipeline =========="
PYTHONPATH=src python src/run_pipeline_stocktwits.py

# Step 2: Evaluate (on login node if yfinance needs internet)
echo ""
echo "========== STEP 2: Evaluate =========="
mkdir -p results
PYTHONPATH=src python evaluation/evaluate_custom.py --data-dir data/processed_stocktwits 2>&1 | tee results/stocktwits_evaluation.txt

echo ""
echo "=========================================="
echo "  Completed: $(date)"
echo "=========================================="
