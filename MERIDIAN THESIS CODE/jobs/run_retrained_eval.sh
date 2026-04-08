#!/bin/bash
# Run retrained evaluations on the login node (needs internet for yfinance).
# Usage: bash jobs/run_retrained_eval.sh

set -e

source $HOME/miniforge3/bin/activate myproject
cd ~/social_signal_platform

echo "=========================================="
echo "  Retrained Model Evaluations"
echo "  Started: $(date)"
echo "=========================================="

mkdir -p results

echo ""
echo "========== StockTwits RoBERTa (retrained) =========="
PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed_stocktwits \
    --output results/stocktwits_retrained_evaluation.txt

echo ""
echo "========== Fine-tuned FinBERT (retrained) =========="
PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed_finetuned \
    --output results/finetuned_retrained_evaluation.txt

echo ""
echo "=========================================="
echo "  Completed: $(date)"
echo "=========================================="
