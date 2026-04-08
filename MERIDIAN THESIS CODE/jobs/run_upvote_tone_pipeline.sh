#!/bin/bash
#SBATCH --job-name=upvote-tone
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log

set -e

source $HOME/miniforge3/bin/activate myproject
cd ~/social_signal_platform

echo "=========================================="
echo "  Upvote-Weighted Tone Pipeline"
echo "  Started: $(date)"
echo "=========================================="

# Step 1: Run pipeline with upvote-weighted tone aggregation
echo ""
echo "========== STEP 1: Run upvote-weighted tone pipeline =========="
PYTHONPATH=src python src/run_pipeline_upvote_tone.py

# Step 2: Evaluate
echo ""
echo "========== STEP 2: Evaluate =========="
mkdir -p results
PYTHONPATH=src python evaluation/evaluate_custom.py --data-dir data/processed_upvote_tone 2>&1 | tee results/upvote_tone_evaluation.txt

echo ""
echo "=========================================="
echo "  Completed: $(date)"
echo "=========================================="
