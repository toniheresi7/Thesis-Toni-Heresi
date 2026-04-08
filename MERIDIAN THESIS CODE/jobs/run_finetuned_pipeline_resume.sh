#!/bin/bash
#SBATCH --job-name=finetuned-resume
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
echo "  Finetuned FinBERT Pipeline (Resume)"
echo "  Started: $(date)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi

# Step 2 already completed - pseudo_labels.csv exists
echo "Step 2 already complete: $(wc -l < data/pseudo_labelled/pseudo_labels.csv) lines in pseudo_labels.csv"

# Step 3: Fine-tune FinBERT
echo ""
echo "========== STEP 3: Fine-tune FinBERT =========="
PYTHONPATH=src python src/finetune_finbert.py

# Step 6: Run pipeline with finetuned model
echo ""
echo "========== STEP 6: Run finetuned pipeline =========="
PYTHONPATH=src python src/run_pipeline_finetuned.py

# Step 7: Evaluate
echo ""
echo "========== STEP 7: Evaluate =========="
mkdir -p results
PYTHONPATH=src python evaluation/evaluate_custom.py --data-dir data/processed_finetuned 2>&1 | tee results/finetuned_evaluation.txt

echo ""
echo "=========================================="
echo "  Completed: $(date)"
echo "=========================================="
