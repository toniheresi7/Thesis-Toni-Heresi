#!/bin/bash
#SBATCH --job-name=finetuned-pipeline
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
echo "  Finetuned FinBERT Pipeline"
echo "  Started: $(date)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 2: Generate pseudo-labels
echo ""
echo "========== STEP 2: Generate pseudo-labels =========="
PYTHONPATH=src python src/generate_pseudo_labels.py

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
