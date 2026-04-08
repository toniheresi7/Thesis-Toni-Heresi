#!/bin/bash
#SBATCH --job-name=meridian-finbert
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log

# ── Environment ───────────────────────────────────────────────────────────────
source $HOME/miniforge3/bin/activate meridian

# ── Working directory ─────────────────────────────────────────────────────────
cd $HOME/social_signal_platform

echo "======================================================="
echo "  Meridian — Social Signal Batch Pipeline"
echo "  Job:  $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "======================================================="

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run full pipeline: process all days + train model
# --skip-existing  → safe to re-run; won't reprocess days already in data/processed/
python run_batch_cluster.py --skip-existing

echo ""
echo "[job] Pipeline complete at $(date)"
echo "[job] Processed CSVs in data/processed/"
echo "[job] Model saved to data/model.pkl"
