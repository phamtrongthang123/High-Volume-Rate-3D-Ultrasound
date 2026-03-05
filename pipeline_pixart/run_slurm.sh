#!/bin/bash
#SBATCH --job-name=pixart-recon
#SBATCH --output=/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound/pipeline_pixart/slurm_%j.out
#SBATCH --partition=vgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

set -eo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

PROJECT_ROOT="/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound"
CONDA_ENV="/home/tp030/.conda/envs/us3d"

export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT/pipeline_pixart"

echo "=== Step 1: Reconstruct volume with PixArt-α ==="
"$CONDA_ENV/bin/python" -u 03_reconstruct_volume.py

echo ""
echo "=== Step 2: Evaluate ==="
"$CONDA_ENV/bin/python" -u 05_evaluate.py

echo ""
echo "=== Pipeline complete ==="
echo "Outputs:"
ls -la "$PROJECT_ROOT/outputs/reconstructed_volume_pixart.npy" 2>/dev/null || echo "  (not found)"
ls -la "$PROJECT_ROOT/outputs/pixart_metrics.json" 2>/dev/null || echo "  (not found)"
ls -la "$PROJECT_ROOT/outputs/pixart_evaluation.png" 2>/dev/null || echo "  (not found)"

echo "Job finished at: $(date)"
