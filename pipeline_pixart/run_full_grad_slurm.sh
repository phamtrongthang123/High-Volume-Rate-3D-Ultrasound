#!/bin/bash
#SBATCH --job-name=pixart-fullgrad
#SBATCH --output=/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound/pipeline_pixart/slurm_%j.out
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Run DPS reconstruction with full gradient through transformer (true DPS).
# Requires A100 GPU for sufficient memory (gradient through 600M transformer + VAE).
# Submit with: sbatch pipeline_pixart/run_full_grad_slurm.sh

set -eo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

PROJECT_ROOT="/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound"
CONDA_ENV="/home/tp030/.conda/envs/us3d"
PYTHON="$CONDA_ENV/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/outputs/full_grad_test"

export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT/pipeline_pixart"

# Use patient01 ED as test case (copy existing GT volume)
if [ ! -f "$OUTPUT_DIR/pseudo_volume.npy" ]; then
    echo "=== Step 1: Prepare volume ==="
    $PYTHON -u prepare_volume.py \
        --patient-id patient01 \
        --phase ED \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=== Step 2: Reconstruct volume (full gradient DPS) ==="
$PYTHON -u 03_reconstruct_volume.py \
    --output-dir "$OUTPUT_DIR" \
    --full-grad

echo ""
echo "=== Step 3: Evaluate ==="
$PYTHON -u 05_evaluate.py \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Done ==="
echo "Metrics:"
cat "$OUTPUT_DIR/pixart_metrics.json" 2>/dev/null || echo "  (metrics not found)"
echo "Job finished at: $(date)"
