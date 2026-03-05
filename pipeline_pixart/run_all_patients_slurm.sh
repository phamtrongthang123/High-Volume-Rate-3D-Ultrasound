#!/bin/bash
#SBATCH --job-name=pixart-cetus
#SBATCH --output=/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound/pipeline_pixart/slurm_%A_%a.out
#SBATCH --partition=vgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=1-45

# Run DPS reconstruction for all 45 CETUS patients (ED phase).
# Submit with: sbatch pipeline_pixart/run_all_patients_slurm.sh

set -eo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array task: $SLURM_ARRAY_TASK_ID"

PROJECT_ROOT="/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound"
CONDA_ENV="/home/tp030/.conda/envs/us3d"
PYTHON="$CONDA_ENV/bin/python"

export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# Map array task ID (1-45) to patient ID
PATIENT_NUM=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
PATIENT_ID="patient${PATIENT_NUM}"
PHASE="ED"
OUTPUT_DIR="$PROJECT_ROOT/outputs/${PATIENT_ID}_${PHASE}"

echo "Patient: $PATIENT_ID, Phase: $PHASE"
echo "Output dir: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT/pipeline_pixart"

echo "=== Step 1: Prepare volume ==="
$PYTHON -u prepare_volume.py \
    --patient-id "$PATIENT_ID" \
    --phase "$PHASE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 2: Reconstruct volume ==="
$PYTHON -u 03_reconstruct_volume.py \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 3: Evaluate ==="
$PYTHON -u 05_evaluate.py \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Done: $PATIENT_ID $PHASE ==="
echo "Metrics:"
cat "$OUTPUT_DIR/pixart_metrics.json" 2>/dev/null || echo "  (metrics not found)"
echo "Job finished at: $(date)"
