#!/bin/bash
set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$ROOT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate zea
cd "$SCRIPT_DIR"
export KERAS_BACKEND=jax
export ZEA_CACHE_DIR="$SCRIPT_DIR/cache"

mkdir -p outputs

echo "=== Multi-Rate Sweep ==="
echo "Acceleration rates: 2 3 4 6 10"

# Ensure GT volume exists
if [ ! -f outputs/pseudo_volume.npy ]; then
    echo ""
    echo "=== Preparing pseudo-volume (GT) ==="
    python 02_prepare_pseudo_volume.py
fi

# Sweep over acceleration rates
for r in 2 3 4 6 10; do
    echo ""
    echo "=== r=$r: Reconstructing ==="
    python 03_reconstruct_volume.py --accel-rate "$r" --output-dir "outputs/r$r"

    echo ""
    echo "=== r=$r: Evaluating ==="
    python 05_evaluate.py --accel-rate "$r" --output-dir "outputs/r$r" --json-output "outputs/r$r/metrics.json"
done

# Summary
echo ""
echo "=== Generating sweep summary ==="
python 09_sweep_summary.py

echo ""
echo "=== Sweep complete ==="
echo "Results: outputs/09_sweep_table.txt"
echo "Figure:  outputs/09_sweep_summary.png"
echo "CSV:     outputs/09_sweep_summary.csv"
