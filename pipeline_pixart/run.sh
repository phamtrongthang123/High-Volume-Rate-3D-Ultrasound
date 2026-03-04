#!/bin/bash
# run.sh — Full PixArt reconstruction pipeline
# Usage: bash pipeline_pixart/run.sh
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

echo "=== Step 1: Reconstruct volume with PixArt-α ==="
python 03_reconstruct_volume.py

echo ""
echo "=== Step 2: Evaluate ==="
python 05_evaluate.py

echo ""
echo "=== Pipeline complete ==="
echo "Outputs:"
echo "  outputs/reconstructed_volume_pixart.npy"
echo "  outputs/pixart_metrics.json"
echo "  outputs/pixart_evaluation.png"
echo "  outputs/pixart_elevation_profile.png"
