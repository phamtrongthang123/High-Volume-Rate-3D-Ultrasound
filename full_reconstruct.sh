#!/bin/bash
set -euo pipefail

echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "Running: High-Rate 3D Ultrasound Reconstruction"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
bash "$SCRIPT_DIR/run_reconstruct.sh"

echo "Finished at: $(date)"
