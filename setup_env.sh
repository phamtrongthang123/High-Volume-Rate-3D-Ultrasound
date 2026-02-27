#!/bin/bash
set -euo pipefail

echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "Running: High-Rate 3D Environment Setup"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
bash "$SCRIPT_DIR/run_setup_env.sh"

echo "Finished at: $(date)"
