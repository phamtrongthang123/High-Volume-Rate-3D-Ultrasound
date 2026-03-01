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

echo "=== Environment ==="
python --version
python -c "import jax; print(f'JAX: {jax.__version__}')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
python -c "import keras; print(f'Keras: {keras.__version__}, backend: {keras.backend.backend()}')"

# Step 1: Verify prior
echo ""
echo "=== Step 1: Verify prior ==="
python 01_verify_prior.py

# Step 2: Prepare pseudo-volume
echo ""
echo "=== Step 2: Prepare pseudo-volume ==="
python 02_prepare_pseudo_volume.py

# Step 3: Reconstruct volume (main algorithm)
echo ""
echo "=== Step 3: Reconstruct volume ==="
python 03_reconstruct_volume.py

# Step 4: SeqDiff temporal demo
echo ""
echo "=== Step 4: SeqDiff temporal acceleration ==="
python 04_seqdiff_temporal.py

# Step 5: Evaluate
echo ""
echo "=== Step 5: Evaluate ==="
python 05_evaluate.py

echo ""
echo "=== All steps completed ==="
echo "Outputs saved to: $SCRIPT_DIR/outputs/"
