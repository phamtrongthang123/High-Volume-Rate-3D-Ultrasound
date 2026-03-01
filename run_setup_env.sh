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

echo "=== Setting up environment ==="

# Install/upgrade zea
pip install --upgrade zea

# Verify JAX GPU
echo ""
echo "=== Verifying JAX GPU ==="
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
assert len(gpu_devices) > 0, 'No GPU detected! JAX GPU required.'
print(f'GPU devices: {gpu_devices}')
"

# Verify Keras backend
echo ""
echo "=== Verifying Keras backend ==="
python -c "
import keras
print(f'Keras version: {keras.__version__}')
print(f'Keras backend: {keras.backend.backend()}')
"

# Verify ZEA model preset loads
echo ""
echo "=== Verifying ZEA model ==="
python -c "
from zea.models.diffusion import DiffusionModel
model = DiffusionModel.from_preset('diffusion-echonet-dynamic')
print(f'Model loaded: {model}')
print(f'Input shape: {model.input_shape}')
print('ZEA diffusion model verified successfully.')
"

# Download data
echo ""
echo "=== Downloading CAMUS dataset ==="
python 00_download_data.py

echo ""
echo "=== Environment setup complete ==="
