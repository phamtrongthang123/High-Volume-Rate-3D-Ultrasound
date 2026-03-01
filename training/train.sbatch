#!/bin/bash
#SBATCH --job-name=pixart-lora
#SBATCH --output=training/slurm_logs/slurm_%j.out
#SBATCH --partition=agpu72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint=1a100
#SBATCH --time=24:00:00

set -eo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

PROJECT_ROOT="/scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"
CONDA_ENV_NAME="zea"

apptainer exec --nv --writable-tmpfs \
    --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
    --bind /home/tp030:/home/tp030 \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
conda activate '${CONDA_ENV_NAME}'

echo '=== Installing missing Python packages ==='
pip install --quiet datasets diffusers peft transformers torchvision sentencepiece tiktoken

cd '${PROJECT_ROOT}'
bash training/run_train.sh
"
