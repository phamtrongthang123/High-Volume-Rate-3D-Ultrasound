#!/bin/bash
#SBATCH --job-name=pixart-lora
#SBATCH --output=training/slurm_%j.out
#SBATCH --partition=agpu72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint=1a100
#SBATCH --time=24:00:00

# Load conda
source ~/.bashrc
conda activate zea

cd /scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound

bash training/run_train.sh
