#!/bin/bash
# run_train.sh — Full PixArt-α LoRA training pipeline for CETUS B-planes.
#
# Usage:
#   bash training/run_train.sh           # Full pipeline
#   bash training/run_train.sh --skip-dataset  # Skip dataset prep
#   bash training/run_train.sh --skip-vae      # Skip VAE decoder fine-tuning
#   bash training/run_train.sh --sample-only   # Only generate samples
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SKIP_DATASET=false
SKIP_VAE=false
SAMPLE_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-dataset) SKIP_DATASET=true ;;
        --skip-vae) SKIP_VAE=true ;;
        --sample-only) SAMPLE_ONLY=true ;;
    esac
done

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate zea 2>/dev/null || echo "Warning: 'zea' conda env not found, using current env"
fi

cd "$PROJECT_DIR"

# Step 1: Prepare dataset
if [ "$SKIP_DATASET" = false ] && [ "$SAMPLE_ONLY" = false ]; then
    echo "=== Step 1: Preparing CETUS B-plane dataset ==="
    python training/prepare_dataset.py
    echo ""
fi

# Step 1.5: Fine-tune VAE decoder
if [ "$SAMPLE_ONLY" = false ] && [ "$SKIP_VAE" = false ]; then
    echo "=== Step 1.5: Fine-tuning VAE decoder on ultrasound B-planes ==="
    accelerate launch --config_file training/accelerate_config.yaml \
        training/train_vae_decoder.py \
        --pretrained_model_name_or_path PixArt-alpha/PixArt-XL-2-512x512 \
        --train_data_dir training/dataset/train \
        --val_data_dir training/dataset/val \
        --output_dir training/checkpoints/vae_decoder_finetuned \
        --num_train_epochs 20 \
        --train_batch_size 8 \
        --learning_rate 1e-5 \
        --lr_scheduler cosine \
        --mixed_precision bf16 \
        --validation_epochs 5 \
        --report_to wandb \
        --seed 42
    echo ""
fi

# Step 2: Train LoRA
if [ "$SAMPLE_ONLY" = false ]; then
    echo "=== Step 2: Training PixArt-α LoRA ==="
    accelerate launch --config_file training/accelerate_config.yaml \
        training/train_pixart_lora.py \
        --pretrained_model_name_or_path PixArt-alpha/PixArt-XL-2-512x512 \
        --dataset_name training/dataset/train \
        --caption_column text \
        --resolution 512 \
        --train_batch_size 4 \
        --num_train_epochs 50 \
        --rank 8 \
        --learning_rate 1e-4 \
        --lr_scheduler cosine \
        --lr_warmup_steps 100 \
        --checkpointing_epochs 10 \
        --checkpoints_total_limit 5 \
        --validation_prompt "a cardiac ultrasound b-plane image" \
        --validation_epochs 5 \
        --mixed_precision bf16 \
        --output_dir training/checkpoints/cetus_pixart_lora \
        --report_to wandb \
        --val_data_dir training/dataset/val \
        --seed 42
    echo ""
fi

# Step 3: Verify with samples
echo "=== Step 3: Generating prior samples ==="
python training/sample_prior.py
echo ""

echo "=== Training pipeline complete ==="
echo "LoRA weights: training/checkpoints/cetus_pixart_lora/transformer_lora/"
echo "Sample images: outputs/pixart_prior_samples.png"
echo ""
echo "To run reconstruction:"
echo "  python training/reconstruct_volume_pixart.py"
