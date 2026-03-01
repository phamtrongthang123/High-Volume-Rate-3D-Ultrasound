"""
sample_prior.py — Generate samples from the trained PixArt-α model to verify quality.

Loads the PixArt-α pipeline with LoRA weights, generates 16 unconditional
samples, and saves them as a grid image alongside real CETUS B-planes
for visual comparison.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pixart_diffusion_model import PixArtDiffusionModel

# --- Config ---
PRETRAINED_PATH = "PixArt-alpha/PixArt-XL-2-512x512"
LORA_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "cetus_pixart_lora", "transformer_lora")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
N_SAMPLES = 16
N_STEPS = 20
GRID_COLS = 4

# Override LoRA path from command line
if len(sys.argv) > 1:
    LORA_PATH = sys.argv[1]


def make_grid(images, ncols=4, padding=2):
    """Arrange a list of (H, W) images into a grid."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    h, w = images[0].shape[:2]
    grid = np.ones(
        (nrows * (h + padding) - padding, ncols * (w + padding) - padding),
        dtype=np.uint8,
    ) * 255

    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        y = row * (h + padding)
        x = col * (w + padding)
        grid[y:y + h, x:x + w] = img

    return grid


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check LoRA path
    lora_path = LORA_PATH if os.path.exists(LORA_PATH) else None
    if lora_path is None:
        print(f"WARNING: LoRA weights not found at {LORA_PATH}")
        print("Generating samples with base PixArt-α (no fine-tuning).")
    else:
        print(f"Loading LoRA weights from {lora_path}")

    # Load model
    print(f"Loading PixArt-α from {PRETRAINED_PATH}...")
    model = PixArtDiffusionModel(PRETRAINED_PATH, lora_path=lora_path, device=device)
    print("Model loaded.")

    # Generate samples
    print(f"Generating {N_SAMPLES} samples with {N_STEPS} diffusion steps...")
    all_samples = []
    batch_size = 4  # Generate in batches to manage memory

    for i in range(0, N_SAMPLES, batch_size):
        n = min(batch_size, N_SAMPLES - i)
        samples = model.sample(n_samples=n, n_steps=N_STEPS)  # (n, 112, 112, 1)
        all_samples.append(samples)
        print(f"  Generated batch {i // batch_size + 1}/{(N_SAMPLES + batch_size - 1) // batch_size}")

    all_samples = np.concatenate(all_samples, axis=0)  # (N_SAMPLES, 112, 112, 1)

    # Convert to uint8 images
    sample_images = []
    for i in range(N_SAMPLES):
        img = all_samples[i, :, :, 0]  # (112, 112)
        img = (img + 1) / 2 * 255  # [-1, 1] → [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)
        sample_images.append(img)

    # Make grid of generated samples
    gen_grid = make_grid(sample_images, ncols=GRID_COLS)
    gen_path = os.path.join(OUTPUT_DIR, "pixart_prior_samples.png")
    Image.fromarray(gen_grid).save(gen_path)
    print(f"Saved generated sample grid to {gen_path}")

    # Load real CETUS B-planes for comparison (if available)
    volume_path = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
    if os.path.exists(volume_path):
        X_gt = np.load(volume_path)  # (112, 112, 112, 1)
        real_images = []
        # Pick evenly spaced azimuth slices
        indices = np.linspace(0, X_gt.shape[1] - 1, N_SAMPLES).astype(int)
        for j in indices:
            bplane = X_gt[:, j, :, 0]  # (112, 112)
            bplane = (bplane + 1) / 2 * 255
            bplane = np.clip(bplane, 0, 255).astype(np.uint8)
            real_images.append(bplane)

        real_grid = make_grid(real_images, ncols=GRID_COLS)

        # Side-by-side comparison
        pad = 20
        combined_h = max(gen_grid.shape[0], real_grid.shape[0])
        combined_w = gen_grid.shape[1] + pad + real_grid.shape[1]
        combined = np.ones((combined_h, combined_w), dtype=np.uint8) * 255

        combined[:gen_grid.shape[0], :gen_grid.shape[1]] = gen_grid
        combined[:real_grid.shape[0], gen_grid.shape[1] + pad:] = real_grid

        comparison_path = os.path.join(OUTPUT_DIR, "pixart_prior_vs_real.png")
        Image.fromarray(combined).save(comparison_path)
        print(f"Saved comparison (generated | real) to {comparison_path}")
    else:
        print("Ground truth volume not found, skipping comparison.")

    print("\nSample generation complete.")
    print(f"  Sample range: [{all_samples.min():.3f}, {all_samples.max():.3f}]")
    print(f"  Sample mean: {all_samples.mean():.3f}")
    print(f"  Sample std: {all_samples.std():.3f}")


if __name__ == "__main__":
    main()
