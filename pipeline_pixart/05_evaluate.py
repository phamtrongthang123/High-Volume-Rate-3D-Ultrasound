"""
05_evaluate.py — Metrics and visualization for PixArt pipeline (pure PyTorch).

Computes PSNR, SSIM, LPIPS between ground truth and reconstructed volume
on the missing (reconstructed) elevation planes only. Generates:
- Side-by-side comparison: GT | Reconstruction | |Difference|
- Elevation profile: pixel intensity + inter-plane TV along elevation axis

Does NOT use JAX or ZEA — uses scikit-image and torchvision for metrics.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
ACCEL_RATE = 4
N_DISPLAY = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- VGG16 perceptual feature extractor for LPIPS approximation ---
class _VGGPerceptual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use layers up to relu2_2 (index 9) like original LPIPS
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:10])
        self.features.requires_grad_(False)
        # ImageNet normalisation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)


def _compute_lpips_vgg(imgs_a, imgs_b, vgg_model):
    """Compute VGG perceptual distance (LPIPS approximation).

    Args:
        imgs_a, imgs_b: (N, H, W) float numpy arrays in [-1, 1].
        vgg_model: _VGGPerceptual instance on device.

    Returns:
        float: mean perceptual distance.
    """
    def _prep(arr):
        # (N, H, W) → (N, 3, 224, 224) in [0, 1]
        t = torch.from_numpy(arr).float().unsqueeze(1)  # (N, 1, H, W)
        t = (t + 1.0) / 2.0  # [-1,1] → [0,1]
        t = t.repeat(1, 3, 1, 1)  # grey→RGB
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        return t.to(device)

    with torch.no_grad():
        fa = vgg_model(_prep(imgs_a))
        fb = vgg_model(_prep(imgs_b))
        # normalise features per-channel
        fa = F.normalize(fa.view(fa.shape[0], fa.shape[1], -1), dim=2)
        fb = F.normalize(fb.view(fb.shape[0], fb.shape[1], -1), dim=2)
        dist = (fa - fb).pow(2).mean().item()
    return dist


def compute_metrics(gt_planes, recon_planes, vgg_model):
    """Compute PSNR, SSIM, LPIPS on a stack of 2D planes.

    Args:
        gt_planes, recon_planes: (N, H, W, 1) float numpy arrays in [-1, 1].
        vgg_model: _VGGPerceptual instance.

    Returns:
        dict with 'psnr', 'ssim', 'lpips' keys.
    """
    N = gt_planes.shape[0]
    psnr_vals, ssim_vals = [], []

    for i in range(N):
        g = gt_planes[i, ..., 0]      # (H, W)
        r = recon_planes[i, ..., 0]   # (H, W)
        # PSNR: data_range = 2.0 since images are in [-1, 1]
        psnr_vals.append(peak_signal_noise_ratio(g, r, data_range=2.0))
        ssim_vals.append(structural_similarity(g, r, data_range=2.0))

    lpips_val = _compute_lpips_vgg(
        gt_planes[..., 0], recon_planes[..., 0], vgg_model
    )

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(lpips_val),
    }


def main():
    # --- Load data ---
    print("Loading volumes...")
    volume_gt = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume.npy"))
    volume_recon = np.load(os.path.join(OUTPUT_DIR, "reconstructed_volume_pixart.npy"))
    print(f"GT shape: {volume_gt.shape}, Recon shape: {volume_recon.shape}")

    # --- Identify missing planes ---
    observed_indices = list(range(0, volume_gt.shape[0], ACCEL_RATE))
    missing_indices = [i for i in range(volume_gt.shape[0]) if i not in observed_indices]
    print(f"Observed planes: {observed_indices[:8]}...")
    print(f"Missing planes: {missing_indices[:8]}...")

    # --- Load VGG model for LPIPS ---
    print("Loading VGG16 for perceptual metric...")
    vgg_model = _VGGPerceptual().to(device).eval()

    # --- Compute metrics ---
    print("\n=== Computing Metrics (missing planes only) ===")
    gt_missing = volume_gt[missing_indices]
    recon_missing = volume_recon[missing_indices]

    results = compute_metrics(gt_missing, recon_missing, vgg_model)

    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # --- Save JSON ---
    json_data = dict(results)
    json_data["accel_rate"] = ACCEL_RATE
    json_path = os.path.join(OUTPUT_DIR, "pixart_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved metrics JSON to {json_path}")

    # --- Per-plane metrics ---
    print("\n=== Per-Plane Metrics ===")
    for plane_idx in missing_indices:
        g = volume_gt[plane_idx:plane_idx+1]
        r = volume_recon[plane_idx:plane_idx+1]
        pr = compute_metrics(g, r, vgg_model)
        vals = " | ".join(f"{k}: {float(v):.4f}" for k, v in pr.items())
        print(f"  Plane {plane_idx}: {vals}")

    # --- Visualization ---
    print("\n=== Generating Visualization ===")
    mid = len(missing_indices) // 2
    half_span = N_DISPLAY // 2
    display_planes = missing_indices[mid - half_span : mid - half_span + N_DISPLAY]
    n_display = len(display_planes)

    fig, axes = plt.subplots(n_display, 3, figsize=(15, 5 * n_display))
    if n_display == 1:
        axes = axes[np.newaxis, :]

    for row, plane_idx in enumerate(display_planes):
        gt_img = volume_gt[plane_idx, ..., 0]
        recon_img = volume_recon[plane_idx, ..., 0]
        diff_img = np.abs(gt_img - recon_img)

        axes[row, 0].imshow(gt_img, cmap="gray", vmin=-1, vmax=1)
        axes[row, 0].set_title(f"Ground Truth (plane {plane_idx})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon_img, cmap="gray", vmin=-1, vmax=1)
        axes[row, 1].set_title(f"Reconstruction (plane {plane_idx})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(diff_img, cmap="hot", vmin=0, vmax=1)
        axes[row, 2].set_title(f"|Difference| (plane {plane_idx})")
        axes[row, 2].axis("off")

    plt.suptitle(
        f"PixArt Volume Reconstruction Evaluation ({ACCEL_RATE}x acceleration)\n"
        + " | ".join(f"{k}: {float(v):.4f}" for k, v in results.items()),
        fontsize=14,
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "pixart_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation figure to {save_path}")

    # --- Elevation profile ---
    print("Generating elevation profile...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    H = volume_gt.shape[1]
    gt_profile = volume_gt[:, H // 2, volume_gt.shape[2] // 2, 0]
    recon_profile = volume_recon[:, H // 2, volume_recon.shape[2] // 2, 0]

    axes[0].plot(gt_profile, "b-o", label="Ground Truth", markersize=4)
    axes[0].plot(recon_profile, "r--s", label="Reconstructed", markersize=4)
    for idx in observed_indices:
        if idx < len(gt_profile):
            axes[0].axvline(x=idx, color="green", alpha=0.3, linestyle=":")
    axes[0].set_xlabel("Elevation plane index")
    axes[0].set_ylabel("Pixel intensity")
    axes[0].set_title("Elevation Profile (center pixel)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diff_elev_gt = np.abs(np.diff(volume_gt, axis=0)).mean(axis=(1, 2, 3))
    diff_elev_recon = np.abs(np.diff(volume_recon, axis=0)).mean(axis=(1, 2, 3))

    axes[1].plot(diff_elev_gt, "b-o", label="GT", markersize=4)
    axes[1].plot(diff_elev_recon, "r--s", label="Reconstructed", markersize=4)
    axes[1].set_xlabel("Elevation plane transition")
    axes[1].set_ylabel("Mean absolute difference")
    axes[1].set_title("Inter-Plane Smoothness (TV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "pixart_elevation_profile.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved elevation profile to {save_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
