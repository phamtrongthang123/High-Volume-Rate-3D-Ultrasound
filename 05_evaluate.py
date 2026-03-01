"""
05_evaluate.py — Metrics and visualization.

Computes PSNR, SSIM, LPIPS between ground truth and reconstructed volume
on the missing (reconstructed) elevation planes only. Generates:
- Side-by-side comparison: GT | Reconstruction | |Difference|
- Elevation profile: pixel intensity + inter-plane TV along elevation axis

The paper evaluates on B-plane PSNR/LPIPS and A-plane SSIM/LPIPS across
acceleration rates r ∈ {2, 3, 6, 10} (Section IV, Figures 5-7).
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from zea import init_device
from zea.metrics import Metrics
from zea.ops import Pipeline, ScanConvert

# --- Config ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
ACCEL_RATE = 4
N_DISPLAY = 4  # Number of planes to display

# --- Init device ---
init_device(verbose=False)

# --- Load data ---
print("Loading volumes...")
volume_gt = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume.npy"))
volume_recon = np.load(os.path.join(OUTPUT_DIR, "reconstructed_volume.npy"))
print(f"GT shape: {volume_gt.shape}, Recon shape: {volume_recon.shape}")

# --- Identify observed vs missing planes ---
observed_indices = list(range(0, volume_gt.shape[0], ACCEL_RATE))
missing_indices = [i for i in range(volume_gt.shape[0]) if i not in observed_indices]
print(f"Observed planes: {observed_indices}")
print(f"Missing planes: {missing_indices}")

# --- Compute metrics on missing planes only ---
# Data is in [-1, 1] range
print("\n=== Computing Metrics (missing planes only) ===")
metrics = Metrics(["psnr", "ssim", "lpips"], image_range=(-1, 1))

gt_missing = volume_gt[missing_indices]
recon_missing = volume_recon[missing_indices]

results = metrics(gt_missing, recon_missing)

for metric_name, value in results.items():
    print(f"  {metric_name}: {float(value):.4f}")

# --- Per-plane metrics ---
print("\n=== Per-Plane Metrics ===")
for i, plane_idx in enumerate(missing_indices):
    gt_plane = volume_gt[plane_idx:plane_idx+1]
    recon_plane = volume_recon[plane_idx:plane_idx+1]
    plane_results = metrics(gt_plane, recon_plane)
    vals = " | ".join(f"{k}: {float(v):.4f}" for k, v in plane_results.items())
    print(f"  Plane {plane_idx}: {vals}")

# --- Visualization ---
print("\n=== Generating Visualization ===")
pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = pipeline.prepare_parameters(
    theta_range=[-0.78, 0.78],
    rho_range=[0, 1],
)


def scan_convert_img(img_2d):
    """Scan convert a single 2D image."""
    img = img_2d[np.newaxis]  # add batch dim
    out = pipeline(data=img, **parameters)["data"]
    return np.array(out[0])


# Select planes to display — spread across the middle of the volume
# (edge planes have little signal in CETUS data)
mid = len(missing_indices) // 2
half_span = N_DISPLAY // 2
display_planes = missing_indices[mid - half_span : mid - half_span + N_DISPLAY]
n_display = len(display_planes)

fig, axes = plt.subplots(n_display, 3, figsize=(15, 5 * n_display))
if n_display == 1:
    axes = axes[np.newaxis, :]

for row, plane_idx in enumerate(display_planes):
    gt_img = scan_convert_img(volume_gt[plane_idx, ..., 0])
    recon_img = scan_convert_img(volume_recon[plane_idx, ..., 0])
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
    f"Volume Reconstruction Evaluation ({ACCEL_RATE}x acceleration)\n"
    + " | ".join(f"{k}: {float(v):.4f}" for k, v in results.items()),
    fontsize=14,
)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "05_evaluation.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved evaluation figure to {save_path}")

# --- Elevation profile visualization ---
print("Generating elevation profile...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pick a central pixel
H = volume_gt.shape[1]
mid_row = H // 2

gt_profile = volume_gt[:, mid_row, volume_gt.shape[2]//2, 0]
recon_profile = volume_recon[:, mid_row, volume_recon.shape[2]//2, 0]

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

# TV along elevation
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
save_path = os.path.join(OUTPUT_DIR, "05_elevation_profile.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved elevation profile to {save_path}")

print("\nEvaluation complete.")
