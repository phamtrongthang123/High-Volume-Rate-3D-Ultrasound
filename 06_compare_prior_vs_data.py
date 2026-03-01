"""
06_compare_prior_vs_data.py — Visual comparison of diffusion prior vs actual B-planes.

Generates 4 unconditional prior samples and extracts 4 B-planes from the pseudo_volume,
then plots them side-by-side to reveal the distribution mismatch that bounds reconstruction
quality at ~17 dB PSNR.
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.ops import Pipeline, ScanConvert
from zea.visualize import plot_image_grid

# --- Config ---
N_SAMPLES = 4
N_STEPS = 200
B_PLANE_INDICES = [0, 37, 74, 111]  # azimuth positions across the volume
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Init device ---
init_device(verbose=False)

# --- Load model and generate prior samples ---
print("Loading pretrained diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
print(f"Model loaded. Input shape: {model.input_shape}")

print(f"Generating {N_SAMPLES} unconditional prior samples ({N_STEPS} steps)...")
samples = model.sample(n_samples=N_SAMPLES, n_steps=N_STEPS, verbose=True)
samples_np = np.array(samples)  # (4, 112, 112, 1)
print(f"Prior samples shape: {samples_np.shape}, range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")

# --- Load B-planes from pseudo_volume ---
volume_path = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
print(f"Loading pseudo_volume from {volume_path}...")
volume = np.load(volume_path)  # (112, 112, 112, 1)
print(f"Volume shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")

b_planes = volume[:, B_PLANE_INDICES, :, :]   # (112, 4, 112, 1)
b_planes = np.transpose(b_planes, (1, 0, 2, 3))  # (4, 112, 112, 1)
print(f"B-planes shape: {b_planes.shape}")

# --- ScanConvert pipeline ---
pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = pipeline.prepare_parameters(theta_range=[-0.78, 0.78], rho_range=[0, 1])

def scan_convert(images_np):
    """Apply scan conversion to a (N, H, W, 1) array, returns converted array."""
    imgs = jnp.squeeze(jnp.array(images_np), axis=-1)  # (N, H, W)
    converted = pipeline(data=imgs, **parameters)["data"]
    return np.array(converted)

print("Applying ScanConvert to prior samples...")
prior_converted = scan_convert(samples_np)

print("Applying ScanConvert to B-planes...")
data_converted = scan_convert(b_planes)

# --- Plot 2-row x 4-column grid ---
vmin, vmax = -1, 1
ncols = N_SAMPLES
nrows = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
fig.suptitle(
    "Diffusion Prior vs Actual B-Planes\n"
    "Top: unconditional prior samples  |  Bottom: pseudo_volume B-planes",
    fontsize=13,
)

for col_idx in range(ncols):
    # Top row: prior samples
    ax_top = axes[0, col_idx]
    ax_top.imshow(prior_converted[col_idx], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ax_top.set_title(f"Prior sample {col_idx + 1}", fontsize=10)
    ax_top.axis("off")

    # Bottom row: actual B-planes
    ax_bot = axes[1, col_idx]
    ax_bot.imshow(data_converted[col_idx], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    ax_bot.set_title(f"B-plane j={B_PLANE_INDICES[col_idx]}", fontsize=10)
    ax_bot.axis("off")

axes[0, 0].set_ylabel("Prior samples", fontsize=11, labelpad=8)
axes[1, 0].set_ylabel("Actual B-planes", fontsize=11, labelpad=8)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "06_prior_vs_data.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved comparison to {save_path}")
print("Done.")
