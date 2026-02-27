"""
01_verify_prior.py — Verify pretrained diffusion model works.

Loads the pretrained echonet-dynamic diffusion model (the score model
ε_θ from Eq. 4, eq:dsm) and runs unconditional sampling — i.e. sampling
from the learned prior p(x) via reverse diffusion (Eq. 3, eq:tweedie).
Saves visualization to verify the prior generates plausible cardiac echo frames.
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import jax.numpy as jnp
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.ops import Pipeline, ScanConvert
from zea.visualize import plot_image_grid

# --- Config ---
N_SAMPLES = 8
N_STEPS = 200  # Use 50 for quick testing
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading pretrained diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
print(f"Model loaded. Input shape: {model.input_shape}")

# --- Unconditional sampling ---
print(f"Generating {N_SAMPLES} unconditional samples ({N_STEPS} steps)...")
samples = model.sample(n_samples=N_SAMPLES, n_steps=N_STEPS, verbose=True)
samples_np = np.array(samples)
print(f"Samples shape: {samples_np.shape}, range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")

# --- Scan convert + visualize ---
# Scan conversion params: these define the ultrasound sector geometry.
# theta_range: angular sweep in radians (0.78 rad ≈ π/4 ≈ 45°, ~90° total FOV).
# rho_range: normalized depth [0,1].
# Standard values for echonet-dynamic dataset. The sampled from diffusion is in polor coord. 
pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = pipeline.prepare_parameters(
    theta_range=[-0.78, 0.78],
    rho_range=[0, 1],
)
processed = jnp.squeeze(samples, axis=-1)
processed = pipeline(data=processed, **parameters)["data"]

fig, _ = plot_image_grid(
    processed,
    vmin=-1,
    vmax=1,
    ncols=4,
)
fig.suptitle("Unconditional Samples from Diffusion Prior (echonet-dynamic)")

save_path = os.path.join(OUTPUT_DIR, "01_prior_samples.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {save_path}")
print("Prior verification complete.")
