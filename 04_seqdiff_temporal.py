"""
04_seqdiff_temporal.py — SeqDiff warm-start demo (Algorithm 1, lines 16-23).

Demonstrates temporal acceleration: instead of running full 200-step
diffusion for every frame, use the previous frame's reconstruction
as a warm start and run only ~50 steps.

Paper Algorithm 1 mapping:
  Cold start (no X^prev):   X_τ ~ N(0, σ²_T I), τ' ← T       (Algo 1 lines 21-22)
  SeqDiff (X^prev avail):   X_0 ← X^prev                       (Algo 1 line 17)
                             X_τ ← α_τ' X_0 + σ_τ' ε           (Algo 1 line 19,
                                                                  Eq. 2 eq:forward-diffusion)
                             → runs from τ' instead of T

  Cold start (Frame 1):  200 steps from pure noise
  SeqDiff (Frame 2):     50 steps from forward-diffused Frame 1 result
  Expected speedup:      ~4x

Uses manual DPS loop (matching 03_reconstruct_volume.py) instead of
model.posterior_sample() which produces near-constant dark output.
"""

import env_setup  # noqa: F401 — must be first

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.ops import Pipeline, ScanConvert

# --- Config ---
N_STEPS = 200        # T, total diffusion steps (Algo 1 line 25)
SEQDIFF_TAU = 50     # τ', warm-start step (Algo 1 line 11, 19)
GAMMA = 15.0         # γ, guidance strength (matching 03_reconstruct_volume.py)
ACCEL_RATE = 4       # Elevation acceleration rate
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape  # (H, W, 1)
H, W = img_shape[0], img_shape[1]
print(f"Model loaded. Input shape: {img_shape}")

# --- Load both pseudo-volumes ---
volume_t1 = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume.npy"))
volume_t2 = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume_t2.npy"))
N_el, N_az, N_ax, C = volume_t1.shape
print(f"Volume t1 shape: {volume_t1.shape}")
print(f"Volume t2 shape: {volume_t2.shape}")

# --- Choose azimuth index for B-plane reconstruction ---
j = N_az // 2
print(f"Reconstructing B-plane at azimuth index j={j}")

# --- Elevation row mask (matching 03_reconstruct_volume.py) ---
observed_rows = list(range(0, N_el, ACCEL_RATE))
A = np.zeros((N_el, N_ax, C), dtype=np.float32)
A[observed_rows] = 1.0
A_batch = A[np.newaxis]  # (1, H, W, C) — broadcasts over batch
print(f"Elevation mask: {len(observed_rows)} observed rows out of {N_el} "
      f"(acceleration {ACCEL_RATE}x)")


# --- Manual DPS step (adapted from 03_reconstruct_volume.py) ---
def one_diffusion_step(
    model,
    x_tau,
    y,
    A,
    sigma_tau,
    alpha_tau,
    sigma_tau_minus_1,
    alpha_tau_minus_1,
    gamma,
    debug=False,
):
    """One diffusion step with DPS guidance (Algo 1 lines 27-32).

    Args:
        model: DiffusionModel instance.
        x_tau: Current noisy B-planes x_τ, shape (B, H, W, C).
        y: Measurements y = Ax, shape (B, H, W, C).
        A: Measurement operator (binary mask), shape (1, H, W, C).
        sigma_tau: Noise rate σ_τ, shape (1, 1, 1, 1).
        alpha_tau: Signal rate α_τ, shape (1, 1, 1, 1).
        sigma_tau_minus_1: Noise rate σ_{τ-1}, shape (1, 1, 1, 1).
        alpha_tau_minus_1: Signal rate α_{τ-1}, shape (1, 1, 1, 1).
        gamma: DPS guidance weight γ (scalar).

    Returns:
        x_tau_minus_1: Updated B-planes x_{τ-1}, shape (B, H, W, C).
        x_0_tau: Tweedie estimate x_{0|τ}, shape (B, H, W, C).
    """
    x_tau = jnp.asarray(x_tau)
    y = jnp.asarray(y)
    A = jnp.asarray(A)

    batch_size = x_tau.shape[0]

    # Line 27: ε = ε_θ(x_τ, σ²_τ)
    def eps_theta(x):
        sigma_2 = jnp.full((batch_size, 1, 1, 1), sigma_tau**2, dtype=x.dtype)
        return model.ema_network([x, sigma_2], training=False)

    epsilon, vjp_fn = jax.vjp(eps_theta, x_tau)

    # Line 28: x_{0|τ} = (x_τ - σ_τ ε) / α_τ  (Tweedie, Eq. 3)
    x_0_tau = (x_tau - sigma_tau * epsilon) / alpha_tau

    # Line 29: M ← y − A x_{0|τ}  (measurement error)
    M = y - A * x_0_tau

    # Line 30: P ← (I − σ_τ ∇ε_θ)^T A^T  (Projection via VJP)
    def P(v):
        u = A * v
        return u - sigma_tau * vjp_fn(u)[0]

    PM = P(M)
    M_norm = jnp.sqrt(jnp.sum(M**2)) + 1e-8

    # Line 31: DDIM reverse step
    x_tau_minus_1 = alpha_tau_minus_1 * x_0_tau + sigma_tau_minus_1 * epsilon

    # Line 32: DPS guidance
    x_tau_minus_1 = x_tau_minus_1 + gamma / alpha_tau * PM / M_norm

    if debug:
        print(f"  M_norm={float(M_norm):.3f}, "
              f"x range=[{float(x_tau_minus_1.min()):.3f}, "
              f"{float(x_tau_minus_1.max()):.3f}]")

    return np.array(x_tau_minus_1), np.array(x_0_tau)


# --- Precompute diffusion schedule ---
alphas = []
sigmas = []
step_size = model.max_t / N_STEPS
for step in range(N_STEPS + 1):
    diffusion_times = np.ones((1, 1, 1, 1)) * model.max_t - step * step_size
    sigma, alpha = model.diffusion_schedule(diffusion_times)
    alphas.append(float(np.array(alpha)[0, 0, 0, 0]))
    sigmas.append(float(np.array(sigma)[0, 0, 0, 0]))
alphas = np.array(alphas)
sigmas = np.array(sigmas)

# --- Extract GT B-planes at azimuth j ---
gt_t1 = volume_t1[:, j, :, :]  # (N_el, N_ax, C)
gt_t2 = volume_t2[:, j, :, :]  # (N_el, N_ax, C)

# Measurements: y = A * gt (observed elevation rows)
y_t1 = (A * gt_t1)[np.newaxis]  # (1, N_el, N_ax, C)
y_t2 = (A * gt_t2)[np.newaxis]  # (1, N_el, N_ax, C)

# === Frame 1: Cold start (full 200 steps) ===
print(f"\n=== Frame 1: Cold start ({N_STEPS} steps) ===")
x_tau = np.random.randn(1, N_el, N_ax, C).astype(np.float32)

t_start = time.time()
for step in range(N_STEPS):
    sigma_tau = jnp.full((1, 1, 1, 1), sigmas[step])
    alpha_tau = jnp.full((1, 1, 1, 1), alphas[step])
    sigma_tau_minus_1 = jnp.full((1, 1, 1, 1), sigmas[step + 1])
    alpha_tau_minus_1 = jnp.full((1, 1, 1, 1), alphas[step + 1])

    do_debug = step < 5 or (step + 1) % 50 == 0
    x_tau, _ = one_diffusion_step(
        model, x_tau, y_t1, A_batch,
        sigma_tau, alpha_tau, sigma_tau_minus_1, alpha_tau_minus_1,
        GAMMA, debug=do_debug,
    )
    # Data consistency: replace observed rows with noised GT
    a_next = float(np.array(alpha_tau_minus_1).ravel()[0])
    s_next = float(np.array(sigma_tau_minus_1).ravel()[0])
    noise_dc = np.random.randn(1, len(observed_rows), N_ax, C).astype(np.float32)
    x_tau[:, observed_rows, :, :] = (
        a_next * gt_t1[np.newaxis][:, observed_rows, :, :]
        + s_next * noise_dc
    )
    if do_debug:
        print(f"  Step {step + 1}/{N_STEPS}")

recon_t1 = x_tau  # (1, N_el, N_ax, C)
time_cold = time.time() - t_start
print(f"Cold start time: {time_cold:.2f}s")
print(f"Recon t1 range: [{recon_t1.min():.3f}, {recon_t1.max():.3f}]")

# === Frame 2: SeqDiff warm-start (50 steps) ===
start_step = N_STEPS - SEQDIFF_TAU
print(f"\n=== Frame 2: SeqDiff warm-start "
      f"(start_step={start_step}, running {SEQDIFF_TAU} steps) ===")

# Forward-diffuse Frame 1 result to step start_step (Algo 1 line 19)
alpha_tau_prime = alphas[start_step]
sigma_tau_prime = sigmas[start_step]
noise = np.random.randn(*recon_t1.shape).astype(np.float32)
x_tau = alpha_tau_prime * recon_t1 + sigma_tau_prime * noise
print(f"Forward-diffused recon_t1 to step {start_step}: "
      f"α={alpha_tau_prime:.4f}, σ={sigma_tau_prime:.4f}")

t_start = time.time()
for step in range(start_step, N_STEPS):
    sigma_tau = jnp.full((1, 1, 1, 1), sigmas[step])
    alpha_tau = jnp.full((1, 1, 1, 1), alphas[step])
    sigma_tau_minus_1 = jnp.full((1, 1, 1, 1), sigmas[step + 1])
    alpha_tau_minus_1 = jnp.full((1, 1, 1, 1), alphas[step + 1])

    do_debug = (step - start_step) < 3 or (step + 1) % 50 == 0
    x_tau, _ = one_diffusion_step(
        model, x_tau, y_t2, A_batch,
        sigma_tau, alpha_tau, sigma_tau_minus_1, alpha_tau_minus_1,
        GAMMA, debug=do_debug,
    )
    # Data consistency: replace observed rows with noised GT
    a_next = float(np.array(alpha_tau_minus_1).ravel()[0])
    s_next = float(np.array(sigma_tau_minus_1).ravel()[0])
    noise_dc = np.random.randn(1, len(observed_rows), N_ax, C).astype(np.float32)
    x_tau[:, observed_rows, :, :] = (
        a_next * gt_t2[np.newaxis][:, observed_rows, :, :]
        + s_next * noise_dc
    )
    if do_debug:
        print(f"  Step {step + 1}/{N_STEPS}")

recon_t2 = x_tau  # (1, N_el, N_ax, C)
time_warm = time.time() - t_start
print(f"SeqDiff warm-start time: {time_warm:.2f}s")
print(f"Recon t2 range: [{recon_t2.min():.3f}, {recon_t2.max():.3f}]")

# --- Compare ---
speedup = time_cold / time_warm if time_warm > 0 else float("inf")
print(f"\n=== Results ===")
print(f"Cold start:   {time_cold:.2f}s ({N_STEPS} steps)")
print(f"SeqDiff:      {time_warm:.2f}s ({SEQDIFF_TAU} steps)")
print(f"Speedup:      {speedup:.1f}x")

# --- Save results ---
np.save(os.path.join(OUTPUT_DIR, "seqdiff_recon_t1.npy"), recon_t1)
np.save(os.path.join(OUTPUT_DIR, "seqdiff_recon_t2.npy"), recon_t2)

# --- Visualization ---
pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = pipeline.prepare_parameters(
    theta_range=[-0.78, 0.78],
    rho_range=[0, 1],
)


def scan_convert_img(img_2d):
    """Scan convert a single 2D image."""
    img = jnp.squeeze(img_2d, axis=-1) if img_2d.ndim == 3 else img_2d
    img = img[np.newaxis]  # add batch dim
    out = pipeline(data=img, **parameters)["data"]
    return np.array(out[0])


gt_t1_cart = scan_convert_img(gt_t1)
gt_t2_cart = scan_convert_img(gt_t2)
recon_t1_cart = scan_convert_img(recon_t1[0])
recon_t2_cart = scan_convert_img(recon_t2[0])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(gt_t1_cart, cmap="gray", vmin=-1, vmax=1)
axes[0].set_title(f"GT Frame 1 (ED, az={j})")
axes[1].imshow(recon_t1_cart, cmap="gray", vmin=-1, vmax=1)
axes[1].set_title(f"Cold Start ({N_STEPS} steps)\n{time_cold:.1f}s")
axes[2].imshow(gt_t2_cart, cmap="gray", vmin=-1, vmax=1)
axes[2].set_title(f"GT Frame 2 (ES, az={j})")
axes[3].imshow(recon_t2_cart, cmap="gray", vmin=-1, vmax=1)
axes[3].set_title(f"SeqDiff ({SEQDIFF_TAU} steps)\n{time_warm:.1f}s, {speedup:.1f}x faster")

for ax in axes:
    ax.axis("off")

plt.suptitle("SeqDiff Temporal Acceleration Demo")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "04_seqdiff_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualization to {save_path}")
print("SeqDiff demo complete.")
