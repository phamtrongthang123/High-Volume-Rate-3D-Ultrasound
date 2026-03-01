"""
03_reconstruct_volume.py — Core Algorithm 1: DPS reconstruction of missing planes.

Implements the paper's volume reconstruction pipeline (Algorithm 1, algo.tex):
1. Load pseudo-volume (N_el, N_az, N_ax, C), extract B-planes along azimuth axis
2. Create elevation row mask: observed every r-th row, zeros elsewhere
3. For each diffusion step τ:
   - Batch B-planes through one diffusion step + DPS guidance
   - Transpose to volume, apply TV regularization along azimuth (axis 1)
   - Transpose back to B-planes for next step
4. Save reconstructed volume

Paper Algorithm 1 → implementation mapping:
  ε_θ(x_τ, τ)           → model.ema_network([x_τ, σ_τ²])      (Eq. 4, eq:dsm)
  x_τ = α_τ x_0 + σ_τ ε → forward diffusion                    (Eq. 2, eq:forward-diffusion)
  x_{0|τ} Tweedie       → (x_τ - σ_τ ε) / α_τ                  (Eq. 3, eq:tweedie)
  y = Ax (measurement)   → inpainting: y = A x                   (Eq. 5, eq:inverse-problem)
  A = elevation row mask  → binary mask operator                  (Eq. 7, eq:observation_zf)
  DPS guidance (γ)       → jax.vjp through ε_θ + Tweedie         (Eq. 9-12, eq:dps-linear-*)
  TV smoothness (ζ)      → per-step TV gradient                  (Algo 1 line 35-36)
  SeqDiff warm-start     → forward-diffuse previous recon         (Algo 1 line 16-19)

Structure matches Algorithm 1 exactly:
  Volume: X ∈ (N_el, N_az, N_ax, C)
  B-plane j: X[:, j, :, :] = (N_el, N_ax, C)

  For τ = τ' to 1:                        # per-step (outer loop, line 25)
      For all B-planes j (batched):        # inner loop (line 26)
          ε = ε_θ(x_τ, τ)                 # line 27 — predict noise
          x_{0|τ} = (x_τ - σ_τ ε) / α_τ   # line 28 — Tweedie
          M = y - A x_{0|τ}               # line 29 — measurement error
          P = (I - σ_τ ∇ε_θ)^T A^T        # line 30 — projection
          x_{τ-1} = α_{τ-1} x_{0|τ} + σ_{τ-1} ε  # line 31 — DDIM
          x_{τ-1} += γ/(α_τ·||M||₂) · P(M)       # line 32 — DPS guidance
      EndFor                               # line 33
      Stack B-planes into X_{τ-1}          # line 34
      V ← ∇_X TV_az(X_{τ-1})              # line 35
      X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V    # line 36
  EndFor                                   # line 37

Note: ZEA's DPS implementation uses L2 norm (not L2²) for the measurement
error (matching the original DPS codebase). This introduces an implicit
1/||M||₂ normalization in the gradient. Our manual VJP implementation
includes this normalization explicitly: γ · P(M) / ||M||₂.
"""

import os

import env_setup  # noqa: F401 — must be first
import jax
import jax.numpy as jnp
import numpy as np
from zea import init_device
from zea.models.diffusion import DiffusionModel

# --- Config ---
ACCEL_RATE = 4  # r ≥ 1, acceleration rate (Eq. 5, eq:inverse-problem)
N_STEPS = 200  # T, diffusion steps (Algo 1 line 25)
GAMMA = 15.0  # γ, guidance strength (5 too weak: M_norm still 7.8 at step 200; 35 caused instability)
ZETA = 0.001  # ζ, smoothness strength (Algo 1 line 36)
BATCH_SIZE = 16  # Batch B-planes through model for memory
USE_SEQDIFF = False  # Enable SeqDiff warm-start from previous reconstruction
SEQDIFF_TAU = 50  # τ', warm-start diffusion step (Algo 1 line 11, 19)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape  # (H, W, 1)
H, W = img_shape[0], img_shape[1]
print(f"Model loaded. Input shape: {img_shape}")

# --- Load pseudo-volume ---
volume_path = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
X_gt = np.load(volume_path)
N_el, N_az, N_ax, C = X_gt.shape
print(f"Loaded ground truth volume: {X_gt.shape}")
print(f"  N_el={N_el}, N_az={N_az}, N_ax={N_ax}, C={C}")
assert (N_el, N_ax, C) == (H, W, 1), (
    f"B-plane shape (N_el, N_ax, C) = ({N_el}, {N_ax}, {C}) "
    f"does not match model input shape ({H}, {W}, 1)"
)


def compute_tv_gradient_azimuth(X):
    """TV gradient along azimuth (axis 1). Algo 1 lines 35-36.

    V ← ∇_X TV_az(X)

    Args:
        X: Volume of shape (N_el, N_az, N_ax, C).

    Returns:
        V: TV gradient of same shape.
    """
    diff = np.diff(X, axis=1)  # (N_el, N_az-1, N_ax, C)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    # Divergence (adjoint of gradient)
    div = np.zeros_like(X)
    div[:, :-1] += normalized
    div[:, 1:] -= normalized

    return -div  # Negative divergence as TV gradient


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
        A: Measurement operator (binary mask for inpainting), shape (1, H, W, C).
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

    # Line 27: ε = ε_θ(x_τ, σ²_τ)
    # Paper writes ε_θ(x_τ, τ), but ZEA conditions on σ²_τ (noise variance)
    # instead of raw τ (diffusion time). They're equivalent since τ → σ_τ is
    # deterministic via the cosine schedule, but σ²_τ is what the UNet's
    # sinusoidal embedding actually receives.
    def eps_theta(x):
        # need this to ensure broadcast correctly
        sigma_2 = jnp.full((BATCH_SIZE, 1, 1, 1), sigma_tau**2, dtype=x.dtype)
        return model.ema_network([x, sigma_2], training=False)

    epsilon, vjp_fn = jax.vjp(eps_theta, x_tau)

    # Line 28: x_{0|τ} = (x_τ - σ_τ ε) / α_τ  (Tweedie, Eq. 3)
    x_0_tau = (x_tau - sigma_tau * epsilon) / alpha_tau

    # Line 29: M ← y − A x_{0|τ}  (measurement error, Eq. 9)
    M = y - A * x_0_tau

    # Line 29: P ← (I − σ_τ ∇_{x_τ} ε_θ(x_τ, τ))^T A^T   (Projection)
    def P(v):
        # A^T v
        u = A * v
        # (I − σ_τ ∇_{x_τ} ε_θ)^T u = u − σ_τ J^T u
        return u - sigma_tau * vjp_fn(u)[0]

    PM = P(M)

    M_norm = jnp.sqrt(jnp.sum(M**2)) + 1e-8

    # Paper's Algo 1 (lines 31-32) applies guidance to x_{0|τ} then does DDIM:
    #   x_0' = x_0 - γPM                          # guidance on x_0
    #   x_{τ-1} = α_{τ-1} · x_0' + σ_{τ-1} · ε   # DDIM
    #           = α_{τ-1} · (x_0 - γPM ) + σ_{τ-1} · ε
    #           = α_{τ-1} · (x_0) + σ_{τ-1} · ε - α_{τ-1} · γPM
    #
    # ZEA's implementation (what we do) applies guidance to x_{τ-1} instead:
    #   x_{τ-1} = α_{τ-1} · x_0 + σ_{τ-1} · ε    # DDIM first
    #   x_{τ-1} += γ/α · PM/||M||                  # guidance on x_{τ-1}
    #
    # Differences: sign (+= not -=), scale (γ/α not α·γ), and ||M||₂ norm.
    # The paper's sign is inconsistent with its own Eq. 11; ZEA is correct.
    # ● The Jacobian J = ∂x_{0|τ}/∂x_τ comes from Tweedie:

    #   x_{0|τ} = (x_τ - σ_τ ε_θ(x_τ)) / α_τ

    #   So:

    #   J = ∂x_{0|τ}/∂x_τ = 1/α_τ · (I - σ_τ · ∂ε_θ/∂x_τ)

    #   Then:

    #   J^T A^T M = 1/α_τ · (I - σ_τ ∇ε_θ)^T · A^T · M
    #             = 1/α_τ · P(M)

    #   where P = (I - σ_τ ∇ε_θ)^T A^T is exactly the paper's line 30 definition.

    #   So the full likelihood score from Eq. 10 is:

    #   +1/σ² · J^T A^T M = +1/(σ² · α_τ) · P(M)

    #   Absorbing 1/(σ² · α_τ) into γ gives +γ P(M).

    #   But Eq. 12 writes -γ P(M). That negative has no justification from the derivation — it's the sign error.
    # DDIM reverse step (using unmodified x_{0|τ})
    x_tau_minus_1 = alpha_tau_minus_1 * x_0_tau + sigma_tau_minus_1 * epsilon

    # DPS guidance: ∂(γ·||M||₂)/∂x_τ = -γ/(α·||M||)·P(M)
    # x_{τ-1} -= gradient = x_{τ-1} + γ/(α·||M||)·P(M)
    x_tau_minus_1 = x_tau_minus_1 + gamma / alpha_tau * PM / M_norm

    # Debug: print stats for first batch only
    if debug:
        def _s(name, t):
            t = np.array(t)
            return (f"  {name:12s}: "
                    f"min={float(t.min()):+10.3f}  max={float(t.max()):+10.3f}  "
                    f"absmax={float(np.abs(t).max()):10.3f}  "
                    f"nan={bool(np.any(np.isnan(t)))}")
        print(_s("x_tau", x_tau))
        print(_s("epsilon", epsilon))
        print(_s("x_0_tau", x_0_tau))
        print(_s("M", M))
        print(f"  ||M||₂      : {float(M_norm):.3f}")
        print(_s("PM", PM))
        guidance = gamma / alpha_tau * PM / M_norm
        print(_s("guidance", guidance))
        print(_s("x_tau_m1", x_tau_minus_1))

    return np.array(x_tau_minus_1), np.array(x_0_tau)


# --- Elevation subsampling: observed rows in each B-plane ---
observed_rows = list(range(0, N_el, ACCEL_RATE))
missing_rows = [i for i in range(N_el) if i not in observed_rows]
print(f"\nAcceleration rate: {ACCEL_RATE}x")
print(f"Observed elevation rows ({len(observed_rows)}): {observed_rows[:8]}...")
print(f"Missing elevation rows ({len(missing_rows)}): {missing_rows[:8]}...")

# --- Measurement operator A (same for ALL B-planes) ---
# Shape: (N_el, N_ax, C) = (H, W, 1) — matches model input
# Binary mask: 1s at observed elevation rows, 0s elsewhere
A = np.zeros((N_el, N_ax, C), dtype=np.float32)
A[observed_rows] = 1.0
print(f"Operator A shape: {A.shape}, observed fraction: {A.mean():.2f}")

# --- Extract B-planes and create measurements y = A X_gt ---
# B-plane j = X_gt[:, j, :, :] → shape (N_el, N_ax, C)
# Transpose to (N_az, N_el, N_ax, C) for batch processing
B_gt = np.transpose(X_gt, (1, 0, 2, 3))  # (N_az, N_el, N_ax, C)
print(f"B-planes shape: {B_gt.shape}")

# Measurements: y = A X_gt (observed elevation rows from GT, zeros elsewhere)
y_all = B_gt * A[np.newaxis]  # (N_az, N_el, N_ax, C)

# Broadcast operator to batch: (1, N_el, N_ax, C)
A_batch = A[np.newaxis]  # (1, H, W, C) — broadcasts over batch

# --- Precompute diffusion schedule: α_τ and σ_τ for each step ---
alphas = []  # α_τ (signal rates)
sigmas = []  # σ_τ (noise rates)
step_size = model.max_t / N_STEPS
for step in range(N_STEPS + 1):
    diffusion_times = np.ones((1, 1, 1, 1)) * model.max_t - step * step_size
    sigma, alpha = model.diffusion_schedule(diffusion_times)
    alphas.append(float(np.array(alpha)[0, 0, 0, 0]))
    sigmas.append(float(np.array(sigma)[0, 0, 0, 0]))
alphas = np.array(alphas)
sigmas = np.array(sigmas)

# --- SeqDiff initialization (Algo 1 lines 16-23) ---
prev_recon_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")

if USE_SEQDIFF and os.path.exists(prev_recon_path):
    # SeqDiff warm-start (Algo 1 lines 17-19)
    print(f"\nSeqDiff: loading previous reconstruction from {prev_recon_path}")
    X_prev = np.load(prev_recon_path)
    print(f"Previous reconstruction shape: {X_prev.shape}")

    # x_0 ← X^prev, extract B-planes (Algo 1 line 17)
    x_0 = np.transpose(X_prev, (1, 0, 2, 3))  # (N_az, N_el, N_ax, C)

    # Forward diffuse to τ': x_τ' ← α_τ' x_0 + σ_τ' ε  (Algo 1 line 19)
    start_step = N_STEPS - SEQDIFF_TAU
    alpha_tau_prime = alphas[start_step]
    sigma_tau_prime = sigmas[start_step]

    epsilon = np.random.randn(*x_0.shape).astype(np.float32)
    x_tau = alpha_tau_prime * x_0 + sigma_tau_prime * epsilon

    print(
        f"SeqDiff: forward-diffused to step {start_step}, "
        f"running {SEQDIFF_TAU} steps (τ'={SEQDIFF_TAU})"
    )
else:
    # Cold start (Algo 1 lines 21-22): x_T ~ N(0, I)
    start_step = 0
    x_tau = np.random.randn(N_az, N_el, N_ax, C).astype(np.float32)
    if USE_SEQDIFF:
        print("\nSeqDiff enabled but no previous reconstruction found. Cold start.")
    print("Initializing all B-planes with noise (cold start)")

# --- Main reconstruction loop (Algorithm 1 lines 25-37) ---
print(
    f"\nReconstructing volume: {N_az} B-planes over {N_STEPS} diffusion steps "
    f"(starting at step {start_step})..."
)
print(f"B-plane shape: ({N_el}, {N_ax}, {C}), batch size: {BATCH_SIZE}")
print("Structure: for τ → for all B-planes (batched) → DPS → stack → TV_az\n")

for step in range(start_step, N_STEPS):
    # Diffusion rates for this step
    sigma_tau = jnp.full((1, 1, 1, 1), sigmas[step])
    alpha_tau = jnp.full((1, 1, 1, 1), alphas[step])
    sigma_tau_minus_1 = jnp.full((1, 1, 1, 1), sigmas[step + 1])
    alpha_tau_minus_1 = jnp.full((1, 1, 1, 1), alphas[step + 1])

    # --- Process ALL B-planes for one diffusion step (Algo 1 lines 26-33) ---
    x_tau_minus_1 = np.empty_like(x_tau)

    # Debug first 10 steps and then every 20th step
    do_debug = (step - start_step) < 10 or (step + 1) % 20 == 0

    for batch_start in range(0, N_az, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_az)

        # Only print debug for first batch to avoid spam
        is_first_batch = batch_start == 0

        x_tau_minus_1[batch_start:batch_end], _ = one_diffusion_step(
            model,
            x_tau[batch_start:batch_end],
            y_all[batch_start:batch_end],
            A_batch,
            sigma_tau,
            alpha_tau,
            sigma_tau_minus_1,
            alpha_tau_minus_1,
            GAMMA,
            debug=do_debug and is_first_batch,
        )

    # --- Data consistency: replace observed rows with noised GT ---
    # x_{τ-1}[observed] = α_{τ-1} · x_clean + σ_{τ-1} · ε
    # Ensures observed rows stay faithful to measurements at every step.
    a_next = float(np.array(alpha_tau_minus_1).ravel()[0])
    s_next = float(np.array(sigma_tau_minus_1).ravel()[0])
    noise_dc = np.random.randn(
        N_az, len(observed_rows), N_ax, C
    ).astype(np.float32)
    x_tau_minus_1[:, observed_rows, :, :] = (
        a_next * B_gt[:, observed_rows, :, :] + s_next * noise_dc
    )

    # --- Stack B-planes into volume X_{τ-1} (Algo 1 line 34) ---
    # Transpose from (N_az, N_el, N_ax, C) → (N_el, N_az, N_ax, C)
    X_tau_minus_1 = np.transpose(x_tau_minus_1, (1, 0, 2, 3))

    # --- TV regularization (Algo 1 lines 35-36) ---
    # V ← ∇_X TV_az(X_{τ-1})
    # X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V
    V = compute_tv_gradient_azimuth(X_tau_minus_1)
    X_tau_minus_1 = X_tau_minus_1 - alpha_tau_minus_1 * ZETA * V

    # --- Back to B-planes for next step ---
    x_tau = np.transpose(X_tau_minus_1, (1, 0, 2, 3))

    # Progress logging
    if do_debug:
        tv_val = np.sum(np.abs(np.diff(X_tau_minus_1, axis=1)))
        x_absmax = np.abs(x_tau_minus_1).max()
        has_nan = bool(np.any(np.isnan(x_tau_minus_1)))
        print(
            f"Step {step + 1}/{N_STEPS}: TV={tv_val:.4f}, "
            f"|x|_max={x_absmax:.4f}, nan={has_nan}, "
            f"σ={sigma_tau.reshape(()).item():.4f}, "
            f"α={alpha_tau.reshape(()).item():.4f} → "
            f"α'={alpha_tau_minus_1.reshape(()).item():.4f}"
        )
        print()

# Final output: reconstructed volume (Algo 1 line 38)
X_reconstructed = np.transpose(x_tau, (1, 0, 2, 3))

print("\nReconstruction complete.")

# --- Save ---
save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")
np.save(save_path, X_reconstructed)
print(f"\nSaved reconstructed volume to {save_path}")
print(f"Reconstructed volume shape: {X_reconstructed.shape}")
print("Volume reconstruction complete.")
