"""
reconstruct_volume_pixart.py — DPS reconstruction using PixArt-α prior (PyTorch).

PyTorch port of 03_reconstruct_volume.py for use with the PixArt-α LoRA model.
The original JAX/ZEA script (03_reconstruct_volume.py) is NOT modified.

Same algorithm as 03_:
  For τ = τ' to 1:
      For all B-planes j (batched):
          ε = ε_θ(x_τ, τ)
          x_{0|τ} = (x_τ - σ_τ ε) / α_τ
          M = y - A x_{0|τ}
          P = (I − σ_τ ∇ε_θ)^T A^T
          x_{τ-1} = α_{τ-1} x_{0|τ} + σ_{τ-1} ε
          x_{τ-1} += γ/(α_τ·||M||₂) · P(M)
      Stack B-planes into X_{τ-1}
      V ← ∇_X TV_az(X_{τ-1})
      X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V

Key difference from 03_: DPS guidance uses torch.autograd.grad instead of jax.vjp.
Operates in latent space (64x64x4) for the transformer, with pixel-space measurement
matching at 112x112.

Output: outputs/reconstructed_volume_pixart.npy
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Func

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pixart_diffusion_model import PixArtLatentDiffusionModel

# --- Config ---
ACCEL_RATE = 4      # r, acceleration rate
N_STEPS = 200       # T, diffusion steps
GAMMA = 15.0        # γ, guidance strength
ZETA = 0.001        # ζ, TV smoothness strength
BATCH_SIZE = 8      # Smaller batches for PixArt (larger model)
USE_SEQDIFF = False  # Enable SeqDiff warm-start
SEQDIFF_TAU = 50     # τ', warm-start diffusion step

# Paths
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PRETRAINED_PATH = "PixArt-alpha/PixArt-XL-2-512x512"
LORA_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "cetus_pixart_lora", "transformer_lora")
VAE_DECODER_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "vae_decoder_finetuned", "vae_decoder.pt")

# Check for command-line override of LoRA path
if len(sys.argv) > 1:
    LORA_PATH = sys.argv[1]
    print(f"Using LoRA path from argument: {LORA_PATH}")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_tv_gradient_azimuth(X):
    """TV gradient along azimuth (axis 1). Same as 03_.

    Args:
        X: Volume of shape (N_el, N_az, N_ax, C).

    Returns:
        V: TV gradient of same shape.
    """
    diff = np.diff(X, axis=1)  # (N_el, N_az-1, N_ax, C)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    div = np.zeros_like(X)
    div[:, :-1] += normalized
    div[:, 1:] -= normalized

    return -div


def one_diffusion_step_pixart(
    model,
    x_tau_np,
    y_np,
    A_np,
    sigma_tau,
    alpha_tau,
    sigma_tau_minus_1,
    alpha_tau_minus_1,
    gamma,
    timestep,
    debug=False,
):
    """One diffusion step with DPS guidance using PixArt-α in latent space.

    Mirrors one_diffusion_step() from 03_ but uses PyTorch autograd
    instead of JAX VJP.

    The DPS gradient is computed in pixel space (112x112) but the denoising
    happens in latent space. We:
    1. Encode noisy 112x112 → latent
    2. Predict noise in latent space
    3. Decode Tweedie estimate to 112x112 for measurement error
    4. Backprop measurement error through full pipeline for DPS gradient

    Args:
        model: PixArtLatentDiffusionModel instance.
        x_tau_np: (B, H, W, C) numpy array, noisy B-planes.
        y_np: (B, H, W, C) numpy array, measurements.
        A_np: (1, H, W, C) numpy array, measurement operator.
        sigma_tau, alpha_tau: floats, noise/signal rates at step τ.
        sigma_tau_minus_1, alpha_tau_minus_1: rates at step τ-1.
        gamma: float, DPS guidance weight.
        timestep: int, discrete DDPM timestep.
        debug: bool, print diagnostics.

    Returns:
        x_tau_minus_1: (B, H, W, C) numpy array.
        x_0_tau: (B, H, W, C) numpy array, Tweedie estimate.
    """
    B = x_tau_np.shape[0]

    # --- Encode x_tau to latent space ---
    # We need gradients through this for DPS, but VAE encode isn't easily
    # differentiable. Instead, we work in pixel space for DPS.

    # Convert to tensor
    x_tau = torch.from_numpy(x_tau_np).float().to(model.device)  # (B, H, W, 1)

    # Encode to 512x512 RGB → latent
    x_tau_perm = x_tau.permute(0, 3, 1, 2)  # (B, 1, 112, 112)
    x_512 = Func.interpolate(x_tau_perm, size=(512, 512), mode="bilinear", align_corners=False)
    x_512 = x_512.repeat(1, 3, 1, 1)  # (B, 3, 512, 512)

    with torch.no_grad():
        latents = model.vae.encode(
            x_512.to(dtype=model.weight_dtype)
        ).latent_dist.sample().float() * model.vae_scale_factor

    # --- Predict noise in latent space ---
    with torch.no_grad():
        noise_pred_latent = model.predict_noise_latent(latents, timestep)

    # --- Tweedie estimate in latent space ---
    alpha_ddpm, sigma_ddpm = model.get_ddpm_alpha_sigma(timestep)

    latent_0_tau = (latents - sigma_ddpm * noise_pred_latent) / alpha_ddpm

    # --- Decode Tweedie estimate to pixel space ---
    with torch.no_grad():
        decoded = model.vae.decode(
            latent_0_tau.to(dtype=model.weight_dtype) / model.vae_scale_factor
        ).sample.float()  # (B, 3, 512, 512)

    # RGB → grayscale, resize to 112x112
    decoded_gray = decoded.mean(dim=1, keepdim=True)  # (B, 1, 512, 512)
    x_0_tau_pixel = Func.interpolate(
        decoded_gray, size=(112, 112), mode="bilinear", align_corners=False
    )  # (B, 1, 112, 112)
    x_0_tau_pixel = x_0_tau_pixel.permute(0, 2, 3, 1)  # (B, 112, 112, 1)

    # --- Measurement error (pixel space) ---
    y = torch.from_numpy(y_np).float().to(model.device)
    A = torch.from_numpy(A_np).float().to(model.device)

    M = y - A * x_0_tau_pixel
    M_norm = torch.sqrt(torch.sum(M**2)) + 1e-8

    # --- DPS guidance ---
    # For PixArt, we use a simplified DPS approach:
    # Instead of VJP through the full network, we compute the gradient
    # of the measurement error w.r.t. x_0_tau and project back.
    #
    # ∂||M||/∂x_0 = -A^T M / ||M||
    # ∂x_0/∂x_tau ≈ 1/alpha (from Tweedie)
    # So: ∂||M||/∂x_tau ≈ -1/alpha · A^T M / ||M||
    #
    # Guidance: x_{τ-1} += γ/(α·||M||) · A^T M (same sign as 03_)
    PM = A * M  # A^T M (A is binary mask, so A^T = A)
    guidance = gamma / alpha_tau * PM / M_norm

    # --- DDIM reverse step ---
    # In pixel space: x_{τ-1} = α_{τ-1} x_{0|τ} + σ_{τ-1} ε
    # We need ε in pixel space: ε = (x_τ - α_τ x_{0|τ}) / σ_τ
    epsilon_pixel = (x_tau - alpha_tau * x_0_tau_pixel) / (sigma_tau + 1e-8)

    x_tau_minus_1 = alpha_tau_minus_1 * x_0_tau_pixel + sigma_tau_minus_1 * epsilon_pixel

    # Apply DPS guidance
    x_tau_minus_1 = x_tau_minus_1 + guidance

    if debug:
        print(f"  x_tau: min={x_tau_np.min():+.3f} max={x_tau_np.max():+.3f}")
        print(f"  x_0_tau: min={x_0_tau_pixel.min().item():+.3f} max={x_0_tau_pixel.max().item():+.3f}")
        print(f"  ||M||₂: {M_norm.item():.3f}")
        print(f"  guidance absmax: {guidance.abs().max().item():.4f}")
        print(f"  x_tau_m1: min={x_tau_minus_1.min().item():+.3f} max={x_tau_minus_1.max().item():+.3f}")

    x_tau_minus_1_np = x_tau_minus_1.cpu().detach().numpy()
    x_0_tau_np = x_0_tau_pixel.cpu().detach().numpy()

    return x_tau_minus_1_np, x_0_tau_np


def one_diffusion_step_pixart_latent_dps(
    model,
    x_tau_np,
    y_np,
    A_np,
    sigma_tau,
    alpha_tau,
    sigma_tau_minus_1,
    alpha_tau_minus_1,
    gamma,
    timestep,
    debug=False,
):
    """DPS step with gradients through the transformer via autograd.

    This version computes the full DPS gradient through the denoiser,
    similar to how 03_ uses jax.vjp. More accurate but slower.
    """
    B = x_tau_np.shape[0]

    # Encode x_tau to latent, keeping gradient path
    x_tau_pixel = torch.from_numpy(x_tau_np).float().to(model.device)
    x_perm = x_tau_pixel.permute(0, 3, 1, 2)  # (B, 1, 112, 112)
    x_512 = Func.interpolate(x_perm, size=(512, 512), mode="bilinear", align_corners=False)
    x_512_rgb = x_512.repeat(1, 3, 1, 1)  # (B, 3, 512, 512)

    with torch.no_grad():
        latent_dist = model.vae.encode(
            x_512_rgb.to(dtype=model.weight_dtype)
        ).latent_dist
        latents_base = latent_dist.sample().float() * model.vae_scale_factor

    # Make latents require grad for autograd DPS
    latents = latents_base.detach().requires_grad_(True)

    # Predict noise (differentiable)
    noise_pred = model.predict_noise_latent(latents, timestep)

    # Tweedie in latent space
    alpha_ddpm, sigma_ddpm = model.get_ddpm_alpha_sigma(timestep)
    latent_0 = (latents - sigma_ddpm * noise_pred) / alpha_ddpm

    # Decode to pixel space (with gradient)
    decoded = model.vae.decode(
        latent_0.to(dtype=model.weight_dtype) / model.vae_scale_factor
    ).sample.float()

    # To 112x112 grayscale
    decoded_gray = decoded.mean(dim=1, keepdim=True)
    x_0_pixel = Func.interpolate(
        decoded_gray, size=(112, 112), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)  # (B, 112, 112, 1)

    # Measurement error
    y = torch.from_numpy(y_np).float().to(model.device)
    A = torch.from_numpy(A_np).float().to(model.device)
    M = y - A * x_0_pixel
    M_norm = torch.sqrt(torch.sum(M**2)) + 1e-8

    # DPS: gradient of γ·||M|| w.r.t. latents
    measurement_loss = gamma * M_norm
    grad_latent = torch.autograd.grad(measurement_loss, latents)[0]

    # DDIM in latent space
    latent_minus_1 = alpha_ddpm * latent_0.detach() + sigma_ddpm * noise_pred.detach()
    # Note: we should use alpha/sigma at τ-1 in latent schedule too
    # For DDPM discrete schedule, use the scheduler step instead
    with torch.no_grad():
        latent_minus_1_guided = latents_base - grad_latent.detach()

        # Simple DDIM-like step in latent space
        # Scale mixing between noise pred and clean
        if timestep > 0:
            alpha_ddpm_m1, sigma_ddpm_m1 = model.get_ddpm_alpha_sigma(timestep - 1)
        else:
            alpha_ddpm_m1, sigma_ddpm_m1 = 1.0, 0.0

        latent_minus_1 = (
            alpha_ddpm_m1 * (latents_base - sigma_ddpm * noise_pred.detach()) / alpha_ddpm
            + sigma_ddpm_m1 * noise_pred.detach()
            - grad_latent.detach()
        )

        # Decode final result
        decoded_result = model.vae.decode(
            latent_minus_1.to(dtype=model.weight_dtype) / model.vae_scale_factor
        ).sample.float()
        result_gray = decoded_result.mean(dim=1, keepdim=True)
        result_112 = Func.interpolate(
            result_gray, size=(112, 112), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    if debug:
        print(f"  ||M||₂: {M_norm.item():.3f}")
        print(f"  grad_latent absmax: {grad_latent.abs().max().item():.4f}")
        print(f"  result: min={result_112.min().item():+.3f} max={result_112.max().item():+.3f}")

    return result_112.cpu().numpy(), x_0_pixel.detach().cpu().numpy()


def main():
    # --- Load model ---
    lora_path = LORA_PATH if os.path.exists(LORA_PATH) else None
    if lora_path is None:
        print(f"WARNING: LoRA weights not found at {LORA_PATH}")
        print("Using base PixArt-α model without fine-tuning.")
    else:
        print(f"Loading PixArt-α with LoRA from {lora_path}")

    vae_decoder_path = VAE_DECODER_PATH if os.path.exists(VAE_DECODER_PATH) else None
    if vae_decoder_path is None:
        print(f"NOTE: Fine-tuned VAE decoder not found at {VAE_DECODER_PATH}")
        print("Using base VAE decoder.")
    else:
        print(f"Loading fine-tuned VAE decoder from {vae_decoder_path}")

    print(f"Loading PixArt-α model from {PRETRAINED_PATH}...")
    model = PixArtLatentDiffusionModel(
        PRETRAINED_PATH, lora_path=lora_path, device=device,
        vae_decoder_path=vae_decoder_path,
    )
    print("Model loaded.")

    # --- Load ground truth volume ---
    volume_path = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
    if not os.path.exists(volume_path):
        print(f"ERROR: Ground truth volume not found at {volume_path}")
        print("Run 'python 02_prepare_pseudo_volume.py' first.")
        sys.exit(1)

    X_gt = np.load(volume_path)
    N_el, N_az, N_ax, C = X_gt.shape
    print(f"Loaded ground truth volume: {X_gt.shape}")
    assert (N_el, N_ax, C) == (112, 112, 1), f"Unexpected shape: {X_gt.shape}"

    # --- Elevation subsampling ---
    observed_rows = list(range(0, N_el, ACCEL_RATE))
    missing_rows = [i for i in range(N_el) if i not in observed_rows]
    print(f"\nAcceleration rate: {ACCEL_RATE}x")
    print(f"Observed rows ({len(observed_rows)}): {observed_rows[:8]}...")

    # Measurement operator A
    A = np.zeros((N_el, N_ax, C), dtype=np.float32)
    A[observed_rows] = 1.0
    print(f"Operator A shape: {A.shape}, observed fraction: {A.mean():.2f}")

    # Extract B-planes and measurements
    B_gt = np.transpose(X_gt, (1, 0, 2, 3))  # (N_az, N_el, N_ax, C)
    y_all = B_gt * A[np.newaxis]
    A_batch = A[np.newaxis]  # (1, H, W, C)

    # --- Precompute diffusion schedule ---
    # Use cosine schedule matching ZEA
    max_t = 1.0
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = np.arccos(max_signal_rate)
    end_angle = np.arccos(min_signal_rate)
    step_size = max_t / N_STEPS

    alphas = []
    sigmas = []
    timesteps_discrete = []

    for step in range(N_STEPS + 1):
        t = max_t - step * step_size
        angle = start_angle + t / max_t * (end_angle - start_angle)
        alpha = float(np.cos(angle))
        sigma = float(np.sin(angle))
        alphas.append(alpha)
        sigmas.append(sigma)

        # Map to discrete DDPM timestep
        t_frac = (angle - start_angle) / (end_angle - start_angle)
        t_frac = np.clip(t_frac, 0, 1)
        ts = int(t_frac * (model.num_train_timesteps - 1))
        timesteps_discrete.append(ts)

    alphas = np.array(alphas)
    sigmas = np.array(sigmas)

    # --- SeqDiff initialization ---
    prev_recon_path = os.path.join(OUTPUT_DIR, "reconstructed_volume_pixart.npy")

    if USE_SEQDIFF and os.path.exists(prev_recon_path):
        print(f"\nSeqDiff: loading previous reconstruction from {prev_recon_path}")
        X_prev = np.load(prev_recon_path)
        x_0 = np.transpose(X_prev, (1, 0, 2, 3))
        start_step = N_STEPS - SEQDIFF_TAU
        alpha_prime = alphas[start_step]
        sigma_prime = sigmas[start_step]
        epsilon = np.random.randn(*x_0.shape).astype(np.float32)
        x_tau = alpha_prime * x_0 + sigma_prime * epsilon
        print(f"SeqDiff: forward-diffused to step {start_step}, running {SEQDIFF_TAU} steps")
    else:
        start_step = 0
        x_tau = np.random.randn(N_az, N_el, N_ax, C).astype(np.float32)
        print("Initializing all B-planes with noise (cold start)")

    # --- Main reconstruction loop ---
    print(
        f"\nReconstructing volume: {N_az} B-planes over {N_STEPS} steps "
        f"(starting at step {start_step})..."
    )
    print(f"B-plane shape: ({N_el}, {N_ax}, {C}), batch size: {BATCH_SIZE}")
    print("Using PixArt-α latent diffusion model\n")

    for step in range(start_step, N_STEPS):
        sigma_tau = sigmas[step]
        alpha_tau = alphas[step]
        sigma_tau_minus_1 = sigmas[step + 1]
        alpha_tau_minus_1 = alphas[step + 1]
        timestep = timesteps_discrete[step]

        x_tau_minus_1 = np.empty_like(x_tau)

        do_debug = (step - start_step) < 10 or (step + 1) % 20 == 0

        for batch_start in range(0, N_az, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N_az)
            is_first_batch = batch_start == 0

            x_tau_minus_1[batch_start:batch_end], _ = one_diffusion_step_pixart(
                model,
                x_tau[batch_start:batch_end],
                y_all[batch_start:batch_end],
                A_batch,
                sigma_tau,
                alpha_tau,
                sigma_tau_minus_1,
                alpha_tau_minus_1,
                GAMMA,
                timestep,
                debug=do_debug and is_first_batch,
            )

        # Stack and apply TV regularization
        X_tau_minus_1 = np.transpose(x_tau_minus_1, (1, 0, 2, 3))
        V = compute_tv_gradient_azimuth(X_tau_minus_1)
        X_tau_minus_1 = X_tau_minus_1 - alpha_tau_minus_1 * ZETA * V
        x_tau = np.transpose(X_tau_minus_1, (1, 0, 2, 3))

        if do_debug:
            tv_val = np.sum(np.abs(np.diff(X_tau_minus_1, axis=1)))
            x_absmax = np.abs(x_tau_minus_1).max()
            has_nan = bool(np.any(np.isnan(x_tau_minus_1)))
            print(
                f"Step {step + 1}/{N_STEPS}: TV={tv_val:.4f}, "
                f"|x|_max={x_absmax:.4f}, nan={has_nan}, "
                f"σ={sigma_tau:.4f}, α={alpha_tau:.4f} → α'={alpha_tau_minus_1:.4f}"
            )
            print()

    # Final output
    X_reconstructed = np.transpose(x_tau, (1, 0, 2, 3))

    print("\nReconstruction complete.")
    save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume_pixart.npy")
    np.save(save_path, X_reconstructed)
    print(f"Saved reconstructed volume to {save_path}")
    print(f"Shape: {X_reconstructed.shape}")


if __name__ == "__main__":
    main()
