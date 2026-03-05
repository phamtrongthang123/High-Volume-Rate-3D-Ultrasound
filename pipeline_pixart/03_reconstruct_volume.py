"""
reconstruct_volume_pixart.py — DPS reconstruction using PixArt-α prior (PyTorch).

State is maintained in VAE latent space (N_az, 4, 64, 64) throughout the loop.
Measurement comparison happens in pixel space (112x112) after decoding the
Tweedie estimate. The DPS gradient is computed via autograd through the VAE
decode only (stopping gradient at the denoiser for memory efficiency).

Algorithm (latent-space DPS):
  Initialize x_τ ~ N(0, I) in latent space
  For τ = T to 1:
      For all B-planes j (batched):
          [no_grad] ε = ε_θ(x_τ, τ)
          [no_grad] x̂₀ = (x_τ - σ_τ ε) / α_τ    (Tweedie in latent space)
          [autograd] x̂₀_pixel = decode(x̂₀)        (112x112 grayscale)
          M = y - A x̂₀_pixel
          ∂||M||/∂x̂₀ = autograd → ∂||M||/∂x_τ ≈ (1/α_τ) * ∂||M||/∂x̂₀
          x_{τ-1} = α_{τ-1} x̂₀ + σ_{τ-1} ε        (DDIM in latent space)
          x_{τ-1} -= γ * grad_latents
      TV regularization in latent space (across azimuth)
  Decode final latents to pixel space

Output: outputs/reconstructed_volume_pixart.npy
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pixart_diffusion_model import PixArtLatentDiffusionModel

# --- Config ---
ACCEL_RATE = 4       # r, acceleration rate
N_STEPS = 200        # denoising steps (uniformly subsampled from 1000)
GAMMA = 1.0          # γ early (t > 700): baseline guidance strength
GAMMA_MID = 2.0      # γ mid (400 < t <= 700): moderate boost
GAMMA_LATE = 8.0     # γ late (200 < t <= 400): strong boost (gradient is small ~0.06)
GAMMA_VL = 12.0      # γ very late (t <= 200): push ||M||₂ to 0 in final steps
GUIDANCE_CLIP = 0.3  # max per-element guidance for early/mid steps
GUIDANCE_CLIP_LATE = 0.8  # relaxed clip for late steps (gradient naturally small)
LATENT_CLIP = 8.0    # clip latents to [-LATENT_CLIP, LATENT_CLIP] after each step
ZETA = 0.001         # ζ, azimuth TV (between B-planes)
ZETA_EL = 0.020      # elevation TV (within each B-plane, along H axis of latent)
BATCH_SIZE = 4       # B-planes per batch (smaller for gradient memory)
EFFECTIVE_ALPHA_MIN = 0.15  # clamp alpha_t from below to prevent 1/α explosion
USE_SEQDIFF = True   # Enable SeqDiff warm-start
SEQDIFF_TAU = 50     # τ', warm-start diffusion step
FULL_GRAD = False    # Enable true DPS gradient through transformer (needs more GPU RAM)

# Default paths
_DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PRETRAINED_PATH = "PixArt-alpha/PixArt-XL-2-512x512"
LORA_PATH = os.path.join(PROJECT_DIR, "training", "checkpoints", "cetus_pixart_lora", "best-checkpoint", "transformer_lora")
VAE_DECODER_PATH = os.path.join(PROJECT_DIR, "training", "checkpoints", "vae_decoder_finetuned", "vae_decoder.pt")

# Parse CLI args
_parser = argparse.ArgumentParser(description="DPS reconstruction with PixArt-α prior")
_parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                     help="Directory containing pseudo_volume.npy and for saving outputs")
_parser.add_argument("--full-grad", action="store_true",
                     help="Enable true DPS: propagate gradient through transformer + VAE "
                          "(more accurate but ~3x more GPU memory; use with BATCH_SIZE=1)")
_args = _parser.parse_args()
OUTPUT_DIR = _args.output_dir
if _args.full_grad:
    FULL_GRAD = True
    BATCH_SIZE = 1  # Reduce batch size to fit gradient through transformer in memory
    print("Full-gradient DPS enabled (gradient through transformer + VAE). BATCH_SIZE=1.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_tv_gradient_latent(X_latent):
    """TV gradient along azimuth (axis 0) for latent arrays.

    Args:
        X_latent: (N_az, C, H, W) numpy array, latents.

    Returns:
        V: TV gradient of same shape.
    """
    diff = np.diff(X_latent, axis=0)  # (N_az-1, C, H, W)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    div = np.zeros_like(X_latent)
    div[:-1] += normalized
    div[1:] -= normalized

    return -div


def compute_tv_gradient_elevation_latent(X_latent):
    """TV gradient along elevation (axis 2 = H) within each B-plane latent.

    This smooths the elevation profile inside each B-plane's latent image,
    helping to fill in unobserved elevation rows between measurements.

    Args:
        X_latent: (N_az, C, H, W) numpy array, latents.

    Returns:
        V: TV gradient of same shape.
    """
    diff = np.diff(X_latent, axis=2)  # (N_az, C, H-1, W)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    div = np.zeros_like(X_latent)
    div[:, :, :-1, :] += normalized
    div[:, :, 1:, :] -= normalized

    return -div


def get_effective_gamma(timestep):
    """Dynamic GAMMA schedule: larger guidance in late steps where gradient is small.

    Early steps (t > 700): small gamma to stay stable while latents are noisy.
    Mid steps (400 < t <= 700): moderate boost.
    Late steps (t <= 400): strong boost — denoiser settled, gradient drops to 0.02-0.06.
    """
    if timestep > 700:
        return GAMMA, GUIDANCE_CLIP
    elif timestep > 400:
        return GAMMA_MID, GUIDANCE_CLIP
    elif timestep > 200:
        return GAMMA_LATE, GUIDANCE_CLIP_LATE
    else:
        return GAMMA_VL, GUIDANCE_CLIP_LATE


def one_diffusion_step_latent(
    model,
    x_tau_lat,
    y_np,
    A_np,
    alpha_t,
    sigma_t,
    alpha_t_m1,
    sigma_t_m1,
    gamma,
    guidance_clip,
    timestep,
    debug=False,
    full_grad=False,
):
    """One DPS step in latent space.

    Args:
        model: PixArtLatentDiffusionModel.
        x_tau_lat: (B, 4, 64, 64) numpy array, latents at τ.
        y_np: (B, 112, 112, 1) numpy array, measurements.
        A_np: (1, 112, 112, 1) numpy array, binary mask.
        alpha_t, sigma_t: signal/noise rates at τ from DDPM schedule.
        alpha_t_m1, sigma_t_m1: rates at τ-1.
        gamma: DPS guidance strength.
        timestep: discrete DDPM timestep.
        debug: print diagnostics.
        full_grad: if True, propagate gradient through transformer+VAE (true DPS);
                   if False (default), stop gradient at denoiser (approximate DPS).

    Returns:
        x_tau_m1_lat: (B, 4, 64, 64) numpy array, latents at τ-1.
        x_0_pixel: (B, 112, 112, 1) numpy array, Tweedie estimate in pixel space.
    """
    latents = torch.from_numpy(x_tau_lat).float().to(model.device)
    eff_alpha = max(alpha_t, EFFECTIVE_ALPHA_MIN)

    if full_grad:
        # True DPS: gradient flows through transformer + VAE.
        # ∂||M|| / ∂x_τ computed exactly (not just the VAE-only approximation).
        latents.requires_grad_(True)
        # Predict noise (gradient enabled — no torch.no_grad wrapper)
        noise_pred = model.predict_noise_latent(latents, timestep)
        x_0_latent = (latents - sigma_t * noise_pred) / eff_alpha

        # Decode to pixel space (gradient flows through VAE)
        decoded = model.vae.decode(
            x_0_latent.to(dtype=model.weight_dtype) / model.vae_scale_factor
        ).sample.float()

        decoded_gray = decoded.mean(dim=1, keepdim=True)
        x_0_pixel = Func.interpolate(
            decoded_gray, size=(112, 112), mode="bilinear", align_corners=False
        )
        x_0_pixel = x_0_pixel.permute(0, 2, 3, 1)

        y = torch.from_numpy(y_np).float().to(model.device)
        A = torch.from_numpy(A_np).float().to(model.device)
        M = y - A * x_0_pixel
        M_norm = M.norm() + 1e-8

        # Full DPS gradient: ∂||M|| / ∂x_τ (through transformer + VAE)
        grad_latents = torch.autograd.grad(M_norm, latents)[0]

        x_0_latent_d = x_0_latent.detach()
        noise_pred_d = noise_pred.detach()
    else:
        # Approximate DPS: stop gradient at denoiser (memory-efficient).
        # Step 1: Predict noise and compute Tweedie estimate (no grad through denoiser)
        with torch.no_grad():
            noise_pred = model.predict_noise_latent(latents, timestep)
            # Tweedie in latent space: x̂₀ = (x_τ - σ·ε) / α
            x_0_latent = (latents - sigma_t * noise_pred) / eff_alpha

        # Step 2: Compute DPS gradient through VAE decode only
        x_0_lat_grad = x_0_latent.detach().requires_grad_(True)

        decoded = model.vae.decode(
            x_0_lat_grad.to(dtype=model.weight_dtype) / model.vae_scale_factor
        ).sample.float()  # (B, 3, 512, 512)

        decoded_gray = decoded.mean(dim=1, keepdim=True)
        x_0_pixel = Func.interpolate(
            decoded_gray, size=(112, 112), mode="bilinear", align_corners=False
        )
        x_0_pixel = x_0_pixel.permute(0, 2, 3, 1)

        y = torch.from_numpy(y_np).float().to(model.device)
        A = torch.from_numpy(A_np).float().to(model.device)
        M = y - A * x_0_pixel
        M_norm = M.norm() + 1e-8

        # DPS gradient: ∂||M|| / ∂x̂₀_latent, chain-ruled to ∂||M|| / ∂x_τ
        grad_x0 = torch.autograd.grad(M_norm, x_0_lat_grad)[0]
        grad_latents = grad_x0 / eff_alpha

        x_0_latent_d = x_0_latent
        noise_pred_d = noise_pred

    # DDIM step in latent space
    with torch.no_grad():
        x_tau_m1 = alpha_t_m1 * x_0_latent_d + sigma_t_m1 * noise_pred_d
        # Apply DPS correction: clip per-element to avoid destabilizing steps.
        raw_guidance = gamma * grad_latents.detach()
        guidance = torch.clamp(raw_guidance, -guidance_clip, guidance_clip)
        x_tau_m1 = x_tau_m1 - guidance

    if debug:
        print(f"  latents: min={x_tau_lat.min():+.3f} max={x_tau_lat.max():+.3f}")
        print(f"  x_0_pixel: min={x_0_pixel.min().item():+.3f} max={x_0_pixel.max().item():+.3f}")
        print(f"  ||M||₂: {M_norm.item():.3f}")
        print(f"  grad_latents absmax: {grad_latents.abs().max().item():.6f}")
        frac_clipped = (raw_guidance.abs() > guidance_clip).float().mean().item()
        print(f"  guidance absmax: {guidance.abs().max().item():.6f} "
              f"(clipped {frac_clipped*100:.0f}%, clip={guidance_clip})")
        print(f"  x_tau_m1: min={x_tau_m1.min().item():+.3f} max={x_tau_m1.max().item():+.3f}")

    return x_tau_m1.cpu().detach().numpy(), x_0_pixel.detach().cpu().numpy()


def decode_latents_to_volume(model, latents_np, batch_size=8):
    """Decode all latents to 112x112 pixel space.

    Args:
        latents_np: (N, 4, 64, 64) numpy array.
        batch_size: decode this many at a time.

    Returns:
        images: (N, 112, 112, 1) numpy array in [-1, 1].
    """
    N = latents_np.shape[0]
    results = []
    for i in range(0, N, batch_size):
        batch = torch.from_numpy(latents_np[i:i+batch_size]).to(model.device)
        with torch.no_grad():
            decoded = model.vae.decode(
                batch.to(dtype=model.weight_dtype) / model.vae_scale_factor
            ).sample.float()
        gray = decoded.mean(dim=1, keepdim=True)
        gray = Func.interpolate(gray, size=(112, 112), mode="bilinear", align_corners=False)
        gray = gray.permute(0, 2, 3, 1).cpu().numpy()
        results.append(gray)
    return np.concatenate(results, axis=0)


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
        sys.exit(1)

    X_gt = np.load(volume_path)
    N_el, N_az, N_ax, C = X_gt.shape
    print(f"Loaded ground truth volume: {X_gt.shape}")
    assert (N_el, N_ax, C) == (112, 112, 1), f"Unexpected shape: {X_gt.shape}"

    # --- Elevation subsampling ---
    observed_rows = list(range(0, N_el, ACCEL_RATE))
    print(f"\nAcceleration rate: {ACCEL_RATE}x")
    print(f"Observed rows ({len(observed_rows)}): {observed_rows[:8]}...")

    # Measurement operator A (pixel space)
    A = np.zeros((N_el, N_ax, C), dtype=np.float32)
    A[observed_rows] = 1.0
    print(f"Operator A shape: {A.shape}, observed fraction: {A.mean():.2f}")

    # B-planes in pixel space: (N_az, N_el, N_ax, 1)
    B_gt = np.transpose(X_gt, (1, 0, 2, 3))
    y_all = B_gt * A[np.newaxis]   # (N_az, N_el, N_ax, 1)
    A_batch = A[np.newaxis]         # (1, N_el, N_ax, 1)

    # --- Build DDPM timestep schedule ---
    # Sample N_STEPS timesteps uniformly from [T-1, 0]
    T = model.num_train_timesteps  # 1000
    step_ratio = T // N_STEPS  # 5 for 200 steps
    # Decreasing: [999, 994, 989, ..., 4]
    timesteps_list = list(range(T - 1, -1, -step_ratio))[:N_STEPS]
    # Next timestep (for α_{τ-1}, σ_{τ-1})
    next_timesteps = timesteps_list[1:] + [-1]

    print(f"Timestep schedule: {timesteps_list[0]} → {timesteps_list[-1]} "
          f"({len(timesteps_list)} steps)")

    # --- Initialize in latent space ---
    prev_recon_path = os.path.join(OUTPUT_DIR, "reconstructed_volume_pixart.npy")

    if USE_SEQDIFF and os.path.exists(prev_recon_path):
        print(f"\nSeqDiff: loading previous reconstruction from {prev_recon_path}")
        X_prev = np.load(prev_recon_path)
        x_0_pixel = np.transpose(X_prev, (1, 0, 2, 3))  # (N_az, N_el, N_ax, 1)
        # Encode to latent
        print("Encoding previous reconstruction to latent space...")
        x_0_latent = []
        for i in range(0, N_az, BATCH_SIZE):
            batch = x_0_pixel[i:i+BATCH_SIZE]
            lat = model.encode_images(batch)
            x_0_latent.append(lat.cpu().numpy())
        x_0_latent = np.concatenate(x_0_latent, axis=0)  # (N_az, 4, 64, 64)

        # Forward diffuse to warm-start step
        warm_idx = N_STEPS - SEQDIFF_TAU
        ts_warm = timesteps_list[warm_idx]
        ab_warm = float(model.noise_scheduler.alphas_cumprod[ts_warm])
        alpha_warm = math.sqrt(ab_warm)
        sigma_warm = math.sqrt(1 - ab_warm)
        epsilon = np.random.randn(*x_0_latent.shape).astype(np.float32)
        x_tau_latent = alpha_warm * x_0_latent + sigma_warm * epsilon
        timesteps_list = timesteps_list[warm_idx:]
        next_timesteps = next_timesteps[warm_idx:]
        print(f"SeqDiff: forward-diffused to timestep {ts_warm}, "
              f"running {SEQDIFF_TAU} denoising steps")
    else:
        # Cold start: pure Gaussian noise in latent space
        x_tau_latent = np.random.randn(N_az, 4, 64, 64).astype(np.float32)
        print("Initialized latents with Gaussian noise (cold start)")

    print(f"\nRunning {len(timesteps_list)} denoising steps, "
          f"{N_az} B-planes, batch={BATCH_SIZE}")
    print(f"Latent shape per B-plane: (4, 64, 64)")
    print(f"GAMMA={GAMMA}/{GAMMA_MID}/{GAMMA_LATE}/{GAMMA_VL} (early/mid/late/very-late), "
          f"GUIDANCE_CLIP={GUIDANCE_CLIP}/{GUIDANCE_CLIP_LATE}")
    print(f"ZETA={ZETA} (azimuth), ZETA_EL={ZETA_EL} (elevation), "
          f"LATENT_CLIP={LATENT_CLIP}, EFFECTIVE_ALPHA_MIN={EFFECTIVE_ALPHA_MIN}\n")

    # --- Main reconstruction loop ---
    alphas_cumprod = model.noise_scheduler.alphas_cumprod  # tensor of length T

    for i, (timestep, next_ts) in enumerate(zip(timesteps_list, next_timesteps)):
        # Get α, σ at current and next timestep
        ab_t = float(alphas_cumprod[timestep])
        alpha_t = math.sqrt(ab_t)
        sigma_t = math.sqrt(max(1.0 - ab_t, 0.0))

        if next_ts >= 0:
            ab_t_m1 = float(alphas_cumprod[next_ts])
        else:
            ab_t_m1 = 1.0
        alpha_t_m1 = math.sqrt(ab_t_m1)
        sigma_t_m1 = math.sqrt(max(1.0 - ab_t_m1, 0.0))

        x_tau_m1_latent = np.empty_like(x_tau_latent)
        do_debug = i < 10 or (i + 1) % 20 == 0

        gamma_eff, gc_eff = get_effective_gamma(timestep)

        for batch_start in range(0, N_az, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N_az)
            is_first = batch_start == 0

            x_tau_m1_latent[batch_start:batch_end], _ = one_diffusion_step_latent(
                model,
                x_tau_latent[batch_start:batch_end],
                y_all[batch_start:batch_end],
                A_batch,
                alpha_t, sigma_t,
                alpha_t_m1, sigma_t_m1,
                gamma_eff, gc_eff, timestep,
                debug=do_debug and is_first,
                full_grad=FULL_GRAD,
            )

        # Azimuth TV regularization (between B-planes)
        V_az = compute_tv_gradient_latent(x_tau_m1_latent)
        x_tau_latent = x_tau_m1_latent - alpha_t_m1 * ZETA * V_az

        # Elevation TV regularization (within each B-plane, along H axis)
        V_el = compute_tv_gradient_elevation_latent(x_tau_latent)
        x_tau_latent = x_tau_latent - alpha_t_m1 * ZETA_EL * V_el

        # Clip latents to prevent drift from VAE training distribution
        x_tau_latent = np.clip(x_tau_latent, -LATENT_CLIP, LATENT_CLIP)

        if do_debug:
            lat_absmax = np.abs(x_tau_latent).max()
            has_nan = bool(np.any(np.isnan(x_tau_latent)))
            tv_az = np.sum(np.abs(np.diff(x_tau_latent, axis=0)))
            tv_el = np.sum(np.abs(np.diff(x_tau_latent, axis=2)))
            print(
                f"Step {i+1}/{len(timesteps_list)}: t={timestep}, γ={gamma_eff}, "
                f"TV_az={tv_az:.2f}, TV_el={tv_el:.2f}, "
                f"|lat|_max={lat_absmax:.3f}, nan={has_nan}, "
                f"α={alpha_t:.4f}→{alpha_t_m1:.4f}"
            )
            print()

    # --- Final decode: latents → pixel space ---
    print("\nDecoding final latents to pixel space...")
    X_B_planes = decode_latents_to_volume(model, x_tau_latent, batch_size=BATCH_SIZE)
    # X_B_planes: (N_az, N_el, N_ax, 1)
    X_reconstructed = np.transpose(X_B_planes, (1, 0, 2, 3))  # (N_el, N_az, N_ax, 1)

    # Edge plane post-processing: blend planes beyond last observed elevation
    # toward the last observed plane. Planes 109, 110, 111 have no observed
    # measurement signal (last observed is 108); blending prevents the diffusion
    # prior from dominating with unrelated content at the volume boundary.
    last_obs = observed_rows[-1]  # e.g. 108
    EDGE_BLEND = 0.7  # weight for the last observed plane per step away
    for delta in range(1, N_el - last_obs):
        el = last_obs + delta
        w = EDGE_BLEND ** delta  # 0.5, 0.25, 0.125
        X_reconstructed[el] = w * X_reconstructed[last_obs] + (1 - w) * X_reconstructed[el]
        print(f"  Edge blend: plane {el} ← {w:.3f}*plane{last_obs} + {1-w:.3f}*decoded")

    print("Reconstruction complete.")
    save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume_pixart.npy")
    np.save(save_path, X_reconstructed)
    print(f"Saved: {save_path}")
    print(f"Shape: {X_reconstructed.shape}")
    print(f"Range: [{X_reconstructed.min():.3f}, {X_reconstructed.max():.3f}]")


if __name__ == "__main__":
    main()
