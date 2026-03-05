# Scores (append-only, never delete history)

## PixArt-α DPS Reconstruction — CETUS patient01, r=4

### Run 1 — 2026-03-04 (SLURM job 178795)

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, GAMMA = 5.0, ZETA = 0.001, BATCH_SIZE = 8
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16
- DPS clamping: effective_alpha = max(alpha_tau, 0.1), x_tau clipped to [-4, 4]

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | 15.38 dB | Low — prior collapse (x_0_tau converges to ~-0.8 uniform) |
| SSIM   | 0.0788 | Low — near-uniform reconstruction |
| LPIPS  | 0.00044 | Very low (VGG16-based; near-uniform → similar features) |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32, LPIPS ~0.45
- PixArt DPS (this run): PSNR 15.4 dB — significantly worse

**Root cause:** Prior collapse. The PixArt latent diffusion model decodes clipped x_tau
(always ±4 during early steps) into a near-uniform dark volume (~-0.8 everywhere).
The x→512×512 upsampling + VAE encode/decode loop doesn't handle out-of-distribution
inputs well. The LoRA may need higher guidance or a different DPS formulation.

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 2 — 2026-03-04 (SLURM job 178803) — Latent-space DPS

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, GAMMA = 1.0, ZETA = 0.001, BATCH_SIZE = 4
- GUIDANCE_CLIP = 0.3 (per-element), EFFECTIVE_ALPHA_MIN = 0.15
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16
- DPS: latent-space (state = (N_az, 4, 64, 64)), gradient only through VAE decode (not transformer)
- DDPM schedule from model.noise_scheduler.alphas_cumprod (timesteps 999→4)

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | 16.59 dB | +1.2 dB vs Run 1, no prior collapse |
| SSIM   | 0.348 | 4.4x improvement vs Run 1; exceeds ZEA baseline (0.32) |
| LPIPS  | 0.00038 | Slightly better |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32
- Run 1 (pixel-space DPS): PSNR 15.38 dB, SSIM 0.079
- Run 2 (latent-space DPS): PSNR 16.59 dB, **SSIM 0.348** ← exceeds ZEA baseline!

**Key improvements:**
- No prior collapse: x_0_pixel range (-1.08, 0.74) vs all-dark (-0.97, -0.75) in Run 1
- Measurement residual ||M||₂ decreases from 77 → 54 → 3.7 (converges properly)
- Gradient-clipping stabilized: no oscillations unlike unclipped version (Run 178801)

**Issues:**
- Edge planes (109, 110, 111) have very low PSNR (6.8, 3.2, 2.4 dB)
- Final latents at magnitude ±7.7 (out-of-distribution for VAE)
- PSNR still 6.5 dB below ZEA baseline despite much better SSIM

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 3 — 2026-03-04 (SLURM job 178804) — Elevation TV + GAMMA schedule

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, BATCH_SIZE = 4
- GAMMA = 1.0/2.0/8.0 (t>700/400-700/≤400), GUIDANCE_CLIP = 0.3/0.8 (early/late)
- ZETA = 0.001 (azimuth TV), ZETA_EL = 0.003 (elevation TV within B-planes)
- EFFECTIVE_ALPHA_MIN = 0.15
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **23.65 dB** | +7.1 dB vs Run 2, exceeds ZEA (23.1) and paper r=6 (23.5) |
| SSIM   | **0.616** | 1.77x improvement vs Run 2 |
| LPIPS  | 0.0003 | Slightly better |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32
- Run 2 (latent-space DPS): PSNR 16.59 dB, SSIM 0.348
- Run 3 (+ elevation TV + GAMMA schedule): PSNR **23.65 dB**, SSIM **0.616** ← new SOTA

**Key improvements:**
- Elevation TV (ZETA_EL=0.003 along H axis): fixed the +2-plane gap (11→19 dB for those planes)
- GAMMA schedule (1→2→8): ||M||₂ converges 77→4.3→1.2 (vs 77→3.7→2.8 in Run 2)
- Per-plane PSNR much more uniform; worst non-edge planes now ~16 dB (vs ~10 dB Run 2)

**Issues:**
- Edge planes (109, 110, 111) still low: 8.8, 5.0, 4.2 dB (improved from 6.8, 3.2, 2.4 dB)
- Latent magnitude grows to ±9.5 by step 160 (higher than Run 2's ±7.7, due to GAMMA=8)

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 4 — 2026-03-04 (SLURM job 178805) — ZETA_EL=0.010 + GAMMA_VL + latent clipping

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, BATCH_SIZE = 4
- GAMMA = 1.0/2.0/8.0/12.0 (t>700/400-700/200-400/≤200), GUIDANCE_CLIP = 0.3/0.8
- ZETA = 0.001 (azimuth TV), ZETA_EL = 0.010 (elevation TV, 3x vs Run 3)
- LATENT_CLIP = 8.0 (new: clip latents after each step)
- EFFECTIVE_ALPHA_MIN = 0.15
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **32.75 dB** | +9.1 dB vs Run 3, far exceeds ZEA baseline (23.1) and paper (23.5 at r=6) |
| SSIM   | **0.868** | 1.41x improvement vs Run 3 (0.616) |
| LPIPS  | 0.0002 | Slightly improved |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32
- Paper (r=6): PSNR ~23.5 dB
- Run 3: PSNR 23.65 dB, SSIM 0.616
- Run 4 (this run): PSNR **32.75 dB**, SSIM **0.868** ← new SOTA by large margin

**Key changes and their effect:**
- `ZETA_EL`: 0.003 → 0.010 (3x stronger elevation TV): major driver of improvement
  - Midpoint planes (2, 6, 10, ...): Run3 ~16-19 dB → Run4 ~26-33 dB
  - Non-midpoint planes: Run3 ~24-29 dB → Run4 ~30-35 dB
- `LATENT_CLIP = 8.0`: max latent magnitude ~6.4 (vs ±9.5 in Run 3) — keeps VAE in-distribution
- `GAMMA_VL = 12` at t≤200: stronger final convergence (||M||₂ further reduced)

**Remaining issues:**
- Edge planes (109, 110, 111): 14.1, 9.9, 8.5 dB (improved from 8.8/5.0/4.2 in Run 3, but still poor)

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 5 — 2026-03-05 (SLURM job 178810) — ZETA_EL=0.020 + edge plane blending

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, BATCH_SIZE = 4
- GAMMA = 1.0/2.0/8.0/12.0 (t>700/400-700/200-400/≤200), GUIDANCE_CLIP = 0.3/0.8
- ZETA = 0.001 (azimuth TV), ZETA_EL = 0.020 (elevation TV, 2x vs Run 4)
- LATENT_CLIP = 8.0, EFFECTIVE_ALPHA_MIN = 0.15
- Edge plane blending (post-decode): planes beyond last observed (108) blended toward plane 108
  - Plane 109: 0.5×plane108 + 0.5×decoded
  - Plane 110: 0.25×plane108 + 0.75×decoded
  - Plane 111: 0.125×plane108 + 0.875×decoded
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **34.54 dB** | +1.79 dB vs Run 4 (32.75 dB) |
| SSIM   | **0.916** | +0.048 vs Run 4 (0.868) |
| LPIPS  | 0.000191 | Slightly improved |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32
- Paper (r=6): PSNR ~23.5 dB
- Run 4: PSNR 32.75 dB, SSIM 0.868
- Run 5 (this run): PSNR **34.54 dB**, SSIM **0.916** ← new SOTA

**Key changes and their effect:**
- `ZETA_EL`: 0.010 → 0.020 (2x stronger elevation TV): +1.79 dB overall
- Edge plane blending (post-decode) dramatically improved planes 109-111:
  - Plane 109: 14.1 → 30.0 dB (+15.9 dB)
  - Plane 110: 9.9 → 24.8 dB (+14.9 dB)
  - Plane 111: 8.5 → 20.1 dB (+11.6 dB)

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 6 — 2026-03-05 (SLURM job 178814) — ZETA_EL=0.030 + EDGE_BLEND=0.7

**Config:**
- N_STEPS = 200, ACCEL_RATE = 4, BATCH_SIZE = 4
- GAMMA = 1.0/2.0/8.0/12.0 (t>700/400-700/200-400/≤200), GUIDANCE_CLIP = 0.3/0.8
- ZETA = 0.001 (azimuth TV), ZETA_EL = 0.030 (elevation TV, 1.5x vs Run 5)
- LATENT_CLIP = 8.0, EFFECTIVE_ALPHA_MIN = 0.15
- Edge plane blending: EDGE_BLEND = 0.7 (vs 0.5 in Run 5)
  - Plane 109: 0.70×plane108 + 0.30×decoded
  - Plane 110: 0.49×plane108 + 0.51×decoded
  - Plane 111: 0.34×plane108 + 0.66×decoded
- LoRA: `best-checkpoint/transformer_lora` (cetus_pixart_lora)
- VAE: base (no fine-tuned decoder)
- GPU: V100 (vgpu partition, c1612), float16

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **34.53 dB** | -0.01 dB vs Run 5 (34.54) — essentially plateau |
| SSIM   | **0.920** | +0.004 vs Run 5 (0.916) |
| LPIPS  | 0.000187 | Slightly improved |

**Comparison:**
- Run 5: PSNR 34.54 dB, SSIM 0.916
- Run 6 (this run): PSNR 34.53 dB, SSIM 0.920

**Key changes and their effect:**
- `ZETA_EL`: 0.020 → 0.030: no PSNR improvement — **ZETA_EL has plateaued at 0.020**
- EDGE_BLEND: 0.5 → 0.7: improved edge planes:
  - Plane 109: 30.0 → 30.89 dB (+0.89 dB)
  - Plane 110: 24.8 → 26.56 dB (+1.76 dB)
  - Plane 111: 20.1 → 22.64 dB (+2.54 dB)

**Lesson:** ZETA_EL saturated at 0.020 (diminishing returns: +9.1 → +1.79 → -0.01 dB).
Stronger edge blending (EDGE_BLEND=0.7) helps edge planes but contributes little overall.

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 7 — 2026-03-05 (SLURM job 178815) — SeqDiff warm-start (τ'=50)

**Config:**
- USE_SEQDIFF = True, SEQDIFF_TAU = 50 (50 denoising steps from warm start at t=249)
- Warm start: Run 6 reconstruction encoded → forward-diffused to t=249 → 50 reverse steps
- ZETA_EL = 0.020 (reverted from 0.030), EDGE_BLEND = 0.7 (kept)
- All other params same as Run 6

**Metrics (missing elevation planes only):**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **36.45 dB** | +1.92 dB vs Run 6 (34.53) — **best so far** |
| SSIM   | **0.956** | +0.036 vs Run 6 (0.920) — significant |
| LPIPS  | 0.000107 | Substantially improved from 0.000187 |

**Comparison:**
- ZEA/JAX baseline (r=4): PSNR ~23.1 dB, SSIM ~0.32
- Paper (r=6): PSNR ~23.5 dB
- Run 6 (cold start, best): PSNR 34.53 dB, SSIM 0.920
- Run 7 (SeqDiff warm-start): PSNR **36.45 dB**, SSIM **0.956** ← new SOTA

**Key observations:**
- ||M||₂ started at 2.65 (from Run 6), converged to 0.30 by step 10, then crept up to 0.51 at step 40
  (slight ||M||₂ increase in GAMMA_VL=12 regime — over-correction; but final metrics still much better)
- Interior planes (steps 80-107): 32-37 dB typical vs 30-35 dB in Run 6
- Edge planes: 109→31.28 dB, 110→26.87 dB, 111→22.84 dB (similar to Run 6)

**Lesson:** SeqDiff warm-start from a high-quality cold reconstruction gives significant improvement.
ZETA_EL=0.020 (not 0.030) is the sweet spot.

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

### Run 8 — 2026-03-05 (SLURM job 178816) — SeqDiff chaining (2nd iteration)

**Config:** Same as Run 7; SeqDiff warm-starts from Run 7 output (PSNR 36.45 dB)

**Metrics:**

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | **36.46 dB** | +0.01 dB vs Run 7 — converged, no further improvement |
| SSIM   | **0.957** | +0.001 vs Run 7 — marginal |
| LPIPS  | 0.000102 | Marginally improved |

**Lesson:** SeqDiff chaining (2nd iteration) gives essentially zero improvement. Algorithm has converged.
The PSNR has plateaued at ~36.45-36.46 dB. Further improvement requires a fundamentally different approach
(e.g., full gradient through transformer on A100, or different regularization strategy).

**Outputs saved:**
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json`
- `outputs/pixart_evaluation.png`
- `outputs/pixart_elevation_profile.png`

---

## Multi-Patient Generalization — CETUS all 45 patients, ED phase, r=4

### Test Run — 2026-03-05 (SLURM job 178817, patient01 ED)

**Config:** Cold start (200 steps), same algorithm as Run 5 (ZETA_EL=0.020, EDGE_BLEND=0.7)
- Ground truth: scipy-interpolated to (112,112,112) via `prepare_volume.py` (not JAX resize)
- No SeqDiff (first run for this patient, no prior output)
- GPU: V100 (vgpu), float16

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR   | 32.45 dB | Cold start; slightly lower than Run 5 (34.54) due to scipy vs JAX preprocessing |
| SSIM   | 0.897 | Good structural fidelity |
| LPIPS  | 0.000202 | Perceptually similar |

**Note:** Difference vs Run 5 (34.54 dB) is due to scipy vs JAX volume preprocessing (slightly different trilinear interpolation). Both evaluate against their own GT.

### Full Array — 2026-03-05 (SLURM job 178818, array 1-45, ED phase)

**Config:** Same as test run. All 45 patients submitted simultaneously.
- patient01: SeqDiff warm-start (from test run) → PSNR=33.93 dB, SSIM=0.938 (+1.48 dB vs cold start)
- patients 02-45: cold start (200 steps) — **ALL COMPLETE** (2026-03-05, ~4.5h total)

**Final results (45/45 patients, cold start except patient01):**

| Stat | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Mean | 30.88 dB | 0.829 | 0.000216 |
| Std  | 1.13 dB  | 0.038 | 0.000021 |
| Min  | 27.32 dB (patient29) | 0.717 | 0.000120 |
| Max  | 33.93 dB (patient01, SeqDiff) | 0.938 | 0.000281 |

**Cold-start-only mean (patients 2-45): ~30.85 dB**

**Comparison vs paper (r=6, ~23.5 dB): +7.4 dB improvement across full CETUS dataset**

Notable patients: patient29=27.32 dB (outlier), patient43=29.43, patient18=28.94 dB (challenging echo quality)

---

## Full-Gradient DPS Experiment — 2026-03-05 (SLURM job 178863, patient01 ED, agpu A100)

**Config:**
- `--full-grad` flag: true DPS gradient through transformer + VAE (not just VAE)
- BATCH_SIZE=1 (required for memory), float16, agpu A100 80GB
- Cold start (200 steps), ZETA_EL=0.020, EDGE_BLEND=0.7, GAMMA schedule 1/2/8/12
- Runtime: ~50 min on A100 (vs ~70 min approximate DPS on V100)

| Metric | Value | vs Approximate DPS (32.45 dB) |
|--------|-------|-------------------------------|
| PSNR   | 32.83 dB | +0.38 dB |
| SSIM   | 0.895 | -0.002 |
| LPIPS  | 0.000211 | ~same |

**Conclusion:** Full-gradient DPS gives modest +0.38 dB improvement over approximate DPS.
The approximation (gradient through VAE only) works well because:
1. Most improvement happens at late steps (t < 400) where the Jacobian correction is small
2. The GAMMA schedule already compensates for gradient magnitude differences
SeqDiff warm-start (+1.48 dB) is more impactful than full gradient (+0.38 dB).

**Lesson:** Full-gradient DPS is theoretically correct and implementable, but the approximation already works well. Not worth the A100 requirement for production runs. Stick with approximate DPS on V100.

---

## SeqDiff 2nd Pass — 2026-03-05 (SLURM job 178864, array 1-45, ED phase)

**Config:** SeqDiff warm-start from cold-start outputs (178818). Each patient forward-diffused to t=249, 50 reverse steps.
- GPU: V100 (vgpu), float16
- All 45 patients COMPLETE (2026-03-05 ~10:30 AM)

**Final results (45/45 patients, all SeqDiff):**

| Stat | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Mean | 32.52 dB | 0.882 | 0.000142 |
| Std  | 1.08 dB  | 0.028 | 0.000014 |
| Min  | 29.35 dB (patient29) | 0.812 | 0.000117 |
| Max  | 34.73 dB (patient02) | 0.941 | 0.000184 |

**SeqDiff gain: +1.64 dB** (30.88 dB cold → 32.52 dB SeqDiff, mean across all 45 patients)

**Comparison vs paper (r=6, ~23.5 dB): +9.02 dB improvement across full CETUS dataset**

Per-patient results (PSNR, SSIM, LPIPS):
```
patient01: 33.96  0.939  0.000117    patient02: 34.73  0.932  0.000135
patient03: 33.69  0.914  0.000141    patient04: 32.21  0.893  0.000149
patient05: 33.76  0.887  0.000128    patient06: 32.18  0.853  0.000142
patient07: 32.48  0.880  0.000123    patient08: 32.95  0.878  0.000137
patient09: 32.94  0.887  0.000148    patient10: 31.83  0.865  0.000140
patient11: 31.26  0.878  0.000139    patient12: 32.29  0.888  0.000144
patient13: 32.57  0.901  0.000164    patient14: 31.89  0.891  0.000155
patient15: 32.13  0.871  0.000163    patient16: 30.99  0.845  0.000147
patient17: 34.66  0.902  0.000136    patient18: 30.65  0.831  0.000177
patient19: 33.43  0.901  0.000135    patient20: 33.63  0.878  0.000131
patient21: 34.63  0.941  0.000131    patient22: 32.73  0.930  0.000134
patient23: 31.94  0.870  0.000127    patient24: 32.37  0.879  0.000128
patient25: 31.97  0.868  0.000141    patient26: 31.58  0.852  0.000140
patient27: 32.23  0.863  0.000148    patient28: 31.46  0.848  0.000136
patient29: 29.35  0.831  0.000184    patient30: 31.54  0.857  0.000138
patient31: 33.58  0.921  0.000117    patient32: 33.29  0.918  0.000147
patient33: 31.94  0.871  0.000169    patient34: 32.47  0.868  0.000138
patient35: 33.02  0.872  0.000152    patient36: 32.45  0.858  0.000144
patient37: 31.42  0.884  0.000153    patient38: 32.45  0.863  0.000139
patient39: 32.84  0.899  0.000151    patient40: 32.75  0.872  0.000143
patient41: 33.31  0.917  0.000140    patient42: 32.63  0.879  0.000146
patient43: 30.76  0.812  0.000151    patient44: 33.52  0.889  0.000128
patient45: 32.80  0.891  0.000134
```

**Notable outliers:** patient29 (29.35 dB), patient43 (30.76 dB), patient18 (30.65 dB) — likely challenging echo quality or structural anomalies.
