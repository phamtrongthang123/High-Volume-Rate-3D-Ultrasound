# Implementation Plan: pipeline_pixart/

> Full run history and per-run configs are in **SCORES.md**.

## Multi-patient generalization (current focus — branch: to-dup-camus)

Goal: Run on all 45 CETUS patients (ED phase) to evaluate generalizability.

**New scripts:**
- `pipeline_pixart/prepare_volume.py` — pure PyTorch/SimpleITK/scipy CETUS loader, any patient
- `pipeline_pixart/run_all_patients_slurm.sh` — SLURM array job (--array=1-45, ED phase)
- `pipeline_pixart/summarize_patients.py` — aggregate metrics after array completes

**CLI args added:**
- `03_reconstruct_volume.py --output-dir outputs/patientXX_ED/`
- `05_evaluate.py --output-dir outputs/patientXX_ED/`

**Status:**
- SLURM test job 178817 (patient01 ED) — **COMPLETE**: PSNR=32.45 dB, SSIM=0.897 (cold start)
- SLURM array job 178818 (all 45 patients ED, array 1-45) — **RUNNING** (2026-03-05)
  - Batch 1 (1-11): **COMPLETE** (11 patients, ~05:04 AM)
  - Batch 2 (12-22): running ~05:04–06:14 AM
  - Batch 3 (23-33): pending ~06:14–07:24 AM
  - Batch 4 (34-45): pending ~07:24–08:34 AM

**Preliminary results (11/45 patients, cold start except patient01):**
- Mean PSNR: 31.35 dB, SSIM: 0.844
- patient01 SeqDiff: 33.93 dB; patients 2-11 cold start: 30.06–32.39 dB
- Cold-start-only mean: ~30.3 dB
- Full comparison vs paper (r=6, ~23.5 dB): **+7.8 dB improvement**

**Full-gradient DPS experiment:**
- SLURM job 178863 (agpu A100, patient01 ED) — **COMPLETE** 2026-03-05 04:49 AM
- Result: PSNR=32.83 dB, SSIM=0.895 vs approximate 32.45 dB → **+0.38 dB**
- Conclusion: full-grad is slightly better but not worth A100 requirement for production

**New scripts:**
- `pipeline_pixart/run_full_grad_slurm.sh` — agpu A100 full-gradient experiment
- `pipeline_pixart/run_es_patients_slurm.sh` — ES phase array job (1-45)

**All 45 patients COMPLETE (2026-03-05 ~08:40 AM):**
- Mean PSNR: 30.88 dB, SSIM: 0.829 (cold start, patient01 has SeqDiff)
- Cold-start-only mean: ~30.85 dB; **+7.4 dB vs paper (r=6, ~23.5 dB)**
- SLURM 178864 (SeqDiff 2nd pass, all 45 patients) — **COMPLETE** 2026-03-05 10:08 AM

**SeqDiff 2nd pass COMPLETE (2026-03-05 ~10:08 AM, SLURM 178864):**
- Mean PSNR: **32.52 dB**, SSIM: **0.882**, Std: 1.08 dB
- SeqDiff gain: +1.64 dB (30.88 → 32.52 dB); range: 29.35 (patient29) – 34.73 (patient02)
- **+9.02 dB vs paper (r=6, ~23.5 dB)**
- Problematic patients: patient29=29.35 dB, patient18=30.65, patient43=30.76 dB

**ES phase — SUBMITTED** (SLURM 178914, array 1-45, submitted 2026-03-05 10:08 AM)
- Expected: cold-start results ~30-31 dB; completes ~5:00 PM
- After completion: `python pipeline_pixart/summarize_patients.py` (needs `--output-dir` glob for ES)

## Future experiments

1. **Full gradient through transformer** (agpu A100): **DONE** (2026-03-05, SLURM 178863)
   - Result: +0.38 dB vs approximate DPS (32.83 vs 32.45 dB, patient01 cold start)
   - Conclusion: modest gain; approximation works well; not worth A100 for production
   - Implementation: `--full-grad` flag in `03_reconstruct_volume.py`
2. **SEQDIFF_TAU=100**: more aggressive warm-start (t=499, 100 steps). Not yet tried.
   - Risk: t=499 is quite noisy (α≈0.707), may destroy reconstruction structure
   - Expected gain: possibly higher than τ=50 (+1.48 dB) but risky
3. **Different measurement operator**: try soft/noisy measurements instead of hard binary mask.
4. **ES phase evaluation**: script ready (`run_es_patients_slurm.sh`). Submit after ED completes.
5. **SeqDiff 2nd pass (all patients)**: submit `run_all_patients_slurm.sh` again after ED cold start.
   - Expected: mean ~32.8 dB (+1.5 dB from ~31.3 dB cold start)

## Run summary (CETUS patient01, r=4, N_STEPS=200)

| Run | SLURM | PSNR | SSIM | Key change |
|-----|-------|------|------|------------|
| 1 | 178795 | 15.38 dB | 0.079 | Baseline pixel-space DPS (prior collapse) |
| 2 | 178803 | 16.59 dB | 0.348 | Latent-space DPS, gradient through VAE only |
| 3 | 178804 | 23.65 dB | 0.616 | + elevation TV (ZETA_EL=0.003) + GAMMA schedule 1/2/8 |
| 4 | 178805 | 32.75 dB | 0.868 | ZETA_EL=0.010, GAMMA_VL=12 at t≤200, LATENT_CLIP=8.0 |
| 5 | 178810 | 34.54 dB | 0.916 | ZETA_EL=0.020, edge blending (EDGE_BLEND=0.5) |
| 6 | 178814 | 34.53 dB | 0.920 | ZETA_EL=0.030 (no gain), EDGE_BLEND=0.7 |
| 7 | 178815 | **36.45 dB** | **0.956** | SeqDiff warm-start (τ'=50) from Run 6 |
| 8 | 178816 | **36.46 dB** | **0.957** | SeqDiff chaining (2nd iter) from Run 7 — converged |

**Baselines:** ZEA/JAX (r=4): PSNR ~23.1 dB, SSIM ~0.32 | Paper (r=6): PSNR ~23.5 dB

## Current best algorithm (Run 8, SeqDiff-chained — `03_reconstruct_volume.py`)

- State: latent space `(N_az, 4, 64, 64)` throughout; decode only for measurement comparison and final output
- Schedule: DDPM `alphas_cumprod` from `model.noise_scheduler`
- Gradient: through VAE decode only (not transformer) for memory efficiency on V100
- Guidance: `gamma * grad_x0 / eff_alpha`, clipped per-element to `[-GUIDANCE_CLIP, GUIDANCE_CLIP]`
  - GUIDANCE_CLIP = 0.3 (early, t>400), 0.8 (late, t≤400)
- GAMMA schedule: 1.0 / 2.0 / 8.0 / 12.0 for t>700 / 400-700 / 200-400 / ≤200
- TV: azimuth ZETA=0.001 (axis=3), elevation ZETA_EL=0.020 (axis=2 of latent H)
- LATENT_CLIP=8.0 after each step (keeps VAE in-distribution; max |lat| ~6.4)
- EFFECTIVE_ALPHA_MIN=0.15 (prevents NaN from alpha→0 at early steps)
- Edge blending (post-decode): EDGE_BLEND=0.7 (`X[el] = 0.7^delta * X[last_obs] + (1-0.7^delta) * X_decoded[el]`)
- **SeqDiff**: USE_SEQDIFF=True, SEQDIFF_TAU=50 — forward-diffuses last output to t=249, runs 50 reverse steps
  - Phase 1: 200-step cold start (Runs 1-6)
  - Phase 2: SeqDiff 50 steps from Run 6 output → Run 7 (+1.92 dB)
  - Phase 3: SeqDiff 50 steps from Run 7 output → Run 8 (+0.01 dB, converged)

## Invariants and gotchas

| Issue | Fix |
|-------|-----|
| PEFT 0.13 vs checkpoint 0.18.1 | Custom `_merge_lora_into_transformer`: W_merged = W_base + (alpha/r) × lora_B @ lora_A |
| V100 no bfloat16 | Auto-detect cc<8 → float16; cc≥8 → bfloat16 |
| T5-XXL + transformer OOM on 32GB V100 | Load T5 → encode → delete T5 → load VAE + transformer sequentially |
| SLURM stdout silent | `python -u` flag in `run_slurm.sh` |
| DPS NaN at early steps | EFFECTIVE_ALPHA_MIN=0.15 (not 0.0) |
| Edge planes 109-111 | EDGE_BLEND=0.7 → 30.9/26.9/22.8 dB (Run 6+) |
| ZETA_EL saturation | Plateau at 0.020; increasing to 0.030 gives -0.01 dB (worse) |
| SeqDiff convergence | Single SeqDiff +1.92 dB; 2nd iteration +0.01 dB (converged) |
| Full-grad DPS memory | BATCH_SIZE=1 needed on A100; `--full-grad` flag forces this automatically |
| Full-grad DPS gain | Only +0.38 dB over approx DPS (approximate VAE-only gradient is already good) |
| Job submission limit | QOSMaxSubmitJobPerUserLimit: can't submit ES array while ED array is pending |
| Multi-patient variance | Cold-start PSNR varies 30-32 dB (11/45 patients); SeqDiff adds ~+1.5 dB |

## Pipeline files

| File | Role |
|------|------|
| `pipeline_pixart/03_reconstruct_volume.py` | Core DPS reconstruction (latent space); `--full-grad` flag |
| `pipeline_pixart/pixart_diffusion_model.py` | PixArt-α wrapper with manual LoRA merge |
| `pipeline_pixart/05_evaluate.py` | PSNR/SSIM/LPIPS (PyTorch; VGG16-based LPIPS) |
| `pipeline_pixart/prepare_volume.py` | Load CETUS patient (any patient/phase) via scipy |
| `pipeline_pixart/run_slurm.sh` | SLURM submission (vgpu partition, `python -u`) |
| `pipeline_pixart/run_all_patients_slurm.sh` | SLURM array 1-45, ED phase |
| `pipeline_pixart/run_es_patients_slurm.sh` | SLURM array 1-45, ES phase |
| `pipeline_pixart/run_full_grad_slurm.sh` | SLURM agpu A100, full-gradient DPS |
| `pipeline_pixart/summarize_patients.py` | Aggregate per-patient metrics |

Outputs: `outputs/reconstructed_volume_pixart.npy`, `outputs/pixart_metrics.json`, `outputs/pixart_evaluation.png`, `outputs/pixart_elevation_profile.png`
