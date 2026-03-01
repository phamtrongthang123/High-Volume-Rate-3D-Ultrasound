# High-Volume-Rate 3D Ultrasound Reconstruction

Unofficial reproduction of diffusion-based 3D cardiac ultrasound reconstruction.
Uses DPS (Diffusion Posterior Sampling) + TV regularization + SeqDiff warm-starting.

## CETUS Dataset

> **Primary data source.** All scripts read from `data/CETUS/dataset/`.

- **Full name**: Cardiac Echo Training and Ultrasound Segmentation (CETUS)
- **Path**: `data/CETUS/dataset/patientXX/`
- **Format**: NIfTI `.nii.gz` — `patientXX_ED.nii.gz`, `patientXX_ES.nii.gz` (+ `_gt` segmentations)
- **Content**: 45 patients, actual 3D echocardiography volumes (End-Diastole and End-Systole phases)
- **Verified via**: `python 00_download_data.py` (checks `data/CETUS/dataset/` is present and readable; data must be obtained manually)
- **License**: CC BY-NC-SA 4.0 — non-commercial use only
- **Citation (MANDATORY)**: O. Bernard et al., "Standardized Evaluation System for Left Ventricular Segmentation Algorithms in 3D Echocardiography," IEEE TMI 35(4), 2016. https://doi.org/10.1109/tmi.2015.2503890

### How CETUS data flows through the pipeline

```
CETUS NIfTI (patientXX_ED.nii.gz, patientXX_ES.nii.gz)
  → load real 3D echo volume (Z, Y, X) via SimpleITK
  → trilinear resize to (N_el=112, N_az=112, N_ax=112)
  → normalize to [-1, 1] → shape (N_el, N_az, N_ax, 1)
  → DPS reconstruction → evaluated vs. ground truth
```

t1 = ED (End-Diastole), t2 = ES (End-Systole) — same patient, different cardiac phases.

## Pipeline

| Script | Role |
|--------|------|
| `00_download_data.py` | Verify CETUS dataset is present and readable |
| `01_verify_prior.py` | Sanity-check pretrained diffusion model |
| `02_prepare_pseudo_volume.py` | Load real 3D CETUS ED/ES volumes for reconstruction |
| `03_reconstruct_volume.py` | **Core**: DPS + TV regularization reconstruction |
| `04_seqdiff_temporal.py` | SeqDiff warm-start demo (cold 200 vs warm 50 steps) |
| `05_evaluate.py` | PSNR / SSIM / LPIPS metrics + visualizations |
| `06_compare_prior_vs_data.py` | Prior vs. data distribution analysis (optional) |

## Commands

```bash
# Full pipeline
bash run_reconstruct.sh

# Individual steps
python 00_download_data.py
python 02_prepare_pseudo_volume.py
python 03_reconstruct_volume.py
python 04_seqdiff_temporal.py
python 05_evaluate.py
```

## Key Parameters

| Parameter | Variable | Default | Notes |
|-----------|----------|---------|-------|
| Diffusion steps | `N_STEPS` | 200 | 50 for quick testing |
| Observed elevation planes | `N_ELEVATION` | 16 | 8 for quick testing |
| Acceleration rate | `ACCEL_RATE` | 4 | observe every 4th plane |
| DPS guidance strength | `GAMMA` (Ω) | 35.0 | Currently using 15.0 for best results |
| TV smoothness strength | `ZETA` (ζ) | 0.001 | Axial TV within each B-plane |
| Elevation TV strength | `ZETA_EL` | 0.003 | TV across elevation planes |
| SeqDiff warm-start step | `SEQDIFF_TAU` (τ') | 50 | |

## Environment

- **Backend**: JAX (GPU required) + Keras
- **Setup**: `import env_setup` must be first import in every script
  - Sets `ZEA_CACHE_DIR=cache/`, `KERAS_BACKEND=jax`
- **Initial setup**: `bash run_setup_env.sh` (installs ZEA, verifies GPU, downloads CETUS)

## Outputs

All outputs written to `outputs/`:
- `pseudo_volume.npy` — GT 3D volume (CETUS ED phase, patient01)
- `pseudo_volume_t2.npy` — GT 3D volume (CETUS ES phase, patient01)
- `reconstructed_volume.npy` — DPS-reconstructed volume
- `04_seqdiff_comparison.png` — cold vs warm-start visual comparison
- `05_evaluation.png` — GT | Reconstruction | Difference
- `05_elevation_profile.png` — elevation coherence profile

## Expected Metrics (CETUS patient01, r=4)

| Metric | Our result | Paper at r=6 |
|--------|-----------|--------------|
| PSNR   | ~23.1 dB  | ~23.5 dB     |
| SSIM   | ~0.32     | —            |
| LPIPS  | ~0.45     | ~0.16 (B-plane) |

Our PSNR is comparable to the paper's at r=6 despite the domain gap (EchoNet-Dynamic 2D A4C prior vs. paper's matched 3D echo prior). LPIPS is higher due to this domain gap.
