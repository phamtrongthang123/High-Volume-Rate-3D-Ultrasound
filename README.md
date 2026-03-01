# (Unofficial) High Volume Rate 3D Ultrasound Reconstruction with Diffusion Models

Reproducing the paper's pipeline using the [ZEA toolbox](https://github.com/zeahub/zea). This demo uses a pretrained 2D cardiac echo diffusion model and the CAMUS dataset to demonstrate the full algorithm: DPS-based interpolation of missing elevation planes, TV smoothness regularization, and SeqDiff temporal acceleration.

All runnable code is at the repository root.

## Quick Start

```bash
pip install -r requirements.txt
bash run_setup_env.sh      # installs zea, downloads data, verifies GPU
bash run_reconstruct.sh    # runs the full pipeline
```

### File structure

```
setup_env.sh          ← convenience wrapper for run_setup_env.sh
run_setup_env.sh      ← setup logic (install deps, verify GPU, download data)

full_reconstruct.sh   ← convenience wrapper for run_reconstruct.sh
run_reconstruct.sh    ← pipeline logic (runs all 5 steps)
```

### Quick test (faster iteration)

Edit these values in the scripts before running:

| Parameter | Quick test | Paper-faithful |
|-----------|-----------|----------------|
| `N_STEPS` | 50 | 200 |
| `N_ELEVATION` | 8 | 16 |
| `ACCEL_RATE` | 2 | 4 |
| `OMEGA` | 35.0 | 35.0 |
| `ZETA` | 0.001 | 0.001 |
| `SEQDIFF_TAU` | 50 | 50 |

## Pipeline Overview

```
00_download_data.py       Download + cache CAMUS dataset
        |
01_verify_prior.py        Unconditional sampling to verify model works
        |
02_prepare_pseudo_volume.py   Stack 2D images as pseudo-3D volume
        |
03_reconstruct_volume.py  DPS reconstruction of missing planes + TV smoothness
        |
   +---------+----------+
   |                     |
04_seqdiff_temporal.py   05_evaluate.py
SeqDiff speedup demo     Metrics + visualization
```

## What Each Script Does

| Script | Description |
|--------|-------------|
| `setup_env.sh` | Wrapper → runs `run_setup_env.sh` |
| `run_setup_env.sh` | Installs zea, verifies JAX/GPU/Keras, downloads CAMUS data |
| `full_reconstruct.sh` | Wrapper → runs `run_reconstruct.sh` |
| `run_reconstruct.sh` | Runs all 5 Python steps sequentially |
| `00_download_data.py` | Pre-downloads CAMUS dataset |
| `01_verify_prior.py` | Generates 8 unconditional samples, saves `outputs/01_prior_samples.png` |
| `02_prepare_pseudo_volume.py` | Creates `outputs/pseudo_volume.npy` (16 planes) and `pseudo_volume_t2.npy` |
| `03_reconstruct_volume.py` | Reconstructs missing planes via DPS + TV, saves `outputs/reconstructed_volume.npy` |
| `04_seqdiff_temporal.py` | Compares cold-start (200 steps) vs warm-start (50 steps), saves `outputs/04_seqdiff_comparison.png` |
| `05_evaluate.py` | Computes PSNR/SSIM/LPIPS, saves `outputs/05_evaluation.png` and `outputs/05_elevation_profile.png` |

## Outputs

All outputs go to `outputs/`. Key files:

- `01_prior_samples.png` — sanity check that the diffusion prior works
- `pseudo_volume.npy` — ground truth pseudo-3D volume
- `reconstructed_volume.npy` — reconstructed volume after DPS + TV
- `04_seqdiff_comparison.png` — cold vs warm-start comparison
- `05_evaluation.png` — GT / reconstruction / difference side-by-side
- `05_elevation_profile.png` — inter-plane smoothness plot

## Expected Metrics Behavior

`02_prepare_pseudo_volume.py` loads two **consecutive temporal frames** from the 4CH cardiac sequence (frame 0 → t1, frame 1 → t2). Adjacent frames share strong structural similarity — the anatomy changes only slightly between heartbeats — which satisfies SeqDiff's core assumption that consecutive volumes are similar enough to warm-start from the previous reconstruction.

The step 03 ground truth (`pseudo_volume.npy`) uses frame 0 uniformly tiled across all elevation planes. The reconstruction task and metrics are therefore:

**Observed results** (N_STEPS=200, OMEGA=35, ZETA=0.001):

| Metric | Overall (missing planes) |
|--------|--------------------------|
| PSNR   | ~14.6 dB                 |
| SSIM   | ~0.22                    |
| LPIPS  | ~0.39                    |

These scores reflect the difficulty of reconstructing missing elevation planes from a pseudo-volume built from a single 2D image tiled uniformly — not a real 3D dataset. With real 3D ultrasound data (smooth, continuous variation along elevation), scores should be substantially higher.

## Known Limitations

**Data limitations** (CAMUS sample dataset, not the paper's proprietary 3D cardiac data):
- Only 2 unique source images available — causes bimodal metrics (see above)
- No real elevation coherence between stacked planes (different patients)
- Pretrained model is on EchoNet-Dynamic (A4C views), not actual B-plane slices
- No speckle tracking or out-of-distribution experiments possible without real 3D volumes
