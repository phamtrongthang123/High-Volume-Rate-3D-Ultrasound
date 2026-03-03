# (Unofficial) High Volume Rate 3D Ultrasound Reconstruction with Diffusion Models

Reproducing the paper's pipeline using the [ZEA toolbox](https://github.com/tue-bmd/zea). Uses a pretrained 2D cardiac echo diffusion model (EchoNet-Dynamic) and the [CETUS](https://www.creatis.insa-lyon.fr/Challenge/CETUS/) real 3D echocardiography dataset to demonstrate the full algorithm: DPS-based interpolation of missing elevation planes with data consistency replacement, TV smoothness regularization (axial + elevation), and SeqDiff temporal acceleration.

All runnable code is at the repository root.

## Quick Start

```bash
pip install -r requirements.txt
bash run_setup_env.sh      # installs zea, downloads data, verifies GPU
bash run_reconstruct.sh    # runs the full pipeline
bash run_sweep.sh          # multi-rate sweep: r ∈ {2, 3, 4, 6, 10}
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
| `03_reconstruct_volume.py` | Reconstructs missing planes via DPS + data consistency + TV (axial + elevation), saves `outputs/reconstructed_volume.npy` |
| `04_seqdiff_temporal.py` | Compares cold-start (200 steps) vs warm-start (50 steps), saves `outputs/04_seqdiff_comparison.png` |
| `05_evaluate.py` | Computes PSNR/SSIM/LPIPS, saves `outputs/05_evaluation.png` and `outputs/05_elevation_profile.png` |
| `09_sweep_summary.py` | Reads `outputs/r*/metrics.json`, produces comparison table, CSV, and figure |
| `run_sweep.sh` | Runs reconstruction + evaluation for r ∈ {2, 3, 4, 6, 10}, then summary |

## Outputs

All outputs go to `outputs/`. Key files:

- `01_prior_samples.png` — sanity check that the diffusion prior works
- `pseudo_volume.npy` — ground truth pseudo-3D volume
- `reconstructed_volume.npy` — reconstructed volume after DPS + TV
- `04_seqdiff_comparison.png` — cold vs warm-start comparison
- `05_evaluation.png` — GT / reconstruction / difference side-by-side
- `05_elevation_profile.png` — inter-plane smoothness plot
- `r{2,3,4,6,10}/` — per-rate sweep outputs (from `run_sweep.sh`)
- `09_sweep_summary.png` — metrics vs acceleration rate with paper comparison
- `09_sweep_summary.csv` — sweep results as CSV

## Results

### Multi-rate sweep (CETUS patient01)

| r  | PSNR (dB) | SSIM   | LPIPS  |
|----|-----------|--------|--------|
| 2  | 23.48     | 0.3587 | 0.4198 |
| 3  | 23.27     | 0.3300 | 0.4373 |
| 4  | 23.05     | 0.3158 | 0.4462 |
| 6  | 22.67     | 0.2981 | 0.4530 |
| 10 | 21.95     | 0.2730 | 0.4668 |

Parameters: `N_STEPS=200, GAMMA=15.0, ZETA=0.001, ZETA_EL=0.003`

### Paper's reported results (approximate, read from figures)

**PSNR (Figure 6) — Diffusion method vs. baselines:**

| r | Nearest | Linear | U-Net | Neural Field | Diffusion |
|---|---------|--------|-------|-------------|-----------|
| 2 | ~27.0 | ~28.0 | ~29.3 | ~28.3 | ~28.7 |
| 3 | ~23.2 | ~25.8 | ~25.8 | ~24.7 | ~26.3 |
| 6 | ~19.5 | ~21.0 | ~21.6 | ~20.0 | ~23.5 |
| 10 | ~16.8 | ~18.3 | ~18.4 | ~17.1 | ~22.3 |

**LPIPS on B-planes (Figure 7) — Diffusion method vs. baselines:**

| r | Nearest | Linear | U-Net | Neural Field | Diffusion |
|---|---------|--------|-------|-------------|-----------|
| 2 | ~0.12 | ~0.105 | ~0.075 | ~0.145 | ~0.095 |
| 3 | ~0.19 | ~0.165 | ~0.175 | ~0.215 | ~0.13 |
| 6 | ~0.245 | ~0.225 | ~0.26 | ~0.275 | ~0.16 |
| 10 | ~0.305 | ~0.265 | ~0.29 | ~0.29 | ~0.19 |

### Comparison notes

Our PSNR ranges from 23.5 dB (r=2) to 22.0 dB (r=10). At r=6, our 22.7 dB vs. the paper's ~23.5 dB reflects the domain gap. Key differences:

- **Paper**: proprietary 3D cardiac echo data with a diffusion prior trained on matching B-plane data
- **Ours**: CETUS (public 3D echo, different acquisition characteristics) with EchoNet-Dynamic prior (trained on 2D apical 4-chamber views from a different dataset)
- The domain gap between the 2D A4C prior and real 3D echo B-planes limits reconstruction fidelity, particularly for LPIPS

## Known Limitations

- **Domain gap**: The diffusion prior is trained on EchoNet-Dynamic 2D A4C views, not 3D echo B-plane slices. This limits how well the prior can guide reconstruction of elevation planes that look different from A4C views.
- **Single patient**: Results above are for CETUS patient01 only. The paper evaluates across multiple patients and cardiac phases.
- **No speckle tracking**: The paper includes speckle tracking experiments that require proprietary data and are not reproduced here.
