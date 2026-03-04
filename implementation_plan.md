# Implementation Plan: pipeline_pixart/

> Next step: start at Task 1 below.

## Tasks

- [ ] **1. Create `pipeline_pixart/` directory structure**
  - Create `pipeline_pixart/` at project root

- [ ] **2. Copy and adapt `03_reconstruct_volume.py`**
  - Copy `training/reconstruct_volume_pixart.py` → `pipeline_pixart/03_reconstruct_volume.py`
  - Copy `training/pixart_diffusion_model.py` → `pipeline_pixart/pixart_diffusion_model.py`
  - Patch paths in `03_reconstruct_volume.py`:
    - `OUTPUT_DIR` → `../outputs`
    - `LORA_PATH` → `../training/checkpoints/cetus_pixart_lora/best-checkpoint/transformer_lora`
    - `VAE_DECODER_PATH` → `../training/checkpoints/vae_decoder_finetuned/vae_decoder.pt`
    - Output file: `../outputs/reconstructed_volume_pixart.npy`
  - If `best-checkpoint` doesn't exist, fall back to highest-numbered checkpoint

- [ ] **3. Add `pipeline_pixart/05_evaluate.py`**
  - Copy root `05_evaluate.py` → `pipeline_pixart/05_evaluate.py`
  - Load `outputs/reconstructed_volume_pixart.npy` instead of `outputs/reconstructed_volume.npy`
  - Save to `outputs/pixart_evaluation.png` and `outputs/pixart_metrics.json`

- [ ] **4. Add `pipeline_pixart/run.sh`**
  - `cd` to script directory, then run `03_reconstruct_volume.py` then `05_evaluate.py`

- [ ] **5. Smoke-test the pipeline**
  - Verify `outputs/pseudo_volume.npy` exists (prerequisite)
  - Run `bash pipeline_pixart/run.sh` (single GPU, `CUDA_VISIBLE_DEVICES=0`)
  - Confirm outputs: `reconstructed_volume_pixart.npy`, `pixart_metrics.json`, `pixart_evaluation.png`
  - Record PSNR / SSIM / LPIPS in SCORES.md (append-only, never delete history)
