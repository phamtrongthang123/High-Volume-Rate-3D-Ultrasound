## Goal

Run the PixArt-α LoRA checkpoint (trained in `training/`) as the diffusion prior for the
reconstruction pipeline, without modifying the existing root-level scripts (which use JAX/ZEA).
All new work lives in a new folder: `pipeline_pixart/`.

## What already exists

- `training/reconstruct_volume_pixart.py` — PyTorch port of `03_reconstruct_volume.py` that
  uses `PixArtLatentDiffusionModel` as the prior. This is the core reconstruction script.
- `training/pixart_diffusion_model.py` — model wrapper (`PixArtDiffusionModel`,
  `PixArtLatentDiffusionModel`).
- `training/sample_prior.py` — standalone sampler for the PixArt prior.

## Task

Create `pipeline_pixart/` at the project root containing a complete, runnable pipeline:

1. **Copy** `training/reconstruct_volume_pixart.py` → `pipeline_pixart/03_reconstruct_volume.py`
   and `training/pixart_diffusion_model.py` → `pipeline_pixart/pixart_diffusion_model.py`.

2. **Update paths** in `pipeline_pixart/03_reconstruct_volume.py`:
   - `OUTPUT_DIR` → `../outputs` (reuse existing GT volume `outputs/pseudo_volume.npy`)
   - `LORA_PATH` → `../training/checkpoints/cetus_pixart_lora/best-checkpoint/transformer_lora`
   - `VAE_DECODER_PATH` → `../training/checkpoints/vae_decoder_finetuned/vae_decoder.pt`
     (use base VAE decoder if this file doesn't exist — already handled by the script)
   - `PRETRAINED_PATH` → `PixArt-alpha/PixArt-XL-2-512x512` (unchanged, HF hub)
   - Save output as `../outputs/reconstructed_volume_pixart.npy`

3. **Add** `pipeline_pixart/05_evaluate.py` — copy `05_evaluate.py` from root and patch it to:
   - Load `outputs/reconstructed_volume_pixart.npy` instead of `outputs/reconstructed_volume.npy`
   - Save figures/metrics to `outputs/pixart_evaluation.png` and `outputs/pixart_metrics.json`

4. **Add** `pipeline_pixart/run.sh` — a single script that runs the full pipeline:
   ```bash
   cd "$(dirname "$0")"
   python 03_reconstruct_volume.py
   python 05_evaluate.py
   ```

## Checkpoint to use

`training/checkpoints/cetus_pixart_lora/best-checkpoint/transformer_lora`

If `best-checkpoint` doesn't exist, fall back to the highest-numbered checkpoint available
under `training/checkpoints/cetus_pixart_lora/`.


## Machine
**1 GPU** for dev (no multi-GPU training). if you need hpc, **SLURM partitions:** Only use `vgpu` and `agpu` partitions. Do not submit jobs to any other partition. See example_slurm.sh

## Do NOT

- Modify any file in the project root (especially `03_reconstruct_volume.py`, `05_evaluate.py`).
- multi-GPU, or distributed training. Single GPU only (`CUDA_VISIBLE_DEVICES=0` if needed).
- Use JAX or ZEA in `pipeline_pixart/` — this folder is pure PyTorch.

## Success criterion

Running `bash pipeline_pixart/run.sh` produces:
- `outputs/reconstructed_volume_pixart.npy`
- `outputs/pixart_metrics.json` with PSNR / SSIM / LPIPS scores
- `outputs/pixart_evaluation.png`
