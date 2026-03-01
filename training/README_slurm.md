# Submitting Training Jobs on Pinnacle HPC

## Overview

The `train.sbatch` script runs PixArt-alpha LoRA fine-tuning on the `agpu` partition (A100 nodes). It requests 1 GPU, 32 CPU cores, 64 GB RAM, and a 24-hour wall time.

Trained weights are saved to `training/checkpoints/cetus_pixart_lora/`.

## Step-by-step

### 1. SSH into the login node

```bash
ssh pinnacle-l1
```

`pinnacle-l1` is the login node. Compute happens on `agpu72` partition nodes allocated by SLURM.

### 2. Navigate to the project

```bash
cd /scrfs/storage/tp030/home/High-Volume-Rate-3D-Ultrasound
```

### 3. Submit the job

```bash
sbatch training/train.sbatch
```

Output: `Submitted batch job <JOBID>`

### 4. Monitor job status

```bash
squeue -u $USER
squeue -u $USER -o "%.8i %.9P %.20j %.2t %.10M %.6D %R"
```

### 5. Watch live output (once RUNNING)

```bash
tail -f training/slurm_<JOBID>.out
```

Replace `<JOBID>` with the actual job ID from step 3.

### 6. Check for errors

```bash
cat training/slurm_<JOBID>.err
```

### 7. Cancel if needed

```bash
scancel <JOBID>
```

## Output files

| File | Contents |
|------|----------|
| `training/slurm_<JOBID>.out` | Standard output (training logs) |
| `training/slurm_<JOBID>.err` | Standard error (warnings, errors) |
| `training/checkpoints/cetus_pixart_lora/` | Trained LoRA weights |

## Notes

- WandB: training logs appear at https://wandb.ai once the job starts (if wandb is configured in the environment).
- Dry-run validation: `sbatch --test-only training/train.sbatch` (checks resource availability without submitting).
