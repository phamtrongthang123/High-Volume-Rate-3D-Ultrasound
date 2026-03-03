"""
train_vae_decoder.py — Fine-tune VAE decoder on ultrasound B-planes.

The SD VAE (used by PixArt-α) was trained on natural images. Ultrasound
B-planes have a sharp bimodal distribution (black background + bright sector)
that the VAE can't faithfully reconstruct — VAE decode output only reaches
~[-0.8, 0.3] instead of [-1, 1], causing gray backgrounds.

This script fine-tunes only the VAE decoder + post_quant_conv on ultrasound
B-planes. The encoder is frozen, so the latent space the transformer learned
is unchanged. We only improve latent → pixel reconstruction.

Training loop:
    encode(image) → latents → decode(latents) → MSE+L1 loss vs original

No scaling factor in the round-trip — vae.encode() returns raw latents,
vae.decode() expects raw latents. The scaling factor is only used externally
for the diffusion transformer.

Usage:
    accelerate launch --config_file training/accelerate_config.yaml \
        training/train_vae_decoder.py \
        --pretrained_model_name_or_path PixArt-alpha/PixArt-XL-2-512x512 \
        --train_data_dir training/dataset/train \
        --val_data_dir training/dataset/val \
        --output_dir training/checkpoints/vae_decoder_finetuned
"""

import argparse
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VAE decoder on ultrasound B-planes."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path to pretrained model (e.g. PixArt-alpha/PixArt-XL-2-512x512).",
    )
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="training/checkpoints/vae_decoder_finetuned")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--validation_epochs", type=int, default=5)
    parser.add_argument("--l1_weight", type=float, default=0.1,
                        help="Weight for L1 loss term (MSE weight is 1.0).")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def compute_psnr(pred, target):
    """Compute PSNR between two tensors in [-1, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    # Data range is 2.0 (from -1 to 1)
    return 10 * math.log10(4.0 / mse.item())


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir,
    )
    log_with = args.report_to if args.report_to != "none" else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # --- Load VAE ---
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=weight_dtype,
    )

    # Freeze encoder, unfreeze decoder + post_quant_conv
    vae.encoder.requires_grad_(False)
    vae.quant_conv.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.post_quant_conv.requires_grad_(True)

    vae.to(accelerator.device)

    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in vae.parameters())
    logger.info(f"VAE total params: {total_params:,}")
    logger.info(f"VAE trainable params (decoder + post_quant_conv): {total_trainable:,}")

    # For fp16, cast trainable params to float32 (bf16 can train natively)
    if args.mixed_precision == "fp16":
        for p in trainable_params:
            p.data = p.data.to(torch.float32)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # --- Dataset ---
    train_transform = transforms.Compose([
        transforms.Resize(
            args.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples[args.image_column]]
        examples["pixel_values"] = [train_transform(img) for img in images]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        return {"pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float()}

    with accelerator.main_process_first():
        train_data_files = {"train": os.path.join(args.train_data_dir, "**")}
        train_dataset = load_dataset(
            "imagefolder", data_files=train_data_files,
        )["train"].with_transform(preprocess)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Validation dataset
    val_dataloader = None
    if args.val_data_dir is not None:
        with accelerator.main_process_first():
            val_data_files = {"train": os.path.join(args.val_data_dir, "**")}
            val_dataset = load_dataset(
                "imagefolder", data_files=val_data_files,
            )["train"].with_transform(preprocess)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # --- LR scheduler ---
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # --- Prepare with accelerator ---
    # Only wrap the trainable parts + optimizer + dataloader
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("vae-decoder-finetune", config=vars(args))

    logger.info("***** Running VAE decoder fine-tuning *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(
        range(max_train_steps), desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    best_val_psnr = 0.0

    for epoch in range(args.num_train_epochs):
        vae.train()
        # But keep encoder in eval mode (frozen)
        vae.encoder.eval()
        vae.quant_conv.eval()

        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_dataloader:
            with accelerator.accumulate(vae):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                # Encode (frozen) — no scaling factor in round-trip
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()

                # Decode (trainable) — no scaling factor in round-trip
                reconstructed = vae.decode(latents).sample

                # Loss: MSE + L1
                target = pixel_values.float()
                pred = reconstructed.float()
                mse_loss = F.mse_loss(pred, target)
                l1_loss = F.l1_loss(pred, target)
                loss = mse_loss + args.l1_weight * l1_loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.detach().item()
                epoch_steps += 1

                accelerator.log({
                    "train_loss": loss.detach().item(),
                    "mse_loss": mse_loss.detach().item(),
                    "l1_loss": l1_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)

                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    lr=lr_scheduler.get_last_lr()[0],
                )

            if global_step >= max_train_steps:
                break

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.6f}")

        # --- Validation ---
        if (
            val_dataloader is not None
            and accelerator.is_main_process
            and (epoch + 1) % args.validation_epochs == 0
        ):
            vae.eval()
            val_psnrs = []
            val_losses = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_pixels = val_batch["pixel_values"].to(
                        dtype=weight_dtype, device=accelerator.device,
                    )
                    val_latents = vae.encode(val_pixels).latent_dist.sample()
                    val_recon = vae.decode(val_latents).sample

                    val_target = val_pixels.float()
                    val_pred = val_recon.float()
                    val_mse = F.mse_loss(val_pred, val_target)
                    val_l1 = F.l1_loss(val_pred, val_target)
                    val_loss = val_mse + args.l1_weight * val_l1
                    val_losses.append(val_loss.item())

                    # Per-sample PSNR
                    for i in range(val_pred.shape[0]):
                        psnr = compute_psnr(val_pred[i], val_target[i])
                        val_psnrs.append(psnr)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_psnr = sum(val_psnrs) / len(val_psnrs)

            # Log range stats on a sample
            with torch.no_grad():
                sample_batch = next(iter(val_dataloader))
                sample_pixels = sample_batch["pixel_values"].to(
                    dtype=weight_dtype, device=accelerator.device,
                )
                sample_latents = vae.encode(sample_pixels).latent_dist.sample()
                sample_recon = vae.decode(sample_latents).sample.float()
                recon_min = sample_recon.min().item()
                recon_max = sample_recon.max().item()

            accelerator.log({
                "val_loss": avg_val_loss,
                "val_psnr": avg_val_psnr,
                "recon_min": recon_min,
                "recon_max": recon_max,
                "epoch": epoch,
            }, step=global_step)

            logger.info(
                f"  Val: loss={avg_val_loss:.6f}, PSNR={avg_val_psnr:.2f} dB, "
                f"recon range=[{recon_min:.3f}, {recon_max:.3f}]"
            )

            # Save best checkpoint
            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                _save_decoder_checkpoint(vae, args.output_dir, tag="best")
                logger.info(
                    f"  New best val PSNR={avg_val_psnr:.2f} dB, saved."
                )

    # --- Save final checkpoint ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_decoder_checkpoint(vae, args.output_dir, tag="final")
        logger.info(f"Final decoder weights saved to {args.output_dir}")

    accelerator.end_training()


def _save_decoder_checkpoint(vae, output_dir, tag="final"):
    """Save decoder + post_quant_conv state dict."""
    # Unwrap accelerator wrappers
    unwrapped = vae
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    state_dict = {}
    for name, param in unwrapped.named_parameters():
        if name.startswith("decoder.") or name.startswith("post_quant_conv."):
            state_dict[name] = param.data.clone()
    # Also grab buffers (e.g. running stats in norm layers)
    for name, buf in unwrapped.named_buffers():
        if name.startswith("decoder.") or name.startswith("post_quant_conv."):
            state_dict[name] = buf.clone()

    if tag == "best":
        save_path = os.path.join(output_dir, "vae_decoder_best.pt")
    else:
        save_path = os.path.join(output_dir, "vae_decoder.pt")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(state_dict, save_path)
    logger.info(f"  Saved decoder checkpoint ({len(state_dict)} keys) to {save_path}")


if __name__ == "__main__":
    main()
