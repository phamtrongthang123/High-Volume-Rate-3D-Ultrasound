"""
train_pixart_lora.py — LoRA fine-tuning of PixArt-α on CETUS B-planes.

Adapted from medart/medart.py with simplifications:
- Single fixed prompt for all images (no multi-caption handling)
- No color distribution loss (not meaningful for grayscale ultrasound)
- No hub upload / model card code
- No micro_conditions for 1024-MS variant
- Higher default learning rate (1e-4) for larger domain shift
- Text encoder LoRA disabled by default (single fixed prompt)
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Union

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of PixArt-α on CETUS B-planes."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained PixArt-α model or HuggingFace model id.",
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Revision of pretrained model.",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant of the model files (e.g. fp16).",
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Path to dataset directory (ImageFolder format with metadata.csv).",
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None,
        help="Training data directory (alternative to --dataset_name).",
    )
    parser.add_argument(
        "--image_column", type=str, default="image",
    )
    parser.add_argument(
        "--caption_column", type=str, default="text",
    )
    parser.add_argument(
        "--validation_prompt", type=str,
        default="a cardiac ultrasound b-plane image",
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=4,
    )
    parser.add_argument(
        "--validation_epochs", type=int, default=5,
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
    )
    parser.add_argument(
        "--output_dir", type=str, default="training/checkpoints/cetus_pixart_lora",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
    )
    parser.add_argument(
        "--center_crop", default=False, action="store_true",
    )
    parser.add_argument(
        "--random_flip", action="store_true",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=50,
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
    )
    parser.add_argument(
        "--scale_lr", action="store_true", default=False,
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100,
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true",
    )
    parser.add_argument(
        "--allow_tf32", action="store_true",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0,
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float,
    )
    parser.add_argument(
        "--proportion_empty_prompts", type=float, default=0,
    )
    parser.add_argument(
        "--prediction_type", type=str, default=None,
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to", type=str, default="tensorboard",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=500,
    )
    parser.add_argument(
        "--checkpointing_epochs", type=int, default=10,
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=5,
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0,
    )
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--train_text_encoder", action="store_true",
        help="Also train text encoder LoRA (disabled by default for fixed prompt).",
    )
    parser.add_argument(
        "--max_token_length", type=int, default=120,
    )
    parser.add_argument("--local-rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb for wandb logging.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    max_length = args.max_token_length

    # Mixed precision weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer, models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=weight_dtype,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    transformer = Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)

    # LoRA config for transformer
    lora_config = LoraConfig(
        r=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "proj", "linear", "linear_1", "linear_2",
        ],
    )

    transformer.to(accelerator.device)
    transformer = get_peft_model(transformer, lora_config)

    # Optionally train text encoder LoRA
    if args.train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)

    def cast_training_params(
        model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32
    ):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                if param.requires_grad:
                    param.data = param.to(dtype)

    if args.mixed_precision == "fp16":
        models_to_cast = [transformer]
        if args.train_text_encoder:
            models_to_cast.append(text_encoder)
        cast_training_params(models_to_cast, dtype=torch.float32)

    transformer.print_trainable_parameters()
    if args.train_text_encoder:
        text_encoder.print_trainable_parameters()

    # Checkpoint save/load hooks
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_ = accelerator.unwrap_model(transformer)
                lora_state = get_peft_model_state_dict(transformer_, adapter_name="default")
                PixArtLoraSaveDir = os.path.join(output_dir, "transformer_lora")
                transformer_.save_pretrained(PixArtLoraSaveDir)

                if args.train_text_encoder:
                    text_encoder_ = accelerator.unwrap_model(text_encoder)
                    te_lora_state = get_peft_model_state_dict(text_encoder_, adapter_name="default")
                    text_encoder_.save_pretrained(os.path.join(output_dir, "text_encoder_lora"))

                for _ in models:
                    weights.pop()

        def load_model_hook(models, input_dir):
            transformer_ = accelerator.unwrap_model(transformer)
            transformer_.load_adapter(
                os.path.join(input_dir, "transformer_lora"), "default", is_trainable=True,
            )

            if args.train_text_encoder:
                text_encoder_ = accelerator.unwrap_model(text_encoder)
                text_encoder_.load_adapter(
                    os.path.join(input_dir, "text_encoder_lora"), "default", is_trainable=True,
                )

            for _ in range(len(models)):
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    # Collect trainable parameters
    lora_layers = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        lora_layers += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load dataset
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    image_column = args.image_column
    caption_column = args.caption_column

    def tokenize_captions(examples, max_length=120):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(caption[0])
            else:
                raise ValueError(f"Unexpected caption type: {type(caption)}")
        inputs = tokenizer(
            captions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids, inputs.attention_mask

    train_transforms = transforms.Compose([
        transforms.Resize(
            args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip()
        if args.random_flip
        else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["text"] = examples[caption_column]
        examples["input_ids"], examples["prompt_attention_mask"] = tokenize_captions(
            examples, max_length=max_length,
        )
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        text = [example["text"] for example in examples]
        input_ids = torch.stack([example["input_ids"] for example in examples])
        prompt_attention_mask = torch.stack(
            [example["prompt_attention_mask"] for example in examples]
        )
        return {
            "pixel_values": pixel_values,
            "text": text,
            "input_ids": input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare with accelerator
    models_to_prepare = [transformer, optimizer, train_dataloader, lr_scheduler]
    if args.train_text_encoder:
        models_to_prepare.insert(1, text_encoder)
        transformer, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(*models_to_prepare)
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(*models_to_prepare)
        )

    # Recalculate steps after prepare
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("cetus-pixart-lora", config=vars(args))

    # Training
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    def save_checkpoint(step):
        if accelerator.is_main_process:
            if args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    for ckpt in checkpoints[:num_to_remove]:
                        shutil.rmtree(os.path.join(args.output_dir, ckpt))
                        logger.info(f"Removed old checkpoint: {ckpt}")

            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            accelerator.save_state(save_path)

            unwrapped_transformer = accelerator.unwrap_model(
                transformer, keep_fp32_wrapper=False
            )
            unwrapped_transformer.save_pretrained(
                os.path.join(save_path, "transformer_lora")
            )

            if args.train_text_encoder:
                unwrapped_text_encoder = accelerator.unwrap_model(
                    text_encoder, keep_fp32_wrapper=False
                )
                unwrapped_text_encoder.save_pretrained(
                    os.path.join(save_path, "text_encoder_lora")
                )

            logger.info(f"Saved state to {save_path}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()

                # Forward diffusion
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning
                if args.train_text_encoder:
                    prompt_embeds = text_encoder(
                        batch["input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                    )[0]
                else:
                    with torch.no_grad():
                        prompt_embeds = text_encoder(
                            batch["input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                        )[0]

                prompt_attention_mask = batch["prompt_attention_mask"]

                # Target
                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # No micro-conditions (512 model only)
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

                # Predict noise
                model_pred = transformer(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample.chunk(2, 1)[0]

                # Simple MSE loss (no color distribution loss)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    # NaN gradient check
                    grads_finite = True
                    for param in params_to_clip:
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grads_finite = False
                            break
                    if not grads_finite:
                        logger.warning("NaN/Inf gradients detected. Skipping step.")
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # Epoch-level checkpointing
        if args.checkpointing_epochs is not None and (epoch + 1) % args.checkpointing_epochs == 0:
            save_checkpoint(global_step)

        # Validation
        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and (epoch + 1) % args.validation_epochs == 0
            ):
                logger.info(
                    f"Running validation... Generating {args.num_validation_images} images "
                    f"with prompt: {args.validation_prompt}"
                )
                pipeline_kwargs = {
                    "transformer": accelerator.unwrap_model(
                        transformer, keep_fp32_wrapper=False
                    ),
                    "vae": vae,
                    "torch_dtype": weight_dtype,
                }
                if args.train_text_encoder:
                    pipeline_kwargs["text_encoder"] = accelerator.unwrap_model(
                        text_encoder, keep_fp32_wrapper=False
                    )

                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    **pipeline_kwargs,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)

                images = []
                with torch.no_grad():
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(
                                args.validation_prompt,
                                num_inference_steps=20,
                                generator=generator,
                            ).images[0]
                        )

                for i, image in enumerate(images):
                    image.save(
                        os.path.join(args.output_dir, f"val_image_{epoch}_{i}.png")
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log({
                            "validation": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        })

                del pipeline
                torch.cuda.empty_cache()

    # Save final LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_transformer = accelerator.unwrap_model(
            transformer, keep_fp32_wrapper=False
        )
        final_transformer.save_pretrained(
            os.path.join(args.output_dir, "transformer_lora")
        )

        if args.train_text_encoder:
            final_text_encoder = accelerator.unwrap_model(
                text_encoder, keep_fp32_wrapper=False
            )
            final_text_encoder.save_pretrained(
                os.path.join(args.output_dir, "text_encoder_lora")
            )

        logger.info(f"Final LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
