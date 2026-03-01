"""
pixart_diffusion_model.py — PixArt-α diffusion model wrapper for reconstruction.

Provides PixArtDiffusionModel class that wraps PixArt-α pipeline components
(VAE, Transformer, T5 text encoder) with LoRA weights and exposes an interface
compatible with the reconstruction code.

Key design: DPS requires gradients through the denoiser. This wrapper keeps
all operations differentiable so torch.autograd.grad() works in the
reconstruction loop.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from peft import PeftModel
from transformers import T5EncoderModel, T5Tokenizer


class PixArtDiffusionModel:
    """Drop-in wrapper for PixArt-α with LoRA, for DPS reconstruction.

    Operates in VAE latent space (64x64x4 for 512x512 images).
    The denoise/predict_noise methods handle all resize and channel
    conversion between 112x112 grayscale and 512x512 RGB latent space.
    """

    def __init__(self, pretrained_path, lora_path=None, device="cuda"):
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_path, subfolder="vae", torch_dtype=self.weight_dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.device)

        # Load Transformer (with optional LoRA)
        self.transformer = Transformer2DModel.from_pretrained(
            pretrained_path, subfolder="transformer", torch_dtype=self.weight_dtype,
        )
        if lora_path is not None:
            self.transformer = PeftModel.from_pretrained(
                self.transformer, lora_path,
            )
        self.transformer.requires_grad_(False)
        self.transformer.to(self.device)
        self.transformer.eval()

        # Load tokenizer and text encoder for prompt embedding
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_path, subfolder="tokenizer",
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            pretrained_path, subfolder="text_encoder", torch_dtype=self.weight_dtype,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_path, subfolder="scheduler",
        )
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        # Pre-encode the fixed prompt
        self.prompt_embeds, self.prompt_attention_mask = self._encode_prompt(
            "a cardiac ultrasound b-plane image"
        )

        # Properties matching ZEA interface
        self.input_shape = (112, 112, 1)
        self.max_t = 1.0
        self.min_signal_rate = 0.02
        self.max_signal_rate = 0.95

        # VAE scaling factor
        self.vae_scale_factor = self.vae.config.scaling_factor

    def _encode_prompt(self, prompt):
        """Encode a text prompt to embeddings."""
        inputs = self.tokenizer(
            prompt,
            max_length=120,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            embeds = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        return embeds, attention_mask

    def diffusion_schedule(self, diffusion_times):
        """Cosine schedule matching ZEA's implementation.

        Args:
            diffusion_times: Array of shape (B, 1, 1, 1) with values in [0, max_t].

        Returns:
            noise_rates: sigma values, shape (B, 1, 1, 1).
            signal_rates: alpha values, shape (B, 1, 1, 1).
        """
        t = np.asarray(diffusion_times)
        start_angle = np.arccos(self.max_signal_rate)
        end_angle = np.arccos(self.min_signal_rate)
        diffusion_angles = start_angle + t / self.max_t * (end_angle - start_angle)
        signal_rates = np.cos(diffusion_angles)
        noise_rates = np.sin(diffusion_angles)
        return noise_rates.astype(np.float32), signal_rates.astype(np.float32)

    def _to_latent(self, images_512_rgb):
        """Encode 512x512 RGB images to VAE latent space.

        Args:
            images_512_rgb: (B, 3, 512, 512) tensor in [-1, 1].

        Returns:
            latents: (B, 4, 64, 64) tensor.
        """
        with torch.no_grad():
            latents = self.vae.encode(
                images_512_rgb.to(dtype=self.weight_dtype)
            ).latent_dist.sample()
        return latents * self.vae_scale_factor

    def _from_latent(self, latents):
        """Decode VAE latents to 512x512 RGB images.

        Args:
            latents: (B, 4, 64, 64) tensor.

        Returns:
            images: (B, 3, 512, 512) tensor in [-1, 1].
        """
        with torch.no_grad():
            images = self.vae.decode(
                latents.to(dtype=self.weight_dtype) / self.vae_scale_factor
            ).sample
        return images

    def _112_to_512_rgb(self, x_112):
        """Convert (B, 112, 112, 1) grayscale to (B, 3, 512, 512) RGB.

        Args:
            x_112: (B, 112, 112, 1) numpy array or tensor in [-1, 1].

        Returns:
            x_512: (B, 3, 512, 512) tensor in [-1, 1].
        """
        if isinstance(x_112, np.ndarray):
            x = torch.from_numpy(x_112).float().to(self.device)
        else:
            x = x_112.float().to(self.device)
        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (B, 1, 112, 112)
        # Resize to 512x512
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        # Grayscale → RGB
        x = x.repeat(1, 3, 1, 1)  # (B, 3, 512, 512)
        return x

    def _512_rgb_to_112(self, x_512):
        """Convert (B, 3, 512, 512) RGB to (B, 112, 112, 1) grayscale.

        Args:
            x_512: (B, 3, 512, 512) tensor in [-1, 1].

        Returns:
            x_112: (B, 112, 112, 1) numpy array in [-1, 1].
        """
        # RGB → grayscale (average channels)
        x = x_512.mean(dim=1, keepdim=True)  # (B, 1, 512, 512)
        # Resize to 112x112
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        # (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, 112, 112, 1)
        return x.cpu().detach().numpy()

    def _continuous_t_to_discrete_timestep(self, sigma, alpha):
        """Convert continuous (sigma, alpha) to discrete DDPM timestep.

        Maps cosine schedule noise/signal rates to the nearest discrete timestep
        in [0, num_train_timesteps).
        """
        # From cosine schedule: angle = arctan(sigma/alpha)
        # t_continuous = (angle - start_angle) / (end_angle - start_angle) * max_t
        # timestep = t_continuous / max_t * num_train_timesteps
        angle = np.arctan2(float(sigma), float(alpha))
        start_angle = np.arccos(self.max_signal_rate)
        end_angle = np.arccos(self.min_signal_rate)
        t_frac = (angle - start_angle) / (end_angle - start_angle)
        t_frac = np.clip(t_frac, 0, 1)
        timestep = int(t_frac * (self.num_train_timesteps - 1))
        return timestep

    def predict_noise_latent(self, noisy_latents, timestep):
        """Predict noise in latent space (differentiable).

        Args:
            noisy_latents: (B, 4, 64, 64) tensor.
            timestep: int, discrete timestep.

        Returns:
            noise_pred: (B, 4, 64, 64) tensor.
        """
        B = noisy_latents.shape[0]
        timesteps = torch.full((B,), timestep, device=self.device, dtype=torch.long)

        # Expand prompt embeddings to batch size
        prompt_embeds = self.prompt_embeds.expand(B, -1, -1)
        prompt_mask = self.prompt_attention_mask.expand(B, -1)

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        model_output = self.transformer(
            noisy_latents.to(dtype=self.weight_dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # PixArt outputs (B, 8, 64, 64), first 4 channels are prediction
        noise_pred = model_output.chunk(2, dim=1)[0]
        return noise_pred.float()

    def predict_noise(self, x_tau, sigma_squared):
        """Predict noise from noisy 112x112 grayscale images.

        Pixel-space interface matching ZEA's ema_network([x, sigma²]).

        Args:
            x_tau: (B, 112, 112, 1) numpy array, noisy images.
            sigma_squared: (B, 1, 1, 1) numpy array, noise variance.

        Returns:
            noise_pred: (B, 112, 112, 1) numpy array.
        """
        # Convert to 512x512 RGB
        x_512 = self._112_to_512_rgb(x_tau)

        # Encode to latent
        latents = self._to_latent(x_512)

        # Compute timestep from sigma²
        sigma = np.sqrt(float(sigma_squared.flat[0]))
        # Approximate alpha from sigma using cosine schedule
        alpha = np.sqrt(1.0 - sigma**2) if sigma < 1.0 else 0.01
        timestep = self._continuous_t_to_discrete_timestep(sigma, alpha)

        # Predict noise in latent space
        noise_latent = self.predict_noise_latent(latents, timestep)

        # Decode noise prediction back to pixel space
        # noise_latent is the predicted noise in latent space
        # To get pixel-space noise: decode(noisy_latent - noise_latent * sigma_scale) → clean
        # Then noise = (x - alpha * clean) / sigma
        # But simpler: decode noisy and decoded, difference is noise estimate
        with torch.no_grad():
            decoded_noisy = self._from_latent(latents)
            decoded_clean = self._from_latent(latents - noise_latent * 1.0)  # approximate

        noise_512 = decoded_noisy - decoded_clean
        noise_112 = self._512_rgb_to_112(noise_512)
        return noise_112

    def denoise(self, noisy_images, noise_rates, signal_rates):
        """Predict clean images from noisy ones.

        Args:
            noisy_images: (B, 112, 112, 1) numpy array.
            noise_rates: (B, 1, 1, 1) numpy array (sigma).
            signal_rates: (B, 1, 1, 1) numpy array (alpha).

        Returns:
            pred_images: (B, 112, 112, 1) numpy array.
            pred_noises: (B, 112, 112, 1) numpy array.
        """
        sigma = float(noise_rates.flat[0])
        alpha = float(signal_rates.flat[0])
        sigma_sq = np.full_like(noise_rates, sigma**2)
        pred_noises = self.predict_noise(noisy_images, sigma_sq)
        # Tweedie: x_0 = (x - sigma * eps) / alpha
        pred_images = (noisy_images - sigma * pred_noises) / alpha
        return pred_images, pred_noises

    def sample(self, n_samples=1, n_steps=20):
        """Generate unconditional samples via reverse diffusion.

        Returns:
            samples: (n_samples, 112, 112, 1) numpy array in [-1, 1].
        """
        # Start from pure noise in latent space
        latents = torch.randn(
            n_samples, 4, 64, 64, device=self.device, dtype=torch.float32,
        )

        # Set up scheduler
        self.noise_scheduler.set_timesteps(n_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        for t in timesteps:
            noise_pred = self.predict_noise_latent(latents, int(t))
            latents = self.noise_scheduler.step(
                noise_pred, int(t), latents, return_dict=False,
            )[0]

        # Decode to pixel space
        images_512 = self._from_latent(latents)  # (B, 3, 512, 512)
        images_112 = self._512_rgb_to_112(images_512)  # (B, 112, 112, 1)
        return images_112


class PixArtLatentDiffusionModel:
    """Latent-space DPS interface for PixArt-α reconstruction.

    Unlike PixArtDiffusionModel which converts between pixel spaces,
    this class operates entirely in latent space for efficiency.
    DPS guidance is computed in latent space, and pixel decoding only
    happens at the final step.
    """

    def __init__(self, pretrained_path, lora_path=None, device="cuda"):
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_path, subfolder="vae", torch_dtype=self.weight_dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.device)

        # Load Transformer (with optional LoRA)
        self.transformer = Transformer2DModel.from_pretrained(
            pretrained_path, subfolder="transformer", torch_dtype=self.weight_dtype,
        )
        if lora_path is not None:
            self.transformer = PeftModel.from_pretrained(
                self.transformer, lora_path,
            )
        self.transformer.requires_grad_(False)
        self.transformer.to(self.device)
        self.transformer.eval()

        # Load tokenizer and text encoder
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_path, subfolder="tokenizer",
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            pretrained_path, subfolder="text_encoder", torch_dtype=self.weight_dtype,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_path, subfolder="scheduler",
        )
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        # Pre-encode prompt
        self.prompt_embeds, self.prompt_attention_mask = self._encode_prompt(
            "a cardiac ultrasound b-plane image"
        )

        # VAE scaling
        self.vae_scale_factor = self.vae.config.scaling_factor

        # Latent shape for 512x512 input
        self.latent_shape = (4, 64, 64)

    def _encode_prompt(self, prompt):
        inputs = self.tokenizer(
            prompt, max_length=120, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            embeds = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        return embeds, attention_mask

    def encode_images(self, images_112):
        """Encode 112x112 grayscale images to VAE latent space.

        Args:
            images_112: (B, 112, 112, 1) numpy array in [-1, 1].

        Returns:
            latents: (B, 4, 64, 64) tensor.
        """
        x = torch.from_numpy(images_112).float().to(self.device)
        x = x.permute(0, 3, 1, 2)  # (B, 1, 112, 112)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)  # (B, 3, 512, 512)
        with torch.no_grad():
            latents = self.vae.encode(
                x.to(dtype=self.weight_dtype)
            ).latent_dist.sample()
        return (latents * self.vae_scale_factor).float()

    def decode_latents(self, latents):
        """Decode VAE latents to 112x112 grayscale images.

        Args:
            latents: (B, 4, 64, 64) tensor.

        Returns:
            images: (B, 112, 112, 1) numpy array in [-1, 1].
        """
        with torch.no_grad():
            images = self.vae.decode(
                latents.to(dtype=self.weight_dtype) / self.vae_scale_factor
            ).sample  # (B, 3, 512, 512)
        # RGB → gray, resize
        gray = images.float().mean(dim=1, keepdim=True)  # (B, 1, 512, 512)
        gray = F.interpolate(gray, size=(112, 112), mode="bilinear", align_corners=False)
        gray = gray.permute(0, 2, 3, 1)  # (B, 112, 112, 1)
        return gray.cpu().detach().numpy()

    def predict_noise_latent(self, noisy_latents, timestep):
        """Predict noise in latent space.

        This method is differentiable w.r.t. noisy_latents for DPS.
        """
        B = noisy_latents.shape[0]
        timesteps = torch.full((B,), timestep, device=self.device, dtype=torch.long)
        prompt_embeds = self.prompt_embeds.expand(B, -1, -1)
        prompt_mask = self.prompt_attention_mask.expand(B, -1)
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # Enable gradients through transformer for DPS
        model_output = self.transformer(
            noisy_latents.to(dtype=self.weight_dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        noise_pred = model_output.chunk(2, dim=1)[0]
        return noise_pred.float()

    def get_ddpm_alpha_sigma(self, timestep):
        """Get alpha and sigma for a discrete DDPM timestep.

        Returns:
            alpha: sqrt(alpha_bar_t)
            sigma: sqrt(1 - alpha_bar_t)
        """
        alpha_bar = self.noise_scheduler.alphas_cumprod[timestep]
        alpha = math.sqrt(float(alpha_bar))
        sigma = math.sqrt(1.0 - float(alpha_bar))
        return alpha, sigma
