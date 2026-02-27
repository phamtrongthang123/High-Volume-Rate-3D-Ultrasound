"""
02_prepare_pseudo_volume.py — Create pseudo-3D volumes from CAMUS dataset.

Loads 2D cardiac echo images from the CAMUS sample dataset (HuggingFace),
resizes to the diffusion model's input shape, and stacks them as
"elevation planes" to create pseudo-3D volumes X ∈ R^(N_el, N_az, N_ax)
(Eq. 1, eq:ultrasound-volume). Two volumes are created with a 1-plane offset to simulate consecutive
temporal frames for the SeqDiff demo (04). This is a demo hack — no real
temporal 3D data is available, so the offset fakes two similar-but-different
frames. With real 3D temporal captures you'd have a sequence of volumes
X(t=0), X(t=1), ... and SeqDiff would warm-start each frame from the
previous reconstruction.

Note: the paper uses B-mode data in polar coordinates with N_el=48, N_az=64,
N_ax=400. Here we use N_el=112 planes of 112x112 so B-planes are (112, 112, 1)
matching the diffusion model's input shape.
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import jax
import jax.numpy as jnp
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.data import Dataset
from zea.func import translate

# --- Config ---
N_ELEVATION = 112  # Number of planes per pseudo-volume (matches model input for B-plane slicing)
DYNAMIC_RANGE = (-50, 0)  # dB dynamic range
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Init device ---
init_device(verbose=False)

# --- Load model to get input shape ---
print("Loading model to determine input shape...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape[:2]  # (H, W)
print(f"Model input shape: {model.input_shape}, image shape: {img_shape}")

# --- Load CAMUS dataset ---
print("Loading CAMUS dataset...")
dataset = Dataset("hf://zeahub/camus-sample/val", key="image")

n_needed = N_ELEVATION + 1  # Need N+1 for two overlapping volumes
images = []
for i in range(min(n_needed, len(dataset))):
    img = dataset[i]["data"]["image"]  # shape: (n_frames, H, W) or (H, W)
    img = np.array(img)

    # If multi-frame, take first frame
    if img.ndim == 3:
        img = img[0]

    # Add channel dim: (H, W) -> (H, W, 1)
    img = img[..., np.newaxis]

    # Resize to model input shape
    img = jax.image.resize(img, (*img_shape, 1), method="bilinear")

    # Normalize: clip to dynamic range, then translate to [-1, 1]
    img = jnp.clip(img, DYNAMIC_RANGE[0], DYNAMIC_RANGE[1])
    img = translate(img, DYNAMIC_RANGE, (-1, 1))

    images.append(np.array(img))

print(f"Loaded and processed {len(images)} images")

n_original = len(images)
if n_original < N_ELEVATION + 1:
    print(f"Warning: only {n_original} images available, need {N_ELEVATION + 1}.")
    print("Cycling through available images to fill volumes.")
    while len(images) < N_ELEVATION + 1:
        images.append(images[len(images) % n_original])

processed = np.stack(images, axis=0)
print(f"Processed images shape: {processed.shape}, "
      f"range: [{processed.min():.3f}, {processed.max():.3f}]")

# --- Create pseudo-volumes ---
# Volume 1: first N_ELEVATION planes
volume_t1 = processed[:N_ELEVATION]
# Volume 2: offset by 1 for SeqDiff temporal demo
volume_t2 = processed[1:N_ELEVATION + 1]

print(f"Pseudo-volume t1 shape: {volume_t1.shape}")
print(f"Pseudo-volume t2 shape: {volume_t2.shape}")

# --- Save ---
path_t1 = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
path_t2 = os.path.join(OUTPUT_DIR, "pseudo_volume_t2.npy")
np.save(path_t1, volume_t1)
np.save(path_t2, volume_t2)
print(f"Saved pseudo-volume t1 to {path_t1}")
print(f"Saved pseudo-volume t2 to {path_t2}")

dataset.close()
print("Pseudo-volume preparation complete.")
