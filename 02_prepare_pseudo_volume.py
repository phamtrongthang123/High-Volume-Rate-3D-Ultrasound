"""
02_prepare_pseudo_volume.py — Load real 3D CETUS volumes for reconstruction.

Loads actual 3D cardiac echo volumes from the CETUS NIfTI dataset
(data/CETUS/dataset/patientXX/). Two volumes are created from the
End-Diastole (ED) and End-Systole (ES) cardiac phases of a single patient.
ED and ES represent the same anatomy at maximum and minimum ventricular volume,
matching SeqDiff's assumption that consecutive volumes share structural anatomy.

Each volume is resized to (N_el, N_az, N_ax, 1) = (112, 112, 112, 1) using
trilinear interpolation to match the diffusion model's B-plane input shape.
Unlike the previous CAMUS approach, B-planes vary realistically across
elevation — this is genuine 3D echocardiography data.

Note: the paper uses B-mode data in polar coordinates with N_el=48, N_az=64,
N_ax=400. Here we use N_el=112 planes of 112x112 so B-planes are (112, 112, 1)
matching the diffusion model's input shape.
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import jax
import jax.numpy as jnp
import SimpleITK as sitk
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.func import translate

# --- Config ---
PATIENT_ID = "patient01"
N_AZ = 112  # Number of azimuth B-planes per volume
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CETUS", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Init device ---
init_device(verbose=False)

# --- Load model to get input shape ---
print("Loading model to determine input shape...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape[:2]  # (H, W)
H, W = img_shape
print(f"Model input shape: {model.input_shape}, image shape: {img_shape}")


def load_cetus_volume(patient_id, phase):
    """Load a CETUS NIfTI volume and return as float32 numpy array (Z, Y, X)."""
    path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_{phase}.nii.gz")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr


def process_volume(arr_3d):
    """Resize 3D volume to (N_el, N_az, N_ax, 1) and normalize to [-1, 1].

    Axis mapping after SimpleITK GetArrayFromImage (Z, Y, X):
      axis 0 (Z) = depth from probe  → N_el  (elevation rows within each B-plane)
      axis 1 (Y) = azimuth sweep     → N_az  (B-plane index)
      axis 2 (X) = lateral           → N_ax  (columns within each B-plane)
    B-plane j = vol[:, j, :, 0], shape (N_el, N_ax) — apical 4-chamber appearance.
    """
    # Add channel dim → (Z, Y, X, 1), resize to (H, N_AZ, W, 1)
    vol = arr_3d[..., np.newaxis]
    vol = jax.image.resize(vol, (H, N_AZ, W, 1), method="linear")
    # Normalize to [-1, 1]: always use actual data range
    vmin = float(arr_3d.min())
    vmax = float(arr_3d.max())
    vol = translate(vol, (vmin, vmax), (-1, 1))
    return np.array(vol)


# --- Load CETUS volumes ---
print(f"Loading CETUS patient: {PATIENT_ID}")
arr_ed = load_cetus_volume(PATIENT_ID, "ED")
arr_es = load_cetus_volume(PATIENT_ID, "ES")
print(f"ED raw shape: {arr_ed.shape}, dtype: {arr_ed.dtype}, range: [{arr_ed.min():.1f}, {arr_ed.max():.1f}]")
print(f"ES raw shape: {arr_es.shape}, dtype: {arr_es.dtype}, range: [{arr_es.min():.1f}, {arr_es.max():.1f}]")

print("Processing volumes (resize + normalize)...")
volume_t1 = process_volume(arr_ed)  # ED phase → t1
volume_t2 = process_volume(arr_es)  # ES phase → t2

print(f"Volume t1 (ED) shape: {volume_t1.shape}")
print(f"Volume t2 (ES) shape: {volume_t2.shape}")

# --- Save ---
path_t1 = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
path_t2 = os.path.join(OUTPUT_DIR, "pseudo_volume_t2.npy")
np.save(path_t1, volume_t1)
np.save(path_t2, volume_t2)
print(f"Saved volume t1 (ED) to {path_t1}")
print(f"Saved volume t2 (ES) to {path_t2}")

# --- Sanity check ---
assert volume_t1.shape == (H, N_AZ, W, 1), f"Unexpected t1 shape: {volume_t1.shape}"
assert volume_t2.shape == (H, N_AZ, W, 1), f"Unexpected t2 shape: {volume_t2.shape}"
print(f"\n[Sanity check] t1 range: [{volume_t1.min():.3f}, {volume_t1.max():.3f}]")
print(f"[Sanity check] t2 range: [{volume_t2.min():.3f}, {volume_t2.max():.3f}]")
print("[Sanity check] PASSED: volumes loaded with real 3D variation across B-planes.")

print("Volume preparation complete.")
