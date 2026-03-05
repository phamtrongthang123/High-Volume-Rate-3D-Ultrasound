"""
prepare_volume.py — Load a CETUS volume for PixArt pipeline (pure PyTorch, no JAX).

Usage:
    python prepare_volume.py --patient-id patient01 --phase ED --output-dir ../outputs/patient01_ED

Loads a CETUS NIfTI volume, resizes to (112, 112, 112) via scipy trilinear
interpolation, normalizes to [-1, 1], and saves as pseudo_volume.npy.
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "CETUS", "dataset")
TARGET_SHAPE = (112, 112, 112)  # (N_el, N_az, N_ax)


def load_cetus_volume(patient_id, phase):
    path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_{phase}.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CETUS volume not found: {path}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr


def process_volume(arr_3d):
    """Resize to TARGET_SHAPE and normalize to [-1, 1].

    CETUS axis mapping after SimpleITK (Z, Y, X):
        Z = depth from probe  → N_el (elevation)
        Y = azimuth sweep     → N_az (B-plane index)
        X = lateral           → N_ax (columns within B-plane)
    """
    zoom_factors = tuple(t / s for t, s in zip(TARGET_SHAPE, arr_3d.shape))
    resized = zoom(arr_3d, zoom_factors, order=1)  # trilinear (order=1)
    vmin, vmax = resized.min(), resized.max()
    if vmax > vmin:
        normalized = 2.0 * (resized - vmin) / (vmax - vmin) - 1.0
    else:
        normalized = np.zeros_like(resized)
    return normalized[..., np.newaxis].astype(np.float32)  # (112, 112, 112, 1)


def main():
    parser = argparse.ArgumentParser(description="Prepare CETUS volume for PixArt pipeline")
    parser.add_argument("--patient-id", default="patient01", help="CETUS patient ID (e.g. patient01)")
    parser.add_argument("--phase", default="ED", choices=["ED", "ES"], help="Cardiac phase")
    parser.add_argument("--output-dir", required=True, help="Output directory for pseudo_volume.npy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading CETUS {args.patient_id} {args.phase}...")
    arr = load_cetus_volume(args.patient_id, args.phase)
    print(f"  Raw shape: {arr.shape}, range: [{arr.min():.1f}, {arr.max():.1f}]")

    print(f"  Resizing to {TARGET_SHAPE} (trilinear)...")
    vol = process_volume(arr)
    print(f"  Output shape: {vol.shape}, range: [{vol.min():.3f}, {vol.max():.3f}]")

    save_path = os.path.join(args.output_dir, "pseudo_volume.npy")
    np.save(save_path, vol)
    print(f"  Saved: {save_path}")

    assert vol.shape == (112, 112, 112, 1), f"Unexpected shape: {vol.shape}"
    print("Done.")


if __name__ == "__main__":
    main()
