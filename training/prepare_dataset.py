# -*- coding: utf-8 -*-
"""
prepare_dataset.py — Extract CETUS B-planes as 512x512 pseudo-RGB images.

Loads each patient's ED and ES volumes from CETUS NIfTI dataset,
resizes to (112, 112, 112), extracts azimuth slices, resizes to 512x512,
and saves as PNG with metadata.csv for HuggingFace datasets ImageFolder.

Split: patients 1-40 → train, patients 41-45 → val.
"""

import csv
import os
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "CETUS", "dataset")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "dataset", "train")
VAL_DIR = os.path.join(SCRIPT_DIR, "dataset", "val")

N_TARGET = 112  # Target volume size per axis
IMG_SIZE = 512  # Output image resolution
PHASES = ["ED", "ES"]
TRAIN_PATIENTS = list(range(1, 41))  # patients 1-40
VAL_PATIENTS = list(range(41, 46))  # patients 41-45
CAPTION = "a cardiac ultrasound b-plane image"


def load_cetus_volume(patient_id, phase):
    """Load a CETUS NIfTI volume and return as float32 numpy array (Z, Y, X)."""
    path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_{phase}.nii.gz")
    if not os.path.exists(path):
        return None
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr


def resize_volume(arr, target_shape):
    """Resize 3D volume to target_shape using SimpleITK."""
    img = sitk.GetImageFromArray(arr)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(s) for s in target_shape[::-1]])  # SimpleITK uses (X, Y, Z)
    resampler.SetOutputSpacing(
        [float(old / new) for old, new in zip(img.GetSize(), target_shape[::-1])]
    )
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampled = resampler.Execute(img)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def process_patient(patient_num, output_dir, metadata_rows):
    """Extract all B-planes from a patient's ED and ES volumes."""
    patient_id = f"patient{patient_num:02d}"
    count = 0

    for phase in PHASES:
        arr = load_cetus_volume(patient_id, phase)
        if arr is None:
            print(f"  Warning: {patient_id}_{phase}.nii.gz not found, skipping")
            continue

        # Resize to (N_TARGET, N_TARGET, N_TARGET)
        vol = resize_volume(arr, (N_TARGET, N_TARGET, N_TARGET))

        # Normalize to [0, 255]
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-8:
            print(f"  Warning: {patient_id}_{phase} has constant values, skipping")
            continue
        vol_norm = (vol - vmin) / (vmax - vmin) * 255.0

        # Extract azimuth slices: bplane = vol[:, j, :] → shape (N_TARGET, N_TARGET)
        for j in range(N_TARGET):
            bplane = vol_norm[:, j, :]  # (112, 112)

            # Resize to 512x512
            pil_img = Image.fromarray(bplane.astype(np.uint8), mode="L")
            pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            # Convert to RGB
            pil_rgb = pil_img.convert("RGB")

            # Save
            fname = f"{patient_id}_{phase}_az{j:03d}.png"
            pil_rgb.save(os.path.join(output_dir, fname))
            metadata_rows.append({"file_name": fname, "text": CAPTION})
            count += 1

    return count


def main():
    # Check data exists
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: CETUS dataset not found at {DATA_DIR}")
        print("Run 'python 00_download_data.py' first.")
        sys.exit(1)

    # Create output directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Process training patients
    print(f"Processing training patients (1-40) → {TRAIN_DIR}")
    train_metadata = []
    train_total = 0
    for p in TRAIN_PATIENTS:
        n = process_patient(p, TRAIN_DIR, train_metadata)
        train_total += n
        if n > 0:
            print(f"  patient{p:02d}: {n} images")

    # Write train metadata.csv
    train_csv = os.path.join(TRAIN_DIR, "metadata.csv")
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(train_metadata)
    print(f"Train: {train_total} images, metadata → {train_csv}")

    # Process validation patients
    print(f"\nProcessing validation patients (41-45) → {VAL_DIR}")
    val_metadata = []
    val_total = 0
    for p in VAL_PATIENTS:
        n = process_patient(p, VAL_DIR, val_metadata)
        val_total += n
        if n > 0:
            print(f"  patient{p:02d}: {n} images")

    # Write val metadata.csv
    val_csv = os.path.join(VAL_DIR, "metadata.csv")
    with open(val_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(val_metadata)
    print(f"Val: {val_total} images, metadata → {val_csv}")

    print(f"\nDataset preparation complete: {train_total + val_total} total images")


if __name__ == "__main__":
    main()
