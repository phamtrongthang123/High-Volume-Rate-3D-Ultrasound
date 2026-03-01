"""
08_visualize_straus.py — STRAUS dataset visualization.

Generates five figures:

1. 08_straus_overview.png
   Grid of center axial slices for 6 representative patients (one per
   pathology group), frame 0 vs frame 15.

2. 08_straus_orthogonal.png
   Three-plane (axial/sagittal/coronal) view of patient01 US frame 0.

3. 08_straus_bplane_strip.png
   Strip of sampled B-planes across the azimuth axis for patient01 frame 0.

4. 08_straus_temporal.png
   Temporal strip: same center axial slice across 8 evenly-spaced frames
   to show cardiac motion through the cycle (patient01).

5. 08_straus_statistics.png
   Voxel intensity (mean ± std) and volume shape across patients,
   grouped/colored by pathology.

No diffusion model or JAX required — pure SimpleITK + NumPy + Matplotlib.
"""

import env_setup  # noqa: F401 — must be first

import os
import re
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "STRAUS")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_PATIENT = "patient01_healthy"
N_BPLANE_COLS = 8
N_TEMPORAL_COLS = 8

# Pathology colour map for statistics
PATHOLOGY_COLORS = {
    "healthy": "steelblue",
    "lbbb": "coral",
    "lcx": "mediumseagreen",
    "lad": "orchid",
    "rca": "goldenrod",
    "rca2": "mediumpurple",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_patient_dirs():
    """Scan DATA_DIR for patient* directories.

    Returns list of (dirname, pathology) sorted by patient number.
    """
    entries = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.startswith("patient"):
            continue
        full = os.path.join(DATA_DIR, name)
        if not os.path.isdir(full):
            continue
        # Parse pathology from folder name: patientXX_<pathology>
        m = re.match(r"patient\d+_(\w+)", name)
        pathology = m.group(1) if m else "unknown"
        entries.append((name, pathology))
    return entries


def load_us_volume(patient_dir, frame_idx):
    """Load us/image/usfrmXX.mhd → float32 numpy array (Z, Y, X) + spacing."""
    fname = f"usfrm{frame_idx:02d}.mhd"
    path = os.path.join(DATA_DIR, patient_dir, "us", "image", fname)
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr, img.GetSpacing()  # spacing: (X, Y, Z) in mm


def center_slice(arr, axis):
    """Return center slice along the given axis as 2D array."""
    idx = arr.shape[axis] // 2
    return np.take(arr, idx, axis=axis)


def normalize_for_display(arr):
    """Normalize array to [0, 1] for display."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def get_representative_patients(patient_list):
    """Pick one patient per pathology group for the overview grid."""
    seen = {}
    for dirname, pathology in patient_list:
        if pathology not in seen:
            seen[pathology] = (dirname, pathology)
    # Return in the order pathologies first appear
    return list(seen.values())


# ---------------------------------------------------------------------------
# Figure 1 — Multi-patient overview
# ---------------------------------------------------------------------------

def make_overview(patient_list):
    """Grid: rows=representative patients (1 per pathology),
    cols=frame 0 + frame 15, center axial slice."""
    reps = get_representative_patients(patient_list)
    frames = [0, 15]
    frame_labels = ["Frame 0 (start)", "Frame 15 (mid-cycle)"]

    n_rows = len(reps)
    n_cols = len(frames)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    fig.suptitle(
        "STRAUS Dataset — Center Axial Slice Overview\n"
        "(one patient per pathology, frame 0 vs frame 15)",
        fontsize=13, y=1.01,
    )

    for r, (dirname, pathology) in enumerate(reps):
        for c, frame_idx in enumerate(frames):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            try:
                arr, _ = load_us_volume(dirname, frame_idx)
                sl = center_slice(arr, axis=0)
                ax.imshow(normalize_for_display(sl), cmap="gray",
                          origin="upper", aspect="equal")
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes)
            if r == 0:
                ax.set_title(frame_labels[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{dirname}\n({pathology})", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "08_straus_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview to {save_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Orthogonal views
# ---------------------------------------------------------------------------

def make_orthogonal(patient_dir, frame_idx=0):
    """Three-plane view (axial / coronal / sagittal) — no segmentation."""
    arr, spacing = load_us_volume(patient_dir, frame_idx)

    planes = [
        ("Axial (Z center)",    center_slice(arr, 0)),
        ("Coronal (Y center)",  center_slice(arr, 1)),
        ("Sagittal (X center)", center_slice(arr, 2)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"STRAUS {patient_dir} — Frame {frame_idx} — Orthogonal Views\n"
        f"Volume shape: {arr.shape}  |  "
        f"Spacing: ({spacing[2]:.2f}, {spacing[1]:.2f}, {spacing[0]:.2f}) mm (Z,Y,X)",
        fontsize=12,
    )

    for ax, (title, sl) in zip(axes, planes):
        ax.imshow(normalize_for_display(sl), cmap="gray",
                  origin="upper", aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "08_straus_orthogonal.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved orthogonal view to {save_path}")


# ---------------------------------------------------------------------------
# Figure 3 — B-plane strip across azimuth
# ---------------------------------------------------------------------------

def make_bplane_strip(patient_dir, frame_idx=0, n_cols=N_BPLANE_COLS):
    """Single row of B-planes sampled uniformly across the azimuth axis."""
    arr, _ = load_us_volume(patient_dir, frame_idx)

    N_AZ = arr.shape[1]
    indices = np.round(np.linspace(0, N_AZ - 1, n_cols)).astype(int)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3))
    fig.suptitle(
        f"STRAUS {patient_dir} — Frame {frame_idx} — "
        f"B-planes across Azimuth Axis\n"
        f"(N_az = {N_AZ} B-planes, showing {n_cols} uniformly sampled)",
        fontsize=12,
    )

    for col, j in enumerate(indices):
        sl = arr[:, j, :]  # shape (N_el, N_ax)
        axes[col].imshow(normalize_for_display(sl), cmap="gray",
                         origin="upper", aspect="equal")
        axes[col].set_title(f"az={j}", fontsize=8)
        axes[col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "08_straus_bplane_strip.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved B-plane strip to {save_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Temporal strip (unique to STRAUS)
# ---------------------------------------------------------------------------

def make_temporal_strip(patient_dir, n_cols=N_TEMPORAL_COLS):
    """Center axial slice across evenly-spaced temporal frames."""
    # Count available frames
    us_dir = os.path.join(DATA_DIR, patient_dir, "us", "image")
    n_frames = len([f for f in os.listdir(us_dir) if f.endswith(".mhd")])
    frame_indices = np.round(np.linspace(0, n_frames - 1, n_cols)).astype(int)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3))
    fig.suptitle(
        f"STRAUS {patient_dir} — Temporal Strip (center axial slice)\n"
        f"({n_frames} frames total, showing {n_cols} evenly spaced)",
        fontsize=12,
    )

    for col, fidx in enumerate(frame_indices):
        arr, _ = load_us_volume(patient_dir, fidx)
        sl = center_slice(arr, axis=0)
        axes[col].imshow(normalize_for_display(sl), cmap="gray",
                         origin="upper", aspect="equal")
        axes[col].set_title(f"frm {fidx}", fontsize=8)
        axes[col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "08_straus_temporal.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal strip to {save_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Statistics grouped by pathology
# ---------------------------------------------------------------------------

def make_statistics(patient_list):
    """Bar charts of voxel intensity and volume shape, colored by pathology."""
    stats = []
    for dirname, pathology in patient_list:
        try:
            arr, spacing = load_us_volume(dirname, 0)
            stats.append({
                "dirname": dirname,
                "pathology": pathology,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "z": arr.shape[0],
                "y": arr.shape[1],
                "x": arr.shape[2],
            })
        except Exception:
            pass

    if not stats:
        print("No STRAUS data found for statistics.")
        return

    labels = [s["dirname"].replace("patient", "P") for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    colors = [PATHOLOGY_COLORS.get(s["pathology"], "gray") for s in stats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("STRAUS Dataset Statistics (colored by pathology)", fontsize=13)
    x = np.arange(len(labels))
    w = 0.6

    # Panel 1: voxel intensity
    axes[0].bar(x, means, width=w, color=colors, alpha=0.8)
    axes[0].errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    axes[0].set_title("Voxel Intensity (mean ± std)")
    axes[0].set_ylabel("Intensity (raw)")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: volume depth (Z slices)
    axes[1].bar(x, [s["z"] for s in stats], width=w, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    axes[1].set_title("Number of Axial Slices (Z)")
    axes[1].set_ylabel("Slices")
    axes[1].grid(axis="y", alpha=0.3)

    # Legend for pathology colours
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=p, alpha=0.8)
                      for p, c in PATHOLOGY_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, "08_straus_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved statistics to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"STRAUS data directory: {DATA_DIR}")
    print(f"Output directory:      {OUTPUT_DIR}\n")

    patient_list = get_patient_dirs()
    print(f"Found {len(patient_list)} patients: "
          + ", ".join(f"{d} ({p})" for d, p in patient_list))
    print()

    print("=== Figure 1: Multi-patient overview ===")
    make_overview(patient_list)

    print("\n=== Figure 2: Orthogonal views ===")
    make_orthogonal(DETAIL_PATIENT, frame_idx=0)

    print("\n=== Figure 3: B-plane strip across azimuth ===")
    make_bplane_strip(DETAIL_PATIENT, frame_idx=0, n_cols=N_BPLANE_COLS)

    print("\n=== Figure 4: Temporal strip ===")
    make_temporal_strip(DETAIL_PATIENT, n_cols=N_TEMPORAL_COLS)

    print("\n=== Figure 5: Volume statistics ===")
    make_statistics(patient_list)

    print("\nVisualization complete.")
    print(f"All figures saved to {OUTPUT_DIR}/08_straus_*.png")
