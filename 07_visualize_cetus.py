"""
07_visualize_cetus.py — CETUS dataset visualization.

Generates three figures:

1. 07_cetus_overview.png
   Grid of center axial slices for several patients, ED vs ES side-by-side.

2. 07_cetus_orthogonal.png
   Three-plane (axial/sagittal/coronal) view of patient01 ED with GT
   segmentation overlay.

3. 07_cetus_bplane_strip.png
   Strip of sampled B-planes across the elevation axis for patient01 ED
   to illustrate real anatomical variation in the 3D volume.

No diffusion model or JAX required — pure SimpleITK + NumPy + Matplotlib.
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "CETUS", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patients to include in the overview grid
OVERVIEW_PATIENTS = ["patient01", "patient02", "patient03", "patient04",
                     "patient05", "patient06"]
DETAIL_PATIENT = "patient01"

# Segmentation label colours (transparent background, then myo and lumen)
SEG_COLORS = [(0, 0, 0, 0), (1, 0.2, 0.2, 0.6), (0.2, 0.6, 1.0, 0.6)]
SEG_CMAP = ListedColormap(SEG_COLORS)

# Number of B-planes to sample for the strip figure
N_BPLANE_COLS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_volume(patient_id, phase):
    """Load NIfTI volume → float32 numpy array (Z, Y, X)."""
    path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_{phase}.nii.gz")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr, img.GetSpacing()  # spacing: (X, Y, Z) in mm


def load_segmentation(patient_id, phase):
    """Load GT segmentation NIfTI → int32 numpy array (Z, Y, X)."""
    path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_{phase}_gt.nii.gz")
    if not os.path.exists(path):
        return None
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.int32)


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


# ---------------------------------------------------------------------------
# Figure 1 — Multi-patient overview
# ---------------------------------------------------------------------------

def make_overview(patients, phases=("ED", "ES")):
    """Grid: rows=patients, cols=phases, center axial slice."""
    n_rows = len(patients)
    n_cols = len(phases)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    fig.suptitle("CETUS Dataset — Center Axial Slice Overview\n(ED = End-Diastole, ES = End-Systole)",
                 fontsize=13, y=1.01)

    for r, pid in enumerate(patients):
        for c, phase in enumerate(phases):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            try:
                arr, _ = load_volume(pid, phase)
                sl = center_slice(arr, axis=0)  # axial center (Z axis = depth from probe = N_el in pipeline)
                ax.imshow(normalize_for_display(sl), cmap="gray", origin="upper", aspect="equal")
                title = f"{pid}  {phase}" if r == 0 else phase
                ax.set_title(title if r == 0 else "", fontsize=9)
                if c == 0:
                    ax.set_ylabel(pid, fontsize=8)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "07_cetus_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview to {save_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Orthogonal views with segmentation overlay
# ---------------------------------------------------------------------------

def make_orthogonal(patient_id, phase="ED"):
    """Three-plane view (axial / sagittal / coronal) with GT seg overlay."""
    arr, spacing = load_volume(patient_id, phase)
    seg = load_segmentation(patient_id, phase)

    Z, Y, X = arr.shape
    labels = sorted(np.unique(seg)) if seg is not None else []
    n_labels = int(seg.max()) + 1 if seg is not None else 0

    planes = [
        ("Axial (Z center)",    center_slice(arr, 0), center_slice(seg, 0) if seg is not None else None),
        ("Coronal (Y center)",  center_slice(arr, 1), center_slice(seg, 1) if seg is not None else None),
        ("Sagittal (X center)", center_slice(arr, 2), center_slice(seg, 2) if seg is not None else None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"CETUS {patient_id} — {phase} phase — Orthogonal Views\n"
        f"Volume shape: {arr.shape}  |  Spacing: ({spacing[2]:.2f}, {spacing[1]:.2f}, {spacing[0]:.2f}) mm (Z,Y,X)",
        fontsize=12,
    )

    for ax, (title, sl, sl_seg) in zip(axes, planes):
        ax.imshow(normalize_for_display(sl), cmap="gray", origin="upper", aspect="equal")
        if sl_seg is not None and n_labels > 1:
            ax.imshow(sl_seg, cmap=SEG_CMAP, vmin=0, vmax=max(3, n_labels - 1),
                      origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Legend
    if seg is not None and n_labels > 1:
        label_names = {0: "Background", 1: "Myocardium", 2: "Lumen"}
        patches = [
            mpatches.Patch(color=SEG_COLORS[i], label=label_names.get(i, f"Label {i}"))
            for i in labels if i > 0 and i < len(SEG_COLORS)
        ]
        if patches:
            fig.legend(handles=patches, loc="lower center", ncol=len(patches),
                       fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, "07_cetus_orthogonal.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved orthogonal view to {save_path}")


# ---------------------------------------------------------------------------
# Figure 3 — B-plane strip across elevation
# ---------------------------------------------------------------------------

def make_bplane_strip(patient_id, phase="ED", n_cols=N_BPLANE_COLS):
    """
    Show a grid of B-planes sampled uniformly across the azimuth axis (axis 1).
    Also display the matching GT segmentation outline.

    A B-plane is defined as arr[:, j, :] — a fixed-azimuth (axis-1) slice of
    shape (Z, X) = (N_el, N_ax).  These are the actual 2D images fed to the
    diffusion model during reconstruction (each B-plane = one apical cross-section).

    NIfTI axis mapping after SimpleITK load:
      axis 0 (Z) = depth from probe  = N_el (elevation rows within each B-plane)
      axis 1 (Y) = azimuth sweep     = N_az (B-plane index)  ← we sweep here
      axis 2 (X) = lateral           = N_ax (columns within each B-plane)
    """
    arr, _ = load_volume(patient_id, phase)
    seg = load_segmentation(patient_id, phase)

    N_AZ = arr.shape[1]  # number of azimuth B-planes
    indices = np.round(np.linspace(0, N_AZ - 1, n_cols)).astype(int)

    # Two rows: raw volume B-planes (top) and segmentation (bottom, if available)
    n_rows = 2 if seg is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.6 * n_rows))
    fig.suptitle(
        f"CETUS {patient_id} — {phase} — B-planes across Azimuth Axis\n"
        f"(N_az = {N_AZ} B-planes, showing {n_cols} uniformly sampled)",
        fontsize=12,
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]  # make 2-D for uniform indexing

    for col, j in enumerate(indices):
        sl = arr[:, j, :]  # shape (N_el, N_ax) — the actual B-plane
        axes[0, col].imshow(normalize_for_display(sl), cmap="gray", origin="upper", aspect="equal")
        axes[0, col].set_title(f"az={j}", fontsize=8)
        axes[0, col].axis("off")

        if seg is not None:
            sl_seg = seg[:, j, :]  # same azimuth index in segmentation
            axes[1, col].imshow(normalize_for_display(sl), cmap="gray", origin="upper", aspect="equal")
            axes[1, col].imshow(sl_seg, cmap=SEG_CMAP, vmin=0, vmax=3,
                                origin="upper", aspect="equal", interpolation="nearest")
            axes[1, col].axis("off")

    if n_rows > 1:
        axes[0, 0].set_ylabel("Echo", fontsize=9, rotation=90, labelpad=4)
        axes[1, 0].set_ylabel("+ Seg", fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "07_cetus_bplane_strip.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved B-plane strip to {save_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Volume statistics summary
# ---------------------------------------------------------------------------

def make_statistics(patients):
    """
    Bar chart comparing voxel intensity range and volume shape across patients
    for ED and ES phases.
    """
    stats = []
    for pid in patients:
        for phase in ("ED", "ES"):
            try:
                arr, spacing = load_volume(pid, phase)
                seg = load_segmentation(pid, phase)
                lumen_vol_mm3 = None
                if seg is not None:
                    vox_vol = spacing[0] * spacing[1] * spacing[2]  # mm³/voxel (X*Y*Z spacing)
                    lumen_vol_mm3 = float((seg == 2).sum()) * vox_vol
                stats.append({
                    "id": f"{pid[-2:]}\n{phase}",
                    "patient": pid,
                    "phase": phase,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "z": arr.shape[0],
                    "lumen_ml": lumen_vol_mm3 / 1000.0 if lumen_vol_mm3 else None,
                })
            except FileNotFoundError:
                pass

    if not stats:
        return

    labels = [s["id"] for s in stats]
    means = [s["mean"] for s in stats]
    stds = [s["std"] for s in stats]
    lumen_mls = [s["lumen_ml"] for s in stats]
    n_panels = 3 if any(v is not None for v in lumen_mls) else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    fig.suptitle("CETUS Dataset Statistics", fontsize=13)
    x = np.arange(len(labels))
    w = 0.6

    axes[0].bar(x, means, width=w, color="steelblue", alpha=0.8, label="mean")
    axes[0].errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_title("Voxel Intensity\n(mean ± std)")
    axes[0].set_ylabel("Intensity (raw)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [s["z"] for s in stats], width=w, color="coral", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_title("Number of Axial Slices\n(Z depth = N_el in pipeline)")
    axes[1].set_ylabel("Slices")
    axes[1].grid(axis="y", alpha=0.3)

    if n_panels == 3:
        valid_x = [xi for xi, v in zip(x, lumen_mls) if v is not None]
        valid_v = [v for v in lumen_mls if v is not None]
        valid_l = [labels[xi] for xi in valid_x]
        axes[2].bar(valid_x, valid_v, width=w, color="mediumseagreen", alpha=0.8)
        axes[2].set_xticks(valid_x)
        axes[2].set_xticklabels([labels[xi] for xi in valid_x], fontsize=7)
        axes[2].set_title("LV Lumen Volume\n(Label 2, mL)")
        axes[2].set_ylabel("Volume (mL)")
        axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "07_cetus_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved statistics to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"CETUS data directory: {DATA_DIR}")
    print(f"Output directory:     {OUTPUT_DIR}\n")

    print("=== Figure 1: Multi-patient overview ===")
    make_overview(OVERVIEW_PATIENTS)

    print("\n=== Figure 2: Orthogonal views with segmentation overlay ===")
    make_orthogonal(DETAIL_PATIENT, phase="ED")

    print("\n=== Figure 3: B-plane strip across elevation ===")
    make_bplane_strip(DETAIL_PATIENT, phase="ED", n_cols=N_BPLANE_COLS)

    print("\n=== Figure 4: Volume statistics ===")
    make_statistics(OVERVIEW_PATIENTS)

    print("\nVisualization complete.")
    print(f"All figures saved to {OUTPUT_DIR}/07_cetus_*.png")
