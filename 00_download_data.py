"""
00_download_data.py — Verify CETUS dataset is available.

CETUS data is NOT downloaded automatically — it must be obtained manually
and placed at data/CETUS/dataset/ (see MANDATORY_CITATION.md).
This script verifies the data is present and readable.
"""

import env_setup  # noqa: F401 — must be first

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "CETUS" / "dataset"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"CETUS dataset not found at {DATA_DIR}\n"
        "Please download CETUS manually and place it at data/CETUS/dataset/\n"
        "See: https://www.creatis.insa-lyon.fr/Challenge/cetus/"
    )

patients = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("patient")])
if not patients:
    raise FileNotFoundError(f"No patient directories found in {DATA_DIR}")

print(f"Found {len(patients)} patients: {patients[0].name} ... {patients[-1].name}")

# Verify first patient
p = patients[0]
for phase in ("ED", "ES"):
    f = p / f"{p.name}_{phase}.nii.gz"
    assert f.exists(), f"Missing: {f}"
    print(f"  OK: {f.name}  ({f.stat().st_size // 1024} KB)")

print("\nCETUS data verified. Remember mandatory citation:")
print(
    "  O. Bernard et al., 'Standardized Evaluation System for Left Ventricular\n"
    "  Segmentation Algorithms in 3D Echocardiography,' IEEE TMI 35(4), 2016.\n"
    "  https://doi.org/10.1109/tmi.2015.2503890"
)

citation_file = DATA_DIR / "MANDATORY_CITATION.md"
if citation_file.exists():
    print(citation_file.read_text())
