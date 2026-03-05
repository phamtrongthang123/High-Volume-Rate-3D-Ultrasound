"""
summarize_patients.py — Aggregate per-patient metrics from multi-patient CETUS runs.

Usage:
    python summarize_patients.py [--phase ED] [--output-dir ../outputs]

Reads outputs/{patientXX}_{phase}/pixart_metrics.json for each patient
and prints a summary table with mean ± std.
"""

import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="ED", choices=["ED", "ES"])
    parser.add_argument("--outputs-dir", default=os.path.join(PROJECT_DIR, "outputs"))
    args = parser.parse_args()

    results = []
    missing = []
    for i in range(1, 46):
        patient_id = f"patient{i:02d}"
        metrics_path = os.path.join(args.outputs_dir, f"{patient_id}_{args.phase}", "pixart_metrics.json")
        if not os.path.exists(metrics_path):
            missing.append(patient_id)
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        results.append({
            "patient": patient_id,
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "lpips": m["lpips"],
        })

    if not results:
        print("No results found. Check output directories.")
        return

    print(f"\n=== CETUS {args.phase} Phase — {len(results)}/45 patients ===\n")
    print(f"{'Patient':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['patient']:<12} {r['psnr']:>8.2f} {r['ssim']:>8.4f} {r['lpips']:>10.6f}")

    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    lpips = [r["lpips"] for r in results]
    print("-" * 42)
    print(f"{'Mean':<12} {np.mean(psnrs):>8.2f} {np.mean(ssims):>8.4f} {np.mean(lpips):>10.6f}")
    print(f"{'Std':<12} {np.std(psnrs):>8.2f} {np.std(ssims):>8.4f} {np.std(lpips):>10.6f}")
    print(f"{'Min':<12} {np.min(psnrs):>8.2f} {np.min(ssims):>8.4f} {np.min(lpips):>10.6f}")
    print(f"{'Max':<12} {np.max(psnrs):>8.2f} {np.max(ssims):>8.4f} {np.max(lpips):>10.6f}")

    if missing:
        print(f"\nMissing ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
