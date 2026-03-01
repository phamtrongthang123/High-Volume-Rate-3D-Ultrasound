"""
09_sweep_summary.py — Summarize multi-acceleration-rate sweep results.

Reads outputs/r*/metrics.json, produces:
- ASCII table (printed + saved to outputs/09_sweep_table.txt)
- CSV at outputs/09_sweep_summary.csv
- Figure at outputs/09_sweep_summary.png (PSNR, SSIM, LPIPS vs r with paper comparison)
"""

import env_setup  # noqa: F401 — must be first

import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Paper reference values (from README, Figures 6-7) ---
PAPER_REF = {
    2: {"psnr": 28.7, "lpips": 0.095},
    3: {"psnr": 26.3, "lpips": 0.13},
    6: {"psnr": 23.5, "lpips": 0.16},
    10: {"psnr": 22.3, "lpips": 0.19},
}

# --- Load results ---
json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "r*/metrics.json")))
if not json_files:
    print("No metrics.json files found in outputs/r*/. Run run_sweep.sh first.")
    raise SystemExit(1)

results = []
for path in json_files:
    with open(path) as f:
        data = json.load(f)
    results.append(data)

results.sort(key=lambda d: d["accel_rate"])
r_values = [d["accel_rate"] for d in results]

print(f"Found results for r = {r_values}\n")

# --- ASCII table ---
header = f"{'r':>4s} | {'PSNR (dB)':>10s} | {'SSIM':>8s} | {'LPIPS':>8s}"
sep = "-" * len(header)
lines = [header, sep]
for d in results:
    lines.append(
        f"{d['accel_rate']:4d} | {d['psnr']:10.2f} | {d['ssim']:8.4f} | {d['lpips']:8.4f}"
    )

# Add paper reference section
lines.append("")
lines.append("Paper reference (Diffusion method):")
header2 = f"{'r':>4s} | {'PSNR (dB)':>10s} | {'LPIPS':>8s}"
lines.append(header2)
lines.append("-" * len(header2))
for r in sorted(PAPER_REF.keys()):
    ref = PAPER_REF[r]
    lines.append(f"{r:4d} | {ref['psnr']:10.1f} | {ref['lpips']:8.3f}")

table_text = "\n".join(lines)
print(table_text)

table_path = os.path.join(OUTPUT_DIR, "09_sweep_table.txt")
with open(table_path, "w") as f:
    f.write(table_text + "\n")
print(f"\nSaved table to {table_path}")

# --- CSV ---
csv_path = os.path.join(OUTPUT_DIR, "09_sweep_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["accel_rate", "psnr", "ssim", "lpips"])
    for d in results:
        writer.writerow([d["accel_rate"], d["psnr"], d["ssim"], d["lpips"]])
print(f"Saved CSV to {csv_path}")

# --- Figure: 3 subplots (PSNR, SSIM, LPIPS vs r) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

our_r = np.array(r_values)
our_psnr = np.array([d["psnr"] for d in results])
our_ssim = np.array([d["ssim"] for d in results])
our_lpips = np.array([d["lpips"] for d in results])

paper_r = np.array(sorted(PAPER_REF.keys()))
paper_psnr = np.array([PAPER_REF[r]["psnr"] for r in paper_r])
paper_lpips = np.array([PAPER_REF[r]["lpips"] for r in paper_r])

# PSNR
axes[0].plot(our_r, our_psnr, "bo-", label="Ours", markersize=8, linewidth=2)
axes[0].plot(paper_r, paper_psnr, "rs--", label="Paper", markersize=8, linewidth=2)
axes[0].set_xlabel("Acceleration rate (r)")
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_title("PSNR vs Acceleration Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(sorted(set(list(our_r) + list(paper_r))))

# SSIM
axes[1].plot(our_r, our_ssim, "bo-", label="Ours", markersize=8, linewidth=2)
axes[1].set_xlabel("Acceleration rate (r)")
axes[1].set_ylabel("SSIM")
axes[1].set_title("SSIM vs Acceleration Rate")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(sorted(set(list(our_r))))

# LPIPS
axes[2].plot(our_r, our_lpips, "bo-", label="Ours", markersize=8, linewidth=2)
axes[2].plot(paper_r, paper_lpips, "rs--", label="Paper", markersize=8, linewidth=2)
axes[2].set_xlabel("Acceleration rate (r)")
axes[2].set_ylabel("LPIPS")
axes[2].set_title("LPIPS vs Acceleration Rate")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(sorted(set(list(our_r) + list(paper_r))))

plt.suptitle("Multi-Rate Sweep: Our Results vs Paper", fontsize=14)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "09_sweep_summary.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved figure to {fig_path}")

print("\nSweep summary complete.")
