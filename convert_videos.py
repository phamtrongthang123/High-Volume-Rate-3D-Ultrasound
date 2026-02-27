"""Convert EchoNet-Dynamic AVI videos to numpy frame arrays.

Extracts all frames from each split (TRAIN/VAL/TEST), converts to
grayscale uint8, and saves as a single .npy file per split.

Output files (in --output-dir, default same as --data-dir):
  train_frames.npy   (N_train, 112, 112) uint8
  val_frames.npy     (N_val,   112, 112) uint8
  test_frames.npy    (N_test,  112, 112) uint8

Usage:
  python convert_videos.py --data-dir EchoNet-Dynamic
"""

import env_setup  # noqa: F401 — must be first

import argparse
import csv
import os
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Convert EchoNet-Dynamic videos to numpy")
    p.add_argument("--data-dir", default="EchoNet-Dynamic",
                    help="Path to EchoNet-Dynamic directory")
    p.add_argument("--output-dir", default=None,
                    help="Output directory (default: same as --data-dir)")
    p.add_argument("--splits", nargs="+", default=["TRAIN", "VAL"],
                    help="Splits to convert (default: TRAIN VAL)")
    return p.parse_args()


def convert_split(data_dir, split, output_dir):
    """Extract all frames from videos in a given split."""
    filelist_path = os.path.join(data_dir, "FileList.csv")
    video_dir = os.path.join(data_dir, "Videos")

    # First pass: collect video names and total frame count
    videos = []
    total_frames = 0
    with open(filelist_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Split"].strip() == split:
                name = row["FileName"].strip()
                nf = int(row["NumberOfFrames"])
                videos.append((name, nf))
                total_frames += nf

    print(f"  {split}: {len(videos)} videos, {total_frames:,} frames")

    # Pre-allocate output array
    arr = np.empty((total_frames, 112, 112), dtype=np.uint8)

    # Second pass: extract frames
    idx = 0
    t0 = time.time()
    for vi, (vid_name, expected_nf) in enumerate(videos):
        vid_path = os.path.join(video_dir, f"{vid_name}.avi")
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"    Warning: could not open {vid_path}, skipping")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            arr[idx] = frame
            idx += 1

        cap.release()

        if (vi + 1) % 500 == 0:
            elapsed = time.time() - t0
            pct = (vi + 1) / len(videos) * 100
            print(f"    {vi + 1}/{len(videos)} videos ({pct:.0f}%), "
                  f"{idx:,} frames, {elapsed:.0f}s")

    # Trim if some frames were skipped
    if idx < total_frames:
        print(f"    Note: got {idx:,} frames (expected {total_frames:,}), trimming")
        arr = arr[:idx]

    elapsed = time.time() - t0
    print(f"    Done: {idx:,} frames in {elapsed:.0f}s")

    # Save
    out_path = os.path.join(output_dir, f"{split.lower()}_frames.npy")
    print(f"    Saving to {out_path} ...")
    np.save(out_path, arr)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"    Saved: {size_gb:.1f} GB")

    return idx


def main():
    args = parse_args()
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    total = 0
    t0 = time.time()
    for split in args.splits:
        n = convert_split(args.data_dir, split, output_dir)
        total += n
        print()

    elapsed = time.time() - t0
    print(f"All done: {total:,} total frames in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
