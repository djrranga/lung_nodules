import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def annotate_sample(row):
    uid, x, y, z, label = row["seriesuid"], row["coordX"], row["coordY"], row["coordZ"], row["class"]

    # find corresponding npz
    matches = glob.glob(f"dataset/images/subset*/{uid}*.npz")
    if not matches:
        print("Missing npz for:", uid)
        return

    npz_path = matches[0]
    subset = npz_path.split("/")[-2]      # "subset0" etc.

    data = np.load(npz_path)

    origin = data["origin"]                    # world-space origin
    old_spacing = data["original_spacing"]     # original voxel spacing (x,y,z)
    new_spacing = data["resampled_spacing"]    # spacing after resampling
    resampled_size = data["resampled_size"]    # (Z,Y,X) or (X,Y,Z) — you stored it

    # ---------------------------
    # 1. World → original voxel coords
    # ---------------------------
    world = np.array([x, y, z], dtype=np.float32)
    vox_orig = (world - origin) / old_spacing  # in original volume

    # ---------------------------
    # 2. Original voxel → resampled voxel coords
    # ---------------------------
    vox_resampled = vox_orig * (old_spacing / new_spacing)

    # ---------------------------
    # 3. Scaling to the final 2D images (1024×1024)
    # You stored 1024×1024 slices for your detector
    # ---------------------------
    H = W = 1024
    oldX, oldY, oldZ = data["resampled_size"]  # NOTE: your stored order

    scale_x = W / oldX
    scale_y = H / oldY

    px = vox_resampled[0] * scale_x
    py = vox_resampled[1] * scale_y
    pz = vox_resampled[2]                   # slice index BEFORE rounding

    slice_idx = int(round(pz))

    # If slice is outside volume, skip
    if slice_idx < 0 or slice_idx >= data["num_slices"][0]:
        print("Slice outside range for:", uid)
        return

    # ---------------------------
    # 4. Create YOLO bounding box
    # Candidates have no diameter → use fixed radius (5 mm)
    # or use a very small default box (3×3 pixels)
    # ---------------------------
    radius_pix = 6  # arbitrary small box for FPR

    x1 = px - radius_pix
    y1 = py - radius_pix
    x2 = px + radius_pix
    y2 = py + radius_pix

    # Normalize to YoLo format
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    # ---------------------------
    # 5. Save YOLO annotation
    # ---------------------------
    out_dir = Path("dataset/labels") / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / f"{uid}.{slice_idx:03d}.txt"

    # Format: class cx cy w h
    with open(label_path, "w") as f:
        f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print("Annotated:", label_path)


def main():
    df = pd.read_csv("candidates_V2.csv")
    df.columns = ["seriesuid", "coordX", "coordY", "coordZ", "class"]

    for _, row in df.iterrows():
        annotate_sample(row)


if __name__ == "__main__":
    main()
