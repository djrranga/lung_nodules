import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def annotate_sample(row):
    uid, x, y, z, label = row["seriesuid"], row["coordX"], row["coordY"], row["coordZ"], row["class"]

    matches = glob.glob(f"dataset/images/subset*/{uid}*.npz")
    if not matches:
        print("Missing npz for:", uid)
        return

    npz_path = matches[0]
    subset = npz_path.split("/")[-2]

    data = np.load(npz_path)

    origin = data["origin"]
    old_spacing = data["original_spacing"]
    new_spacing = data["resampled_spacing"]
    resampled_size = data["resampled_size"]

    world = np.array([x, y, z], dtype=np.float32)
    vox_orig = (world - origin) / old_spacing 

    vox_resampled = vox_orig * (old_spacing / new_spacing)

    H = W = 1024
    oldX, oldY, oldZ = data["resampled_size"]

    scale_x = W / oldX
    scale_y = H / oldY

    px = vox_resampled[0] * scale_x
    py = vox_resampled[1] * scale_y
    pz = vox_resampled[2]

    slice_idx = int(round(pz))

    if slice_idx < 0 or slice_idx >= data["num_slices"][0]:
        print("Slice outside range for:", uid)
        return

    radius_pix = 6

    x1 = px - radius_pix
    y1 = py - radius_pix
    x2 = px + radius_pix
    y2 = py + radius_pix

    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    out_dir = Path("dataset/labels") / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / f"{uid}.{slice_idx:03d}.txt"

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
