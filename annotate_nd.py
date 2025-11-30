import os
import glob
import numpy as np
from pathlib import Path
import argparse

def annotate(uid, voxel_coords, diameter, dataset_root):
    file = glob.glob(f'{dataset_root}/images/subset*/{uid}*.npz')[0]
    # Ensure corresponding labels directory exists
    labels_dir = Path(file).parent.as_posix().replace(f'{dataset_root}/images', f'{dataset_root}/labels')
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    data = np.load(file)
    old_origin = data['origin']
    old_size = data['resampled_size']
    old_spacing = data['original_spacing']
    new_spacing = data['resampled_spacing']

    vox_resampled = (voxel_coords - old_origin) / new_spacing
    scale_x = 1024 / old_size[0]
    scale_y = 1024 / old_size[1]
    vox_scaled = np.array([vox_resampled[0] * scale_x, vox_resampled[1] * scale_y, vox_resampled[2]])
    radius_x = ((diameter / 2) / new_spacing[0]) * scale_x
    radius_y = ((diameter / 2) / new_spacing[1]) * scale_y

    x1 = vox_scaled[0] - radius_x
    y1 = vox_scaled[1] - radius_y
    x2 = vox_scaled[0] + radius_x
    y2 = vox_scaled[1] + radius_y
    cx = (x1 + x2) / 2 / 1024
    cy = (y1 + y2) / 2 / 1024
    w = (x2 - x1) / 1024
    h = (y2 - y1) / 1024
    sliceno = int(round(vox_scaled[2]))
    
    annofile = file.replace('images', 'labels').replace('.npz', f'.{sliceno:03d}.txt')
    with open(annofile, 'w') as f:
        f.write(f'1 {cx} {cy} {w} {h}\n')

def ensure_label_subsets(dataset_root):
    # Create dataset/labels/subset[0-9] if missing
    for i in range(10):
        Path(f"{dataset_root}/labels/subset{i}").mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="SLURM batch array annotator")
    parser.add_argument("--csv", required=True, help="CSV with rows: uid,x,y,z,diameter")
    parser.add_argument("--index", type=int, help="Row index; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--dataset", default="dataset", help="Root dataset directory (default: dataset)")
    args = parser.parse_args()

    # Resolve index from SLURM if not provided
    idx = args.index
    if idx is None:
        slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_idx is None:
            raise SystemExit("Provide --index or SLURM_ARRAY_TASK_ID")
        idx = int(slurm_idx)

    # Ensure labels subset directories exist
    ensure_label_subsets(args.dataset)

    # Load CSV and pick row
    rows = []
    with open(args.csv, "r") as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            uid = parts[0]
            x, y, z = map(float, parts[1:4])
            diameter = float(parts[4])
            rows.append((uid, np.array([x, y, z], dtype=np.float32), diameter))

    if idx < 0 or idx >= len(rows):
        raise SystemExit(f"Index {idx} out of range for {len(rows)} rows")

    uid, vox, dia = rows[idx]
    annotate(uid, vox, dia, args.dataset)

if __name__ == "__main__":
    main()