# -*- coding: utf-8 -*-
"""
SAFE & CLEAN PREPROCESSING PIPELINE
- YOLO lung slice detection done ONCE per volume BEFORE multiprocessing
- No race conditions, no corrupted npy files
- Workers only load cached slices (read-only)
- Progress bars per subset for lung slice detection
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


############################################################
# ---------- BASIC UTILS ----------
############################################################

def load_itk_image(path):
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)  # (Z,Y,X)
    origin = np.array(itk_img.GetOrigin())[::-1]
    spacing = np.array(itk_img.GetSpacing())[::-1]
    return img, origin, spacing

def world_to_voxel(world, origin, spacing):
    return ((np.abs(world - origin) / spacing)).astype(int)

def normalize_hu(volume):
    return np.clip((volume + 1000.) / 1400., 0, 1)


############################################################
# ---------- CUBE EXTRACTION ----------
############################################################

def extract_3d_cube(volume, center, cube_size=48):
    z, y, x = map(int, center)
    half = cube_size // 2
    Z, Y, X = volume.shape

    z1, z2 = max(0, z-half), min(Z, z+half)
    y1, y2 = max(0, y-half), min(Y, y+half)
    x1, x2 = max(0, x-half), min(X, x+half)

    cube = volume[z1:z2, y1:y2, x1:x2]

    pad = [(0, cube_size - cube.shape[i]) for i in range(3)]
    cube = np.pad(cube, pad, mode="constant", constant_values=-1000)
    return cube


############################################################
# ---------- YOLO ----------
############################################################

print("Loading YOLOv5s (once)…")
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
if torch.cuda.is_available():
    yolo.to("cuda")


############################################################
# ---------- LUNG SLICE CACHE ----------
############################################################

LUNG_CACHE_DIR = "lung_slice_cache"
os.makedirs(LUNG_CACHE_DIR, exist_ok=True)

def lung_cache_path(uid):
    return os.path.join(LUNG_CACHE_DIR, f"{uid}.npy")

def compute_lung_slices(volume, uid):
    slices = []
    for z in range(volume.shape[0]):
        img = ((volume[z] + 1000) / 1400 * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = yolo(img)
        boxes = results.xyxy[0].cpu().numpy()
        if len(boxes) > 0:
            slices.append(z)

    slices = np.array(slices, dtype=np.int32)
    np.save(lung_cache_path(uid), slices)
    return slices

def safe_load_slices(uid):
    path = lung_cache_path(uid)
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


############################################################
# ---------- PRECOMPUTE LUNG SLICES (WITH SUBSET PROGRESS BARS) ----------
############################################################

import time
from datetime import timedelta

def pretty_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def precompute_all_lung_slices(subset_map):
    print("\n=== Precomputing lung slices for all volumes ===")

    # 1. group uids by subset
    subsets = {}
    for uid, path in subset_map.items():
        subset_id = int(path.split("subset")[-1].split(os.sep)[0])
        subsets.setdefault(subset_id, []).append(uid)

    # Total scans for full ETA
    total_scans = sum(len(v) for v in subsets.values())

    # Keep a running count of total scans completed
    total_done = 0
    global_start = time.time()

    # loop subsets in order
    for sid in sorted(subsets.keys()):
        uids = subsets[sid]
        n_scans = len(uids)

        print(f"\nSubset {sid} ({n_scans} scans):")

        subset_start = time.time()

        with tqdm(total=n_scans,
                  desc=f"Subset {sid}",
                  unit="scan",
                  dynamic_ncols=True) as pbar:

            for uid in uids:
                cached = safe_load_slices(uid)
                if cached is None:
                    volume, _, _ = load_itk_image(subset_map[uid])
                    compute_lung_slices(volume, uid)

                pbar.update(1)
                total_done += 1

                # ---- ETA CALCULATIONS ----
                elapsed_sub = time.time() - subset_start
                rate_sub = pbar.n / elapsed_sub if elapsed_sub > 0 else 0
                remaining_sub = (n_scans - pbar.n) / rate_sub if rate_sub > 0 else 0

                elapsed_global = time.time() - global_start
                rate_global = total_done / elapsed_global if elapsed_global > 0 else 0
                remaining_global = (total_scans - total_done) / rate_global if rate_global > 0 else 0

                pbar.set_postfix({
                    "ETA_subset": pretty_time(remaining_sub),
                    "ETA_total": pretty_time(remaining_global)
                })

    print("\n✓ All lung slices cached safely.\n")


############################################################
# ---------- 9-VIEW EXTRACTION ----------
############################################################

def extract_9_views(cube):
    Z, Y, X = cube.shape
    cz, cy, cx = Z//2, Y//2, X//2
    offsets = [-4, 0, 4]

    views = []
    for o in offsets:
        views.append(cube[np.clip(cz+o, 0, Z-1), :, :])
    for o in offsets:
        views.append(cube[:, np.clip(cy+o, 0, Y-1), :])
    for o in offsets:
        views.append(cube[:, :, np.clip(cx+o, 0, X-1)])

    return np.array(views, dtype=np.float32)


############################################################
# ---------- PROCESS ROW ----------
############################################################

def process_row(row, subset_map, out_cube_dir, out_view_dir, cube_size=48):
    uid = row["seriesuid"]
    if uid not in subset_map:
        return 0

    volume_path = subset_map[uid]
    volume, origin, spacing = load_itk_image(volume_path)

    lung_slices = safe_load_slices(uid)
    if lung_slices is None:
        return 0

    world = np.array([row["coordZ"], row["coordY"], row["coordX"]])
    voxel = world_to_voxel(world, origin, spacing)

    if voxel[0] not in lung_slices:
        return 0

    cube = extract_3d_cube(volume, voxel, cube_size)
    cube = normalize_hu(cube)

    cube_path = os.path.join(out_cube_dir, f"{uid}_{row.name}.nii.gz")
    nib.save(nib.Nifti1Image(cube.astype(np.float32), np.eye(4)), cube_path)

    views = extract_9_views(cube)
    view_path = os.path.join(out_view_dir, f"{uid}_{row.name}.npy")
    np.save(view_path, views)

    return 1


############################################################
# ---------- SUBSET MAP ----------
############################################################

def build_subset_map(base_dir, subsets):
    mapping = {}
    for s in subsets:
        folder = os.path.join(base_dir, f"subset{s}")
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".mhd"):
                uid = f[:-4]
                mapping[uid] = os.path.join(folder, f)
    return mapping


############################################################
# ---------- MAIN ----------
############################################################

def preprocess(
    base_dir=".",
    candidate_file="candidates_V2.csv",
    out_cube_dir="cubes",
    out_view_dir="views9",
    subsets=range(10),
    workers=4,
    cube_size=48
):

    os.makedirs(out_cube_dir, exist_ok=True)
    os.makedirs(out_view_dir, exist_ok=True)

    candidates = pd.read_csv(candidate_file)
    subset_map = build_subset_map(base_dir, subsets)

    # PRECOMPUTE ALL LUNG SLICES FIRST (with per-subset progress bars!)
    precompute_all_lung_slices(subset_map)

    # Now safe to parallelize
    func = partial(
        process_row,
        subset_map=subset_map,
        out_cube_dir=out_cube_dir,
        out_view_dir=out_view_dir,
        cube_size=cube_size
    )

    print(f"\nProcessing {len(candidates)} candidates (workers={workers})…\n")
    results = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap(func, [row for _, row in candidates.iterrows()]),
                      total=len(candidates),
                      desc="Candidates"):
            results.append(r)

    print(f"\n✓ Saved {sum(results)} cubes + 9-view files.")


if __name__ == "__main__":
    preprocess(
        base_dir=".",
        candidate_file="candidates_V2.csv",
        subsets=range(10),
        out_cube_dir="cubes",
        out_view_dir="views9",
        workers=4
    )
