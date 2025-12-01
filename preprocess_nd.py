import SimpleITK as sitk
import cv2
import numpy as np
import csv
import os
import argparse
from PIL import Image
from pathlib import Path


def read_csv(filename):
    lines = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines


def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def normalize_planes(npzarray):
    maxHU = 350.0
    minHU = -1150.0
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    return np.clip(npzarray, 0.0, 1.0)


def load_itk_image(filename):
    img = sitk.ReadImage(filename)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = np.array(img.GetSpacing())[::-1]  # (Z, Y, X)
    origin = np.array(img.GetOrigin())[::-1]
    return arr, origin, spacing, img


def preprocess_ct(input_path, output_base_path):
    np_vol, origin, spacing, sitk_img = load_itk_image(input_path)
    old_size = sitk_img.GetSize()
    old_spacing = sitk_img.GetSpacing()
    new_spacing = np.array([old_spacing[0], old_spacing[1], 1.0])
    new_size = np.array(old_size * (old_spacing / new_spacing)).astype(int).tolist()
    resampled = sitk.Resample(
        sitk_img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        sitk_img.GetOrigin(),
        new_spacing,
        sitk_img.GetDirection(),
        -1000.,
        sitk_img.GetPixelID()
    )
    im = sitk.GetArrayFromImage(resampled)
    out = np.full((im.shape[0], 1024, 1024), -1000)
    for i in range(im.shape[0]):
        out[i] = cv2.resize(im[i, :, :], (1024, 1024), interpolation=cv2.INTER_LINEAR)
    norm = normalize_planes(out.astype(np.float32))
    for i in range(norm.shape[0]):
        pil_img = Image.fromarray((norm[i] * 255).astype(np.uint8)).convert("L")
        slice_path = output_base_path.parent / f"{output_base_path.name}.{i:03d}.png"
        pil_img.save(slice_path)
    np.savez(
        output_base_path.parent / f"{output_base_path.name}.npz",
        slices=norm,
        original_size=np.array(old_size),
        original_spacing=np.array(old_spacing),
        origin=np.array(sitk_img.GetOrigin()),
        array_origin_zyx=origin,
        array_spacing_zyx=spacing,
        resampled_size=np.array(resampled.GetSize()),
        resampled_spacing=new_spacing,
        hu_window=np.array([-1150.0, 350.0]),
        num_slices=np.array([norm.shape[0]], dtype=np.int32),
        resize_shape=np.array([1024, 1024], dtype=np.int32)
    )


def main(args):
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    with open(args.filelist, "r") as f:
        ct_paths = [Path(line.strip()) for line in f.readlines()]

    idx = args.index
    ct_path = ct_paths[idx]

    rel_parent = ct_path.relative_to(input_root).parent
    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = ct_path.name
    if base_name.lower().endswith(".mhd"):
        base_name = base_name[:-4]
    out_base = out_dir / base_name
    first_slice = out_dir / f"{base_name}.000.png"
    npz_path = out_dir / f"{base_name}.npz"
    if first_slice.exists() and npz_path.exists():
        print(f"Already exists: {base_name}")
        return
    preprocess_ct(ct_path, out_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    main(args)
