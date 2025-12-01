import SimpleITK as sitk
import cv2
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import os


def load_itk_image(filename):
    img = sitk.ReadImage(filename)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = np.array(img.GetSpacing())[::-1]  # (Z, Y, X)
    origin = np.array(img.GetOrigin())[::-1]
    return arr, origin, spacing, img


def get_patch_mm(np_vol, center_xyz, size_mm, spacing_xyz, pad_value=-1000):
    cz, cy, cx = center_xyz

    half_z = (size_mm / spacing_xyz[0]) / 2.0
    half_y = (size_mm / spacing_xyz[1]) / 2.0
    half_x = (size_mm / spacing_xyz[2]) / 2.0

    half_z_i = int(np.round(half_z))
    half_y_i = int(np.round(half_y))
    half_x_i = int(np.round(half_x))

    z1, z2 = cz - half_z_i, cz + half_z_i
    y1, y2 = cy - half_y_i, cy + half_y_i
    x1, x2 = cx - half_x_i, cx + half_x_i

    patch_shape = tuple(map(int, (z2 - z1, y2 - y1, x2 - x1)))
    patch = np.full(patch_shape, pad_value, dtype=np_vol.dtype)

    Z, Y, X = np_vol.shape

    z1v, z2v = int(max(z1, 0)), int(min(z2, Z))
    y1v, y2v = int(max(y1, 0)), int(min(y2, Y))
    x1v, x2v = int(max(x1, 0)), int(min(x2, X))

    z1p = int(z1v - z1)
    y1p = int(y1v - y1)
    x1p = int(x1v - x1)

    patch[z1p : z1p + (z2v - z1v), y1p : y1p + (y2v - y1v), x1p : x1p + (x2v - x1v)] = (
        np_vol[z1v:z2v, y1v:y2v, x1v:x2v]
    )

    return patch


def normalize_planes(arr):
    minHU = -1150.0
    maxHU = 350.0
    arr = (arr - minHU) / (maxHU - minHU)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def preprocess_ct(input_path, output_npz_path):
    itk_img = sitk.ReadImage(str(input_path))
    np_vol = sitk.GetArrayFromImage(itk_img)  # Z,Y,X
    spacing_sitk = np.array(itk_img.GetSpacing())  # (sx, sy, sz)
    origin_sitk = np.array(itk_img.GetOrigin())
    direction_sitk = itk_img.GetDirection()

    spacing_zyx = spacing_sitk[::-1]

    center_voxel = np.array(np_vol.shape) / 2.0  # Z,Y,X center
    patch76 = get_patch_mm(np_vol, center_voxel, size_mm=76.0, spacing_xyz=spacing_zyx)

    aug_patch = patch76.copy()
    Zp = aug_patch.shape[0]

    for i in range(Zp):
        img = aug_patch[i].astype(np.float32)

        angle = float(np.random.uniform(-20, 20))

        tx_mm = float(np.random.uniform(-1.0, 1.0))
        ty_mm = float(np.random.uniform(-1.0, 1.0))
        tx_pix = tx_mm / spacing_zyx[2]
        ty_pix = ty_mm / spacing_zyx[1]

        rows, cols = img.shape
        center = (cols / 2.0, rows / 2.0)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx_pix
        M[1, 2] += ty_pix

        warped = cv2.warpAffine(
            img,
            M,
            (cols, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1000,
        )
        aug_patch[i] = warped.astype(np.int16)

    center_patch_voxel = np.array(aug_patch.shape) / 2.0
    patch50 = get_patch_mm(
        aug_patch, center_patch_voxel, size_mm=50.0, spacing_xyz=spacing_zyx
    )

    patch50_sitk = sitk.GetImageFromArray(patch50)
    patch50_sitk.SetSpacing(spacing_sitk)
    patch50_sitk.SetOrigin(origin_sitk)
    patch50_sitk.SetDirection(direction_sitk)

    new_spacing = np.array([0.78, 0.78, 0.78])
    out_size = [64, 64, 64]

    resampled = sitk.Resample(
        patch50_sitk,
        out_size,
        sitk.Transform(),
        sitk.sitkLinear,
        patch50_sitk.GetOrigin(),
        new_spacing,
        patch50_sitk.GetDirection(),
        -1000.0,
        patch50_sitk.GetPixelID(),
    )

    out_np = sitk.GetArrayFromImage(resampled)
    norm = normalize_planes(out_np)

    np.savez(
        output_npz_path,
        slices=norm,
        original_size=np.array(itk_img.GetSize()),
        original_spacing=spacing_sitk,
        origin=origin_sitk,
        direction=np.array(direction_sitk),
        resampled_spacing=new_spacing,
        resampled_size=np.array(out_size),
        hu_window=np.array([-1150.0, 350.0]),
        num_slices=np.array([norm.shape[0]], dtype=np.int32),
    )


def main(args):
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    with open(args.filelist, "r") as f:
        ct_paths = [Path(line.strip()) for line in f.readlines()]

    # Resolve index: use --index or SLURM_ARRAY_TASK_ID
    idx = args.index
    if idx is None:
        slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_idx is None:
            raise SystemExit("Provide --index or set SLURM_ARRAY_TASK_ID")
        idx = int(slurm_idx)

    ct_path = ct_paths[idx]

    rel_parent = ct_path.relative_to(input_root).parent
    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = ct_path.name
    if base_name.lower().endswith(".mhd"):
        base_name = base_name[:-4]
    out_base = out_dir / base_name
    npz_path = out_dir / f"{base_name}.npz"
    if npz_path.exists():
        print(f"Already exists: {npz_path}")
        return

    preprocess_ct(ct_path, npz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--index", type=int, required=False)
    args = parser.parse_args()
    main(args)
