import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def get_views_from_volume(vol):
    """
    vol: numpy array (Z, Y, X) = (64,64,64)
    Returns: torch tensor (9, 1, H, W)
    """
    Z, Y, X = vol.shape

    mid_z = Z // 2
    axial = [
        vol[mid_z],
        vol[max(mid_z - 2, 0)],
        vol[min(mid_z + 2, Z - 1)],
    ]

    mid_y = Y // 2
    coronal = [
        vol[:, mid_y, :],
        vol[:, max(mid_y - 2, 0), :],
        vol[:, min(mid_y + 2, Y - 1), :],
    ]

    mid_x = X // 2
    sagittal = [
        vol[:, :, mid_x],
        vol[:, :, max(mid_x - 2, 0)],
        vol[:, :, min(mid_x + 2, X - 1)],
    ]

    views = axial + coronal + sagittal
    views = [torch.tensor(v, dtype=torch.float32).unsqueeze(0) for v in views]
    return torch.stack(views, dim=0)   # (9,1,H,W)


class MultiViewDataset(Dataset):
    def __init__(self, npz_files):
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        data = np.load(f)
        vol = data["slices"]                    # (64,64,64)
        label = data["label"].astype(np.float32)

        views = get_views_from_volume(vol)      # (9,1,64,64)

        label_t = torch.tensor(label, dtype=torch.float32).view(1)

        return views, label_t

class FPRModel(nn.Module):
    def __init__(self, num_views=9, out_dim=1):
        super().__init__()
        base = resnet50(weights=None)

        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 2048
        self.num_views = num_views

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * num_views, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, views):
        # views: (B, 9, 1, H, W)
        B, N, C, H, W = views.shape
        assert N == self.num_views

        feats = []
        for i in range(N):
            v = views[:, i]              # (B,1,H,W)
            f = self.backbone(v)         # (B,2048,1,1)
            f = f.squeeze(-1).squeeze(-1)  # (B,2048)
            feats.append(f)

        feats = torch.cat(feats, dim=1)  # (B, 2048*9)
        out = self.classifier(feats)     # (B,1)
        return out.squeeze(1)            # (B,)


def compute_froc(labels, probs, uids=None):
    """
    Basic FROC:
      - labels: 0/1 per candidate
      - probs: predicted probabilities
      - uids: list of scan IDs (same length as labels); used for FP/scan.
              If None, we approximate num_scans as number of unique uid
              inferred from filename prefixes or fall back to 1.
    """
    labels = np.asarray(labels).astype(np.int32)
    probs = np.asarray(probs).astype(np.float32)

    if uids is not None:
        num_scans = len(set(uids))
    else:
        num_scans = 1

    thresholds = np.linspace(0.0, 1.0, 200)
    sensitivities = []
    fp_per_scan = []

    for t in thresholds:
        preds = (probs >= t).astype(np.int32)

        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))

        sens = tp / (tp + fn + 1e-6)
        fps = fp / max(num_scans, 1)

        sensitivities.append(sens)
        fp_per_scan.append(fps)

    return np.array(fp_per_scan), np.array(sensitivities)


def plot_froc(fp_scan, sens, out_path="froc_curve.png"):
    plt.figure(figsize=(7,5))
    plt.plot(fp_scan, sens, "-b")
    plt.xlabel("False Positives per Scan")
    plt.ylabel("Sensitivity")
    plt.title("FROC Curve")
    plt.xlim([0, max(10, np.max(fp_scan) + 0.5)])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_fpr_model(train_files, val_files,
                    epochs=20,
                    batch_size=16,
                    num_workers=4,
                    device="cuda"):

    train_ds = MultiViewDataset(train_files)
    val_ds   = MultiViewDataset(val_files)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    model = FPRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for views, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            views = views.to(device)           # (B,9,1,H,W)
            labels = labels.to(device).view(-1)  # (B,)

            optimizer.zero_grad()
            outputs = model(views)             # (B,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss += loss.item() * bs
            n_train += bs

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_probs = []
        all_labels = []
        all_uids = []

        with torch.no_grad():
            for i, (views, labels) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            ):
                views = views.to(device)
                labels = labels.to(device).view(-1)

                outputs = model(views)  # (B,)
                loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss += loss.item() * bs
                n_val += bs

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist()

                )

        val_loss /= max(n_val, 1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_fpr_model.pth")
            print("Saved new best model.")

    val_uids = []
    for f in val_files:
        base = os.path.basename(f)
        uid = base.split("_cand")[0]
        val_uids.append(uid)

    return all_labels, all_probs, val_uids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_fpr",
                        help="Root directory containing subset*/ candidate npz files")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pattern = os.path.join(args.data_root, "subset*", "*.npz")
    all_files = sorted(glob.glob(pattern))
    if len(all_files) == 0:
        raise RuntimeError(f"No npz files found under {pattern}")

    print(f"Found {len(all_files)} candidate npz files.")

    indices = list(range(len(all_files)))
    random.shuffle(indices)

    val_size = int(len(indices) * args.val_frac)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_files = [all_files[i] for i in train_indices]
    val_files   = [all_files[i] for i in val_indices]

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    labels, probs, val_uids = train_fpr_model(
        train_files, val_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    fp_scan, sens = compute_froc(labels, probs, uids=val_uids)
    plot_froc(fp_scan, sens, out_path="froc_curve.png")
    print("Saved FROC curve to froc_curve.png")

    import csv
    with open("val_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "uid", "label", "prob"])
        for file, uid, y, p in zip(val_files, val_uids, labels, probs):
            w.writerow([file, uid, int(y), float(p)])
    print("Saved validation predictions to val_preds.csv")


if __name__ == "__main__":
    main()
