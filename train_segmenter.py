"""
train_segmenter.py -- a lightweight U-Net lesion segmenter, so the ROI-focus and
automated-morphology enhancements no longer need expert masks at inference. Train
on BrEaST + BUSI; BUS-BRA is left entirely unseen so it serves as a cross-domain
deployment test. Reports Dice / IoU on a held-out split and saves prod/segmenter.pt.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

import config

SIZE = 224
EPOCHS = 20
BATCH = 16
dev = config.DEVICE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pairs():
    rows = []
    # BrEaST
    import pandas as pd
    f = glob.glob("dataset/*clinical*")[0]
    for _, r in pd.read_excel(f).iterrows():
        img = f"dataset/images/{r['Image_filename']}"
        mask = f"dataset/images/{r['Mask_tumor_filename']}"
        if os.path.exists(img) and os.path.exists(mask):
            rows.append((img, mask))
    # BUSI
    for cls in ("benign", "malignant", "normal"):
        for p in sorted(glob.glob(f"busi/{cls}/*.png")):
            if "_mask" in p:
                continue
            m = sorted(glob.glob(f"{p[:-4]}_mask*.png"))
            if m:
                rows.append((p, m[0]))
    return rows


class SegDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ip, mp = self.rows[i]
        img = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        m = Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        x = (np.asarray(img, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]
        return x, y


@torch.no_grad()
def dice_iou(model, loader):
    model.eval()
    d, i, n = 0.0, 0.0, 0
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        inter = (p * y).sum((1, 2, 3))
        union = ((p + y) > 0).float().sum((1, 2, 3))
        psum, ysum = p.sum((1, 2, 3)), y.sum((1, 2, 3))
        d += (2 * inter / (psum + ysum + 1e-6)).sum().item()
        i += (inter / (union + 1e-6)).sum().item()
        n += len(x)
    return d / n, i / n


def main():
    rows = pairs()
    tr, va = train_test_split(rows, test_size=0.15, random_state=config.SEED)
    print(f"segmenter data: {len(rows)} pairs (BrEaST+BUSI) | train {len(tr)} val {len(va)}", flush=True)
    trl = DataLoader(SegDS(tr), batch_size=BATCH, shuffle=True, num_workers=0)
    val = DataLoader(SegDS(va), batch_size=BATCH, num_workers=0)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")

    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit = model(x)
            loss = bce(logit, y) + dice_loss(logit, y)
            loss.backward(); opt.step()
        dsc, iou = dice_iou(model, val)
        if dsc > best:
            best = dsc
            torch.save(model.state_dict(), "prod/segmenter.pt")
        print(f"epoch {ep+1:2d}  val Dice {dsc:.3f}  IoU {iou:.3f}", flush=True)
    print(f"\nbest val Dice {best:.3f}  -> saved prod/segmenter.pt")


if __name__ == "__main__":
    main()
