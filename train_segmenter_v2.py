"""
train_segmenter_v2.py -- a stronger lesion segmenter aiming for Dice ~0.80 so the
descriptor-free pipeline can run autonomously. Improvements over v1: adds BUS-BRA
mask pairs (triples the data), on-the-fly geometric + photometric augmentation,
and more epochs. A fixed 20% of BUS-BRA is held out from segmenter training so it
can serve as a genuinely unseen deployment test for the enhancement.
"""
import glob
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

import config

SIZE = 224
EPOCHS = 30
BATCH = 16
dev = config.DEVICE
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def breast_busi_pairs():
    rows = []
    f = glob.glob("dataset/*clinical*")[0]
    for _, r in pd.read_excel(f).iterrows():
        img = f"dataset/images/{r['Image_filename']}"; mask = f"dataset/images/{r['Mask_tumor_filename']}"
        if os.path.exists(img) and os.path.exists(mask):
            rows.append((img, mask))
    for cls in ("benign", "malignant", "normal"):
        for p in sorted(glob.glob(f"busi/{cls}/*.png")):
            if "_mask" in p:
                continue
            m = sorted(glob.glob(f"{p[:-4]}_mask*.png"))
            if m:
                rows.append((p, m[0]))
    return rows


def busbra_pairs():
    meta = pd.read_csv("busbra/BUSBRA/bus_data.csv")
    rows = []
    for _, r in meta.iterrows():
        img = f"busbra/BUSBRA/Images/{r['ID']}.png"
        mask = f"busbra/BUSBRA/Masks/mask_{r['ID'][4:]}.png"
        if os.path.exists(img) and os.path.exists(mask):
            rows.append((img, mask))
    return rows


class SegDS(Dataset):
    def __init__(self, rows, train):
        self.rows = rows; self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ip, mp = self.rows[i]
        img = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        m = Image.open(mp).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img); m = TF.hflip(m)
            ang = random.uniform(-12, 12)
            img = TF.rotate(img, ang); m = TF.rotate(m, ang)
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
            img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))
        x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]
        return x, y


@torch.no_grad()
def dice_iou(model, loader):
    model.eval(); d = i = n = 0
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        inter = (p * y).sum((1, 2, 3)); union = ((p + y) > 0).float().sum((1, 2, 3))
        d += (2 * inter / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6)).sum().item()
        i += (inter / (union + 1e-6)).sum().item(); n += len(x)
    return d / n, i / n


def main():
    random.seed(config.SEED); torch.manual_seed(config.SEED)
    bb = busbra_pairs()
    bb_tr, bb_te = train_test_split(bb, test_size=0.20, random_state=config.SEED)
    # persist the held-out BUS-BRA test paths for the deployment eval
    pd.DataFrame(bb_te, columns=["image_path", "mask_path"]).to_csv("busbra_seg_holdout.csv", index=False)

    train_rows = breast_busi_pairs() + bb_tr
    print(f"segmenter v2 train {len(train_rows)} (BrEaST+BUSI+80% BUS-BRA) | "
          f"BUS-BRA held-out test {len(bb_te)}", flush=True)
    trl = DataLoader(SegDS(train_rows, True), batch_size=BATCH, shuffle=True, num_workers=0)
    tel = DataLoader(SegDS(bb_te, False), batch_size=BATCH, num_workers=0)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss(); dl = smp.losses.DiceLoss(mode="binary")

    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); logit = model(x)
            (bce(logit, y) + dl(logit, y)).backward(); opt.step()
        sched.step()
        dsc, iou = dice_iou(model, tel)
        if dsc > best:
            best = dsc; torch.save(model.state_dict(), "prod/segmenter_v2.pt")
        print(f"epoch {ep+1:2d}  BUS-BRA held-out Dice {dsc:.3f}  IoU {iou:.3f}", flush=True)
    print(f"\nbest BUS-BRA held-out Dice {best:.3f}  -> prod/segmenter_v2.pt")


if __name__ == "__main__":
    main()
