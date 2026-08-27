"""
malignancy_model.py -- the visual evidence stream: a ViT-B/16 with a single
malignancy logit, trained image-only to predict malignant vs benign. Mirrors the
finding classifier's setup (same backbone/transform) but binary, class-weighted
for the ~1.6:1 imbalance.

train_fold_proba(train_df, test_df) trains on the fold's train split and returns
P(malignant) for the test rows -- the leak-free per-fold entry used by
evaluate_malignancy.py. A Predictor wrapper is provided for the agent.
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

import config
from finding_model import get_transform, IMAGENET_MEAN, IMAGENET_STD  # reuse transform

EPOCHS = 15
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.05


def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)


class _DS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.tf = get_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.tf(img), torch.tensor([float(r["y"])], dtype=torch.float32)


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _proba(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy().ravel())
    return np.concatenate(out)


def train_fold_proba(train_df, test_df, epochs=EPOCHS, device=None, log_prefix=""):
    device = device or config.DEVICE
    n_pos = float(train_df["y"].sum())
    n_neg = float((train_df["y"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    tr = DataLoader(_DS(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te = DataLoader(_DS(test_df, False), batch_size=BATCH_SIZE, num_workers=0)

    model = build_model().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(x), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)

    p = _proba(model, te, device)
    _free(model, opt, sched, crit)
    return p


class Predictor:
    """Loads a saved malignancy checkpoint -> P(malignant) for one image."""
    def __init__(self, ckpt_path, device=None):
        self.device = device or config.DEVICE
        self.model = build_model().to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.tf = get_transform(train=False)

    @torch.no_grad()
    def predict_proba(self, image_path):
        x = self.tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(x)).item())
