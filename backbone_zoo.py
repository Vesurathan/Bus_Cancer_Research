"""
backbone_zoo.py -- generic timm-backbone malignancy classifier so we can benchmark
architectures beyond the ImageNet ViT (ConvNeXt, Swin, DeiT-distilled, ResNet, ...)
under the SAME leak-free recipe as malignancy_model.py. All models are 224-input,
ImageNet-normalised, so they reuse the existing dataset/transform.

    train_fold_proba(train_df, test_df, timm_name="convnext_small") -> P(malignant)
"""
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

import config
from malignancy_model import _DS, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY


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


def train_fold_proba(train_df, test_df, timm_name, epochs=EPOCHS, device=None, log_prefix=""):
    device = device or config.DEVICE
    n_pos = float(train_df["y"].sum()); n_neg = float((train_df["y"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    tr = DataLoader(_DS(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te = DataLoader(_DS(test_df, False), batch_size=BATCH_SIZE, num_workers=0)

    model = timm.create_model(timm_name, pretrained=True, num_classes=1).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train(); tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            out = model(x)
            if out.ndim > 1 and out.shape[1] == 1:
                out = out
            loss = crit(out.view(-1, 1), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)

    p = _proba(model, te, device)
    _free(model, opt, sched, crit)
    return p
