"""
promote_biomedclip.py -- promote the BiomedCLIP-full image backbone into production.

Non-destructive: keeps the ImageNet ViT artifacts in place and adds a BiomedCLIP
checkpoint + a meta flag, so the change is reversible by flipping meta.json back.

Steps
  1. train BiomedCLIP-full on ALL BrEaST            -> prod/clip_malignancy.pt
  2. refit the image + full Platt calibrators using the leak-free BiomedCLIP OOF
     predictions (backbone_compare.npz 'full') as the image stream
  3. set prod/meta.json  image_backbone = "biomedclip"

Prereq: backbone_compare.npz (from compare_backbones.py, key 'full') and
malig_streams.npz (descriptor/knn/y/fold) must exist.

    python promote_biomedclip.py
"""
import json, os, pickle, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from malignancy_data import load_dataset
from malignancy_model_clip import build_model, _DS, EPOCHS, BATCH_SIZE, WEIGHT_DECAY
from calibrate_fusion import fit_platt, apply_platt, ece, brier
from evaluate_malignancy import fuse_stack

PROD = "./prod"


def train_clip_all(df, device):
    tr = DataLoader(_DS(df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    n_pos = float(df["y"].sum()); n_neg = float((df["y"] == 0).sum())
    pos_w = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    model = build_model(freeze="full").to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.trainable_param_groups(), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        model.train(); tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 4 == 0 or epoch == EPOCHS - 1:
            print(f"[prod-clip] epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model, opt, sched, crit
    return sd


def main():
    dev = config.DEVICE
    df = load_dataset()
    print(f"promoting BiomedCLIP-full on all {len(df)} cases ({int(df['y'].sum())} malignant)")

    # 1) train BiomedCLIP-full on all data
    sd = train_clip_all(df, dev)
    torch.save(sd, f"{PROD}/clip_malignancy.pt")
    print(f"saved {PROD}/clip_malignancy.pt")

    # 2) recalibrate using the leak-free BiomedCLIP OOF as the image stream
    s = np.load("malig_streams.npz"); b = np.load("backbone_compare.npz")
    y, fold = s["y"].astype(int), s["fold"]
    streams = {"descriptor": s["descriptor"], "knn": s["knn"], "vit": b["full"]}  # 'vit' key = image slot
    # back up the ImageNet calibrators once
    for f in ("calibrator_image.pkl", "calibrator_full.pkl"):
        src = f"{PROD}/{f}"
        if os.path.exists(src) and not os.path.exists(f"{PROD}/_imagenet_{f}"):
            shutil.copy(src, f"{PROD}/_imagenet_{f}")
    for name, keys, out in [("full", ("descriptor", "knn", "vit"), "calibrator_full.pkl"),
                            ("image-only", ("knn", "vit"), "calibrator_image.pkl")]:
        sub = {k: streams[k] for k in keys}
        p_oof = fuse_stack(sub, y, fold)
        cal = fit_platt(p_oof, y)
        p_cal = apply_platt(cal, p_oof)
        print(f"  {name:<11}: AUC {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y,p_oof):.3f}"
              f"  ECE {ece(y,p_oof):.3f}->{ece(y,p_cal):.3f}")
        with open(f"{PROD}/{out}", "wb") as f:
            pickle.dump({"platt": cal}, f)

    # 3) flip the backbone flag
    meta = json.load(open(f"{PROD}/meta.json"))
    meta["image_backbone"] = "biomedclip"
    meta["image_backbone_note"] = "BiomedCLIP-full; ImageNet ViT retained as vit_malignancy.pt"
    json.dump(meta, open(f"{PROD}/meta.json", "w"), indent=2)
    print(f"\nmeta.json image_backbone -> biomedclip. Production promoted (reversible).")


if __name__ == "__main__":
    main()
