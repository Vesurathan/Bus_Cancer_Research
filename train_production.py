"""
train_production.py -- train the malignancy evidence streams on ALL of BrEaST and
save deployable artifacts, so the agent (diagnose.py) can run on a brand-new image
(+ optional descriptors). This is the production counterpart to the per-fold,
leak-free evaluate_malignancy.py -- here we intentionally use all data to build the
final model, and reuse the leak-free out-of-fold predictions only to fit the
fusion meta-model (so the fusion weights still generalise).

Artifacts written to ./prod/ :
  vit_malignancy.pt   -- ViT-B/16 malignancy head (trained on all 256)
  descriptor.pkl      -- {featurizer, logreg} for the descriptor stream
  knn_bank.npz        -- BiomedCLIP embeddings + labels of the train corpus
  fusion_full.pkl     -- meta-LR over [descriptor, knn, vit]  (descriptors present)
  fusion_image.pkl    -- meta-LR over [knn, vit]              (image-only fallback)
  meta.json           -- threshold, stream order, class balance

    python train_production.py
"""
import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

import config
from malignancy_data import load_dataset, DescriptorFeaturizer
from malignancy_model import build_model, _DS, _free, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
from evaluate_malignancy import embed_all, STREAM_CACHE

PROD = "./prod"
THRESHOLD = 0.5


def train_vit_all(df, device):
    tr = DataLoader(_DS(df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    n_pos = float(df["y"].sum()); n_neg = float((df["y"] == 0).sum())
    pos_w = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    model = build_model().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        model.train(); tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[prod-vit] epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    _free(model, opt, sched, crit)
    return sd


def main():
    os.makedirs(PROD, exist_ok=True)
    dev = config.DEVICE
    df = load_dataset()
    print(f"training production streams on all {len(df)} cases "
          f"({int(df['y'].sum())} malignant)")

    # 1) descriptor stream (featurizer + logistic regression) on all data
    feat = DescriptorFeaturizer().fit(df)
    Xd = feat.transform(df)
    desc_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xd, df["y"].values)
    with open(f"{PROD}/descriptor.pkl", "wb") as f:
        pickle.dump({"featurizer": feat, "logreg": desc_lr}, f)
    print(f"saved descriptor stream ({len(feat.vocab)} features)")

    # 2) kNN bank: BiomedCLIP embeddings + labels of the whole train corpus
    E = embed_all(df)
    np.savez(f"{PROD}/knn_bank.npz", E=E, y=df["y"].values)
    print(f"saved kNN bank ({E.shape})")

    # 3) ViT malignancy head on all data
    sd = train_vit_all(df, dev)
    torch.save(sd, f"{PROD}/vit_malignancy.pt")
    print("saved ViT malignancy head")

    # 4) fusion meta-models fitted on the LEAK-FREE out-of-fold stream preds
    z = np.load(STREAM_CACHE, allow_pickle=True)
    y = z["y"].astype(int)
    P = {k: z[k] for k in ("descriptor", "knn", "vit")}
    full = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        np.column_stack([P["descriptor"], P["knn"], P["vit"]]), y)
    img = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        np.column_stack([P["knn"], P["vit"]]), y)
    with open(f"{PROD}/fusion_full.pkl", "wb") as f:
        pickle.dump({"meta": full, "order": ["descriptor", "knn", "vit"]}, f)
    with open(f"{PROD}/fusion_image.pkl", "wb") as f:
        pickle.dump({"meta": img, "order": ["knn", "vit"]}, f)
    print("saved fusion meta-models (full + image-only)")

    with open(f"{PROD}/meta.json", "w") as f:
        json.dump({"threshold": THRESHOLD,
                   "n_train": int(len(df)),
                   "n_malignant": int(df["y"].sum()),
                   "knn_k": 7}, f, indent=2)
    print(f"\nProduction artifacts written to {PROD}/")


if __name__ == "__main__":
    main()
