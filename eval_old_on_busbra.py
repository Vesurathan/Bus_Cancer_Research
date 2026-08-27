"""Evaluate the ORIGINAL production model (./prod, trained on BrEaST + 70% BUSI)
on the BUS-BRA held-out fold. The original model has never seen any BUS-BRA
image, so this is a clean cross-site generalisation baseline to compare the
expanded model against."""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from combined_corpus import build
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote
from promote_expanded import boot_ci

dev = config.DEVICE
d = build()

for split_key, name in (("busbra_holdout", "BUS-BRA fold 4"),):
    df = d[split_key]
    y = df["y"].values
    vit = Predictor("./prod/vit_malignancy.pt", dev)
    bank = np.load("./prod/knn_bank.npz")
    p_v = np.array([vit.predict_proba(p) for p in df["image_path"]])
    p_k = knn_vote(bank["E"], bank["y"], embed_paths(df["image_path"].tolist(), dev))
    p = _mean_logit(p_v, p_k)
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p, roc_auc_score)
    # sensitivity-matched operating point for a like-for-like comparison
    thr = np.quantile(p, 1 - 0.818)          # match the expanded model's sensitivity
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    print(f"ORIGINAL model on {name} (never seen): n={len(y)}  "
          f"AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]")
    print(f"   at matched operating point: sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f}")
