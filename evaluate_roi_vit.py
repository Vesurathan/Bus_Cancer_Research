"""
evaluate_roi_vit.py -- leak-free 5-fold CV of the image ViT trained on ROI-cropped
BrEaST images, to test whether lesion cropping strengthens the weak image stream
(baseline full-frame ViT AUC 0.813). Saves OOF predictions for re-fusion.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset, make_folds
from malignancy_model import train_fold_proba


def main():
    df = load_dataset().copy()
    fold = make_folds(df)
    # point the loader at the ROI-cropped images
    df["image_path"] = df["case_id"].apply(lambda c: f"dataset/roi/{c}.png")
    y = df["y"].values

    oof = np.zeros(len(y))
    for k in range(fold.max() + 1):
        tr = df[fold != k].reset_index(drop=True)
        te_idx = np.where(fold == k)[0]
        te = df[fold == k].reset_index(drop=True)
        oof[te_idx] = train_fold_proba(tr, te, log_prefix=f"[roi f{k}] ")
        print(f"fold {k}: partial AUC {roc_auc_score(y[fold!=k] if False else y, oof) if oof.any() else 0:.3f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"\nROI-cropped image ViT: AUC {auc:.3f}  (full-frame baseline 0.813)")
    np.save("vit_roi_oof.npy", oof)


if __name__ == "__main__":
    main()
