"""
evaluate_morphology.py -- is an AUTOMATED, image-derived descriptor stream (from
segmentation morphology) competitive with the radiologist's manual descriptors?

Leak-free stratified 5-fold CV on BrEaST: standardise features on train folds,
fit balanced logistic regression, pool out-of-fold predictions, report AUC.
Compares against the manual-descriptor stream (0.924) reported in the study.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from morphology_features import get_features, FEATURE_NAMES

IMAGES_DIR = "./dataset/images"


def load_breast_with_masks():
    f = glob.glob("dataset/*clinical*")[0]
    df = pd.read_excel(f)
    rows = []
    for _, r in df.iterrows():
        img = os.path.join(IMAGES_DIR, str(r["Image_filename"]))
        mask = os.path.join(IMAGES_DIR, str(r["Mask_tumor_filename"]))
        if not (os.path.exists(img) and os.path.exists(mask)):
            continue
        y = 1 if str(r["Classification"]).strip().lower() == "malignant" else 0
        rows.append({"image_path": img, "mask_path": mask, "y": y})
    return pd.DataFrame(rows)


def main():
    df = load_breast_with_masks()
    print(f"cases with masks: {len(df)} | malignant {int(df['y'].sum())}", flush=True)

    X = np.vstack([get_features(r.image_path, r.mask_path) for r in df.itertuples()])
    y = df["y"].values
    print(f"feature matrix {X.shape}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]

    auc = roc_auc_score(y, oof)
    print(f"\nAUTOMATED morphology descriptor stream: AUC {auc:.3f}")
    print("(manual radiologist-descriptor stream: 0.924 | image ViT: 0.813)")

    # feature importance (|coef| on full standardised data)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    order = np.argsort(-np.abs(lr.coef_[0]))
    print("\ntop discriminative features:")
    for i in order[:6]:
        print(f"  {FEATURE_NAMES[i]:24} coef {lr.coef_[0][i]:+.3f}")
    np.savez("morph_features.npz", X=X, y=y, oof=oof, names=FEATURE_NAMES)


if __name__ == "__main__":
    main()
