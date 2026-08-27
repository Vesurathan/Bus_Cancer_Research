"""
experiment_morph_fusion.py -- does the automated morphology stream strengthen the
fusion, especially the AUTONOMOUS (no manual descriptor) image-only pipeline?

Computes the morphology stream leak-free on the same folds as malig_streams.npz,
then reports stacked-fusion AUC for several stream combinations.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset, make_folds
from morphology_features import get_features, FEATURE_NAMES
from evaluate_malignancy import fuse_stack

IMAGES_DIR = "./dataset/images"


def morph_oof(df, fold):
    X = []
    for r in df.itertuples():
        mask = os.path.join(IMAGES_DIR, f"{r.case_id}_tumor.png")
        X.append(get_features(r.image_path, mask) if os.path.exists(mask)
                 else np.zeros(len(FEATURE_NAMES), dtype=np.float32))
    X = np.vstack(X)
    y = df["y"].values
    oof = np.zeros(len(y))
    for k in range(fold.max() + 1):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def main():
    df = load_dataset()
    fold = make_folds(df)
    y = df["y"].values
    z = np.load("malig_streams.npz", allow_pickle=True)
    streams = {"descriptor": z["descriptor"], "knn": z["knn"], "vit": z["vit"]}
    streams["morph"] = morph_oof(df, fold)
    np.save("morph_oof.npy", streams["morph"])

    def fuse(names):
        return fuse_stack({k: streams[k] for k in names}, y, fold)

    print("single streams (AUC):")
    for k in ("descriptor", "vit", "knn", "morph"):
        print(f"  {k:12} {roc_auc_score(y, streams[k]):.3f}")

    print("\nfusion combinations (AUC):")
    combos = [
        ("descriptor+knn+vit  (current full)", ["descriptor", "knn", "vit"]),
        ("descriptor+knn+vit+morph  (+auto)", ["descriptor", "knn", "vit", "morph"]),
        ("knn+vit  (current image-only)", ["knn", "vit"]),
        ("knn+vit+morph  (AUTONOMOUS, no radiologist)", ["knn", "vit", "morph"]),
    ]
    for label, names in combos:
        print(f"  {label:46} {roc_auc_score(y, fuse(names)):.3f}")


if __name__ == "__main__":
    main()
