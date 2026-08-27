"""
birads_model.py -- a real BI-RADS category predictor (replaces the crude
"estimate from malignancy probability" mapping). BI-RADS is ordinal
(1 < 2 < 3 < 4a < 4b < 4c < 5), so we report exact accuracy, within-one accuracy,
and mean absolute error (on the ordinal scale) -- the metrics that matter for an
ordinal target -- not just top-1.

Features: BiomedCLIP image embedding (+ descriptor multi-hot when available).
BrEaST only (BUSI has no BI-RADS), so this is a BrEaST-trained head; for a new
image with no descriptors it predicts from the image embedding alone.

    python birads_model.py            # leak-free 5-fold CV report
    python birads_model.py --train    # fit on all data -> prod/birads.pkl
"""
import sys, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import config
from malignancy_data import load_dataset, DescriptorFeaturizer
from evaluate_malignancy import embed_all

ORDER = ["1", "2", "3", "4a", "4b", "4c", "5"]
IDX = {b: i for i, b in enumerate(ORDER)}
PROD = "./prod"


def _prep():
    df = load_dataset()
    df = df[df["birads"].isin(ORDER)].reset_index(drop=True)
    y = np.array([IDX[b] for b in df["birads"]])
    E = embed_all(df)
    return df, E, y


def _metrics(y, pred):
    exact = np.mean(pred == y)
    within1 = np.mean(np.abs(pred - y) <= 1)
    mae = np.mean(np.abs(pred - y))
    return exact, within1, mae


def cv():
    df, E, y = _prep()
    print(f"n={len(y)} with BI-RADS | classes {ORDER}")
    feat = DescriptorFeaturizer()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for mode in ("image-only", "image+descriptors"):
        preds = np.zeros(len(y), int)
        for tr, te in skf.split(E, y):
            X = E.copy()
            if mode == "image+descriptors":
                feat.fit(df.iloc[tr])
                D = feat.transform(df)
                X = np.hstack([E, D])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        ex, w1, mae = _metrics(y, preds)
        print(f"  {mode:20s}: exact {ex:.3f}  within-1 {w1:.3f}  MAE {mae:.2f}")
    # a naive baseline: always predict the majority class
    maj = np.bincount(y).argmax()
    print(f"  {'majority baseline':20s}: exact {np.mean(y==maj):.3f}  "
          f"within-1 {np.mean(np.abs(y-maj)<=1):.3f}  MAE {np.mean(np.abs(y-maj)):.2f}")


def train():
    df, E, y = _prep()
    feat = DescriptorFeaturizer().fit(df)
    D = feat.transform(df)
    img = LogisticRegression(max_iter=3000, class_weight="balanced").fit(E, y)
    full = LogisticRegression(max_iter=3000, class_weight="balanced").fit(np.hstack([E, D]), y)
    with open(f"{PROD}/birads.pkl", "wb") as f:
        pickle.dump({"order": ORDER, "featurizer": feat,
                     "img": img, "full": full}, f)
    print(f"saved BI-RADS predictor -> {PROD}/birads.pkl")


if __name__ == "__main__":
    train() if "--train" in sys.argv else cv()
