"""
malignancy_data.py -- data layer for the agentic breast-cancer (malignant vs
benign) predictor. Loads the BrEaST clinical table, builds a per-case dataset
(image path, binary label, BI-RADS, imaging descriptors), a leak-free stratified
5-fold split, and a train-fitted multi-hot featuriser for the descriptor stream.

Label: y = 1 if Classification == "malignant" else 0  (benign/normal -> 0), i.e.
malignant vs non-malignant. 98 positive / 158 negative on 256 cases.

The descriptor features are the radiologist's BI-RADS observations (shape, margin,
echogenicity, posterior features, halo, calcifications, skin thickening, tissue
composition) + age. NOTE: these are strong -- BI-RADS is derived from them -- so
the descriptor stream is reported standalone in the ablation, never hidden inside
the fusion.
"""
import os, glob, re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

DATA_DIR = "./dataset"
IMAGES_DIR = "./dataset/images"
CLINICAL_GLOB = "*clinical*"

# Imaging descriptor columns used as the "descriptor" evidence stream. Multi-value
# cells are '&'-separated. Target-derived columns (Interpretation, BIRADS,
# Diagnosis, Verification, Classification) are intentionally excluded.
DESCRIPTOR_COLS = [
    "Tissue_composition", "Shape", "Margin", "Echogenicity",
    "Posterior_features", "Halo", "Calcifications", "Skin_thickening",
]
N_FOLDS = 5


def _load_table():
    path = glob.glob(os.path.join(DATA_DIR, CLINICAL_GLOB))[0]
    for reader in (lambda p: pd.read_excel(p, engine="openpyxl"),
                   lambda p: pd.read_excel(p), lambda p: pd.read_csv(p)):
        try:
            return reader(path)
        except Exception:
            continue
    raise RuntimeError("could not read clinical table")


def _col(df, name):
    lut = {c.lower().replace(" ", "_"): c for c in df.columns}
    return lut.get(name.lower().replace(" ", "_"))


def _norm(v):
    return re.sub(r"\s+", " ", str(v).lower().strip())


def _num(v):
    """Parse a numeric age, tolerating strings like 'not available'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_dataset():
    """Return a DataFrame: image_path, case_id, y, birads, age, + descriptor cols.
    Only rows whose image file exists are kept."""
    df = _load_table()
    c_img = _col(df, "Image_filename")
    c_cls = _col(df, "Classification")
    c_bir = _col(df, "BIRADS")
    c_age = _col(df, "Age")

    rows = []
    for _, r in df.iterrows():
        img = os.path.join(IMAGES_DIR, str(r[c_img]))
        rec = {
            "image_path": img,
            "case_id": os.path.splitext(str(r[c_img]))[0],
            "y": 1 if _norm(r[c_cls]) == "malignant" else 0,
            "birads": _norm(r[c_bir]) if c_bir and pd.notna(r[c_bir]) else "",
            "age": _num(r[c_age]) if c_age and pd.notna(r[c_age]) else np.nan,
        }
        for name in DESCRIPTOR_COLS:
            c = _col(df, name)
            rec[name] = _norm(r[c]) if c and pd.notna(r[c]) else ""
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out[out["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return out


def make_folds(df, n_folds=N_FOLDS, seed=None):
    """Stratified fold id per row (by the malignancy label)."""
    seed = config.SEED if seed is None else seed
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(df, df["y"].values)):
        fold[test_idx] = k
    return fold


class DescriptorFeaturizer:
    """Multi-hot encoder over descriptor values (split on '&'), fitted on TRAIN
    only (leak-free), plus normalised age. transform() -> float32 matrix."""

    def __init__(self):
        self.vocab = []          # list of (col, value)
        self.age_mean = 0.0
        self.age_std = 1.0

    def _tokens(self, row):
        toks = []
        for col in DESCRIPTOR_COLS:
            for part in str(row[col]).split("&"):
                part = part.strip()
                if part:
                    toks.append((col, part))
        return toks

    def fit(self, df):
        seen = {}
        for _, r in df.iterrows():
            for t in self._tokens(r):
                seen[t] = seen.get(t, 0) + 1
        # keep values seen at least twice to avoid singleton noise
        self.vocab = sorted(t for t, c in seen.items() if c >= 2)
        ages = df["age"].dropna().values
        self.age_mean = float(ages.mean()) if len(ages) else 0.0
        self.age_std = float(ages.std()) or 1.0
        return self

    def transform(self, df):
        idx = {t: j for j, t in enumerate(self.vocab)}
        X = np.zeros((len(df), len(self.vocab) + 1), dtype="float32")
        for i, (_, r) in enumerate(df.iterrows()):
            for t in self._tokens(r):
                if t in idx:
                    X[i, idx[t]] = 1.0
            age = r["age"]
            X[i, -1] = 0.0 if pd.isna(age) else (age - self.age_mean) / self.age_std
        return X


if __name__ == "__main__":
    d = load_dataset()
    fold = make_folds(d)
    print(f"cases: {len(d)} | malignant: {int(d['y'].sum())} | benign/normal: {int((d['y']==0).sum())}")
    print("fold sizes:", np.bincount(fold).tolist())
    print("per-fold malignant count:",
          [int(d['y'].values[fold == k].sum()) for k in range(N_FOLDS)])
    f = DescriptorFeaturizer().fit(d)
    print(f"descriptor vocab size (>=2 occ): {len(f.vocab)}")
    print("birads distribution:", d["birads"].value_counts().to_dict())
