"""
external_roi_eval.py -- do the ROI-crop + automated-morphology enhancements hold
CROSS-DATASET? Trains the baseline (full-frame ViT + k-NN) and the enhanced
(ROI-ViT + k-NN + morphology) autonomous pipelines on the SAME train split
(BrEaST + 70% BUSI) and evaluates both on the SAME held-out external sets
(30% BUSI and all BUS-BRA). Fair, leak-free, image-only.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset
from morphology_features import get_features
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote

dev = config.DEVICE


def breast_frame():
    from morphology_features import FEATURE_NAMES
    df = load_dataset()
    df["roi_path"] = df["case_id"].apply(lambda c: f"dataset/roi/{c}.png")
    df["mask_path"] = df["case_id"].apply(lambda c: f"dataset/images/{c}_tumor.png")
    X = np.vstack([
        get_features(r.image_path, r.mask_path) if os.path.exists(r.mask_path)
        else np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for r in df.itertuples()])
    return df[["image_path", "roi_path", "y"]].copy(), X


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if len(np.unique(y[i])) > 1]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    br, br_m = breast_frame()
    bu = pd.read_csv("busi_pairs.csv"); bu_m = np.load("busi_morph.npy")
    bb = pd.read_csv("busbra_pairs.csv"); bb_m = np.load("busbra_morph.npy")

    idx = np.arange(len(bu))
    tr_i, te_i = train_test_split(idx, test_size=0.30, stratify=bu["y"].values,
                                  random_state=config.SEED)
    train = pd.concat([br[["image_path", "roi_path", "y"]],
                       bu.iloc[tr_i][["image_path", "roi_path", "y"]]], ignore_index=True)
    trainM = np.vstack([br_m, bu_m[tr_i]])
    y_tr = train["y"].values
    print(f"train {len(train)} | test BUSI {len(te_i)} | test BUS-BRA {len(bb)}", flush=True)

    # ---- train both image classifiers -------------------------------------- #
    import torch
    full_df = train[["image_path", "y"]].copy()
    roi_df = train[["roi_path", "y"]].rename(columns={"roi_path": "image_path"}).copy()
    sd_full = train_vit_all(full_df, dev)
    torch.save(sd_full, "/tmp/vit_full.pt")
    sd_roi = train_vit_all(roi_df, dev)
    torch.save(sd_roi, "/tmp/vit_roi.pt")
    print("trained full-frame + ROI ViTs", flush=True)

    vit_full = Predictor("/tmp/vit_full.pt", dev)
    vit_roi = Predictor("/tmp/vit_roi.pt", dev)

    # ---- k-NN banks (full-frame + ROI embeddings) -------------------------- #
    E_full = embed_paths(train["image_path"].tolist(), dev)
    E_roi = embed_paths(train["roi_path"].tolist(), dev)

    # ---- morphology stream ------------------------------------------------- #
    sc = StandardScaler().fit(trainM)
    morph_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(trainM), y_tr)

    def evaluate(name, imgs, rois, morphX, y):
        p_vf = np.array([vit_full.predict_proba(p) for p in imgs])
        p_kf = knn_vote(E_full, y_tr, embed_paths(list(imgs), dev))
        base = _mean_logit(p_vf, p_kf)

        p_vr = np.array([vit_roi.predict_proba(p) for p in rois])
        p_kr = knn_vote(E_roi, y_tr, embed_paths(list(rois), dev))
        p_mo = morph_lr.predict_proba(sc.transform(morphX))[:, 1]
        # mean-logit of the three image-derived streams
        enh = _mean_logit(_mean_logit(p_vr, p_kr), p_mo)

        ab, bb_ = roc_auc_score(y, base), roc_auc_score(y, enh)
        lo, hi = boot_ci(y, enh)
        print(f"  {name:16} baseline {ab:.3f} | ENHANCED {bb_:.3f} [{lo:.3f}-{hi:.3f}]", flush=True)

    print("\nEXTERNAL RESULTS (autonomous, image-only):")
    evaluate("BUSI (site 1)", bu.iloc[te_i]["image_path"].values,
             bu.iloc[te_i]["roi_path"].values, bu_m[te_i], bu.iloc[te_i]["y"].values)
    evaluate("BUS-BRA (site 2)", bb["image_path"].values,
             bb["roi_path"].values, bb_m, bb["y"].values)


if __name__ == "__main__":
    main()
