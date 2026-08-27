"""
external_roi_multiseed.py -- robustness of the ROI + morphology enhancement over
multiple training seeds. Fixes the data split; varies only the ViT training seed
(the sole stochastic component) so baseline-vs-enhanced deltas are not confounded
by seed noise. Reports mean +/- std AUC per external site.
"""
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote
from external_roi_eval import breast_frame

SEEDS = [0, 1, 2]
dev = config.DEVICE


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def main():
    br, br_m = breast_frame()
    bu = pd.read_csv("busi_pairs.csv"); bu_m = np.load("busi_morph.npy")
    bb = pd.read_csv("busbra_pairs.csv"); bb_m = np.load("busbra_morph.npy")

    # fixed split across all seeds
    idx = np.arange(len(bu))
    tr_i, te_i = train_test_split(idx, test_size=0.30, stratify=bu["y"].values,
                                  random_state=config.SEED)
    train = pd.concat([br[["image_path", "roi_path", "y"]],
                       bu.iloc[tr_i][["image_path", "roi_path", "y"]]], ignore_index=True)
    trainM = np.vstack([br_m, bu_m[tr_i]])
    y_tr = train["y"].values

    # deterministic pieces computed once
    E_full = embed_paths(train["image_path"].tolist(), dev)
    E_roi = embed_paths(train["roi_path"].tolist(), dev)
    sc = StandardScaler().fit(trainM)
    morph_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(trainM), y_tr)

    tests = {
        "BUSI": (bu.iloc[te_i]["image_path"].values, bu.iloc[te_i]["roi_path"].values,
                 bu_m[te_i], bu.iloc[te_i]["y"].values),
        "BUS-BRA": (bb["image_path"].values, bb["roi_path"].values, bb_m, bb["y"].values),
    }
    results = {k: {"base": [], "enh": []} for k in tests}

    for s in SEEDS:
        set_seed(s)
        vf = train_vit_all(train[["image_path", "y"]].copy(), dev)
        torch.save(vf, "/tmp/vf.pt")
        set_seed(s + 100)
        vr = train_vit_all(train[["roi_path", "y"]].rename(columns={"roi_path": "image_path"}).copy(), dev)
        torch.save(vr, "/tmp/vr.pt")
        pf, pr = Predictor("/tmp/vf.pt", dev), Predictor("/tmp/vr.pt", dev)

        for name, (imgs, rois, mX, y) in tests.items():
            p_vf = np.array([pf.predict_proba(p) for p in imgs])
            p_kf = knn_vote(E_full, y_tr, embed_paths(list(imgs), dev))
            base = roc_auc_score(y, _mean_logit(p_vf, p_kf))
            p_vr = np.array([pr.predict_proba(p) for p in rois])
            p_kr = knn_vote(E_roi, y_tr, embed_paths(list(rois), dev))
            p_mo = morph_lr.predict_proba(sc.transform(mX))[:, 1]
            enh = roc_auc_score(y, _mean_logit(_mean_logit(p_vr, p_kr), p_mo))
            results[name]["base"].append(base); results[name]["enh"].append(enh)
            print(f"seed {s} {name:8} base {base:.3f} enh {enh:.3f}", flush=True)

    print("\n=== MULTI-SEED SUMMARY (mean +/- std over", len(SEEDS), "seeds) ===")
    for name in tests:
        b = np.array(results[name]["base"]); e = np.array(results[name]["enh"])
        print(f"  {name:8} baseline {b.mean():.3f}+/-{b.std():.3f} | "
              f"ENHANCED {e.mean():.3f}+/-{e.std():.3f} | delta {e.mean()-b.mean():+.3f}")
    np.save("multiseed_results.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()
