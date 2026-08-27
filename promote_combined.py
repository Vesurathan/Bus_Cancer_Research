"""
promote_combined.py -- promote the combined-data image model to production.

Trains the image streams on BrEaST + 70% BUSI, and calibrates + sets the decision
threshold on the held-out 30% BUSI (the same seeded split as evaluate_combined.py,
so nothing that tunes the deployed model has been trained on). Image fusion is
plain mean-logit of ViT+kNN (matches the 0.942 held-out result), so the fusion
itself is not fitted on the held-out set -- only the 1-2 calibration params and
the threshold are.

Overwrites ./prod/ : vit_malignancy.pt, knn_bank.npz, calibrator_image.pkl,
meta.json. The descriptor stream (descriptor.pkl) is unchanged; at inference the
descriptor probability is simply averaged with the calibrated image probability.

    python promote_combined.py
"""
import json, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from malignancy_data import load_dataset
from busi_data import load_busi
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_malignancy import embed_all
from evaluate_external import embed as embed_paths
from calibrate_fusion import fit_platt, apply_platt, ece

PROD = "./prod"
TEST_FRAC = 0.30


def _mean_logit(a, b):
    la = np.log(np.clip(a, 1e-6, 1 - 1e-6) / (1 - np.clip(a, 1e-6, 1 - 1e-6)))
    lb = np.log(np.clip(b, 1e-6, 1 - 1e-6) / (1 - np.clip(b, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(la + lb) / 2))


def knn_vote(E_bank, y_bank, E_q, k=7):
    sims = E_q @ E_bank.T
    out = np.zeros(len(E_q))
    for i in range(len(E_q)):
        nn = np.argsort(-sims[i])[:k]
        w = np.clip(sims[i, nn], 1e-6, None)
        out[i] = np.average(y_bank[nn], weights=w)
    return out


def pick_threshold(y, p, target=0.90):
    best_thr, best_spec = 0.05, -1
    for thr in np.linspace(0.01, 0.99, 197):
        yh = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        se = tp / (tp + fn) if tp + fn else 0
        sp = tn / (tn + fp) if tn + fp else 0
        if se >= target and sp > best_spec:
            best_spec, best_thr = sp, thr
    return best_thr


def main():
    dev = config.DEVICE
    br = load_dataset()
    bu = load_busi()
    bu_tr, bu_te = train_test_split(bu, test_size=TEST_FRAC, stratify=bu["y"],
                                    random_state=config.SEED)
    comb = pd.concat([br[["image_path", "y"]], bu_tr[["image_path", "y"]]], ignore_index=True)
    print(f"combined train {len(comb)} (BrEaST {len(br)} + BUSI {len(bu_tr)}) | "
          f"held-out BUSI {len(bu_te)} (mal {int(bu_te['y'].sum())})")

    # 1) combined ViT
    sd = train_vit_all(comb, dev)
    torch.save(sd, f"{PROD}/vit_malignancy.pt")
    print("saved combined ViT")

    # 2) combined kNN bank
    E_br = embed_all(br)
    E_bu = embed_paths(bu["image_path"].tolist(), dev)
    pos = {p: i for i, p in enumerate(bu["image_path"])}
    E_bu_tr = np.vstack([E_bu[pos[p]] for p in bu_tr["image_path"]])
    E_bu_te = np.vstack([E_bu[pos[p]] for p in bu_te["image_path"]])
    E_bank = np.vstack([E_br, E_bu_tr])
    y_bank = np.concatenate([br["y"].values, bu_tr["y"].values])
    np.savez(f"{PROD}/knn_bank.npz", E=E_bank, y=y_bank)
    print(f"saved combined kNN bank {E_bank.shape}")

    # 3) held-out predictions -> calibrate + threshold (nothing trained on bu_te)
    vit = Predictor(f"{PROD}/vit_malignancy.pt", dev)
    p_vit = np.array([vit.predict_proba(p) for p in bu_te["image_path"]])
    p_knn = knn_vote(E_bank, y_bank, E_bu_te)
    y_te = bu_te["y"].values
    p_img = _mean_logit(p_vit, p_knn)
    auc = roc_auc_score(y_te, p_img)

    cal = fit_platt(p_img, y_te)
    p_cal = apply_platt(cal, p_img)
    with open(f"{PROD}/calibrator_image.pkl", "wb") as f:
        pickle.dump({"platt": cal}, f)
    thr = pick_threshold(y_te, p_cal, 0.90)

    yh = (p_cal >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yh, labels=[0, 1]).ravel()
    print(f"\nheld-out BUSI: AUC {auc:.3f} | ECE {ece(y_te,p_img):.3f}->{ece(y_te,p_cal):.3f} | "
          f"@thr {thr:.3f} sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f}")

    meta = {"threshold": round(float(thr), 3), "knn_k": 7,
            "threshold_target_sensitivity": 0.90,
            "trained_on": "BrEaST(256) + 70% BUSI(546)",
            "image_validated_on": "held-out 30% BUSI(234)",
            "held_out_auc": round(float(auc), 3)}
    json.dump(meta, open(f"{PROD}/meta.json", "w"), indent=2)
    print(f"wrote meta.json (threshold {meta['threshold']})")
    # stale artifacts from the old BrEaST-only fusion are no longer used
    for f in ("fusion_full.pkl", "fusion_image.pkl", "calibrator_full.pkl"):
        import os
        if os.path.exists(f"{PROD}/{f}"):
            os.rename(f"{PROD}/{f}", f"{PROD}/_legacy_{f}")
    print("done -- production promoted to the combined image model")


if __name__ == "__main__":
    main()
