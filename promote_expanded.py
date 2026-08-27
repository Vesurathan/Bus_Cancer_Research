"""
promote_expanded.py -- retrain the image streams on the EXPANDED multi-source
corpus (BrEaST + deduplicated BUSI + BUS-BRA) and evaluate on two independent
held-out external sites.

Split policy (nothing that tunes the model touches a test set):
    train        BrEaST(256) + BUSI 70%(539) + BUS-BRA folds 0-2
    calibrate    BUS-BRA fold 3            <- Platt + decision threshold only
    test A       BUSI 30%                  <- same seeded split as before
    test B       BUS-BRA fold 4            <- new external site
BUS-BRA folds are patient-level, so no patient spans two splits.

Artifacts -> ./prod_expanded/ (the existing ./prod/ is left untouched so the
before/after comparison stays reproducible).

    python promote_expanded.py
"""
import json, os, pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from combined_corpus import build
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from calibrate_fusion import fit_platt, apply_platt, ece
from promote_combined import _mean_logit, knn_vote, pick_threshold

OUT = "./prod_expanded"
CAL_FOLD = 3


def boot_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(name, y, p, thr):
    auc = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p, roc_auc_score)
    tn, fp, fn_, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn_), tn / (tn + fp)
    print(f"  {name:22} n={len(y):4}  AUC {auc:.3f} [{lo:.3f}-{hi:.3f}]  "
          f"sens {sens:.3f}  spec {spec:.3f}  (TN{tn} FP{fp} FN{fn_} TP{tp})")
    return {"n": int(len(y)), "auc": round(float(auc), 3),
            "auc_ci": [round(lo, 3), round(hi, 3)],
            "sens": round(float(sens), 3), "spec": round(float(spec), 3),
            "tn": int(tn), "fp": int(fp), "fn": int(fn_), "tp": int(tp)}


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = config.DEVICE
    d = build()

    bb_tr_all = d["busbra_train_df"]
    cal_df = bb_tr_all[bb_tr_all["fold"] == CAL_FOLD]
    bb_fit = bb_tr_all[bb_tr_all["fold"] != CAL_FOLD]

    train = pd.concat([
        d["breast"][["image_path", "y"]],
        d["train"][d["train"]["source"] == "busi"][["image_path", "y"]],
        bb_fit[["image_path", "y"]]], ignore_index=True)

    print(f"train {len(train)} (mal {int(train['y'].sum())}) | "
          f"calibrate {len(cal_df)} | testA BUSI {len(d['busi_holdout'])} | "
          f"testB BUS-BRA {len(d['busbra_holdout'])}\n", flush=True)

    # 1) ViT on the expanded corpus
    sd = train_vit_all(train, dev)
    torch.save(sd, f"{OUT}/vit_malignancy.pt")
    print("saved expanded ViT", flush=True)

    # 2) kNN bank from the same training corpus
    E_bank = embed_paths(train["image_path"].tolist(), dev)
    y_bank = train["y"].values
    np.savez(f"{OUT}/knn_bank.npz", E=E_bank, y=y_bank)
    print(f"saved kNN bank {E_bank.shape}", flush=True)

    # 3) image probability for any split
    vit = Predictor(f"{OUT}/vit_malignancy.pt", dev)

    def image_prob(df):
        p_v = np.array([vit.predict_proba(p) for p in df["image_path"]])
        p_k = knn_vote(E_bank, y_bank, embed_paths(df["image_path"].tolist(), dev))
        return _mean_logit(p_v, p_k)

    # 4) calibrate + threshold on the dedicated calibration split
    p_cal_raw = image_prob(cal_df)
    y_cal = cal_df["y"].values
    cal = fit_platt(p_cal_raw, y_cal)
    p_cal = apply_platt(cal, p_cal_raw)
    thr = pick_threshold(y_cal, p_cal, 0.90)
    with open(f"{OUT}/calibrator_image.pkl", "wb") as f:
        pickle.dump({"platt": cal}, f)
    print(f"\ncalibration split: ECE {ece(y_cal,p_cal_raw):.3f} -> {ece(y_cal,p_cal):.3f} | "
          f"threshold {thr:.3f}\n")

    # 5) evaluate on the two independent held-out sites
    print("HELD-OUT EXTERNAL RESULTS (expanded corpus)")
    res = {}
    for key, name in (("busi_holdout", "BUSI (site 1)"),
                      ("busbra_holdout", "BUS-BRA (site 2)")):
        df = d[key]
        p = apply_platt(cal, image_prob(df))
        res[key] = report(name, df["y"].values, p, thr)
        res[key]["ece"] = round(float(ece(df["y"].values, p)), 3)

    meta = {"threshold": round(float(thr), 3), "knn_k": 7,
            "trained_on": f"BrEaST(256) + BUSI-clean 70%({int((d['train']['source']=='busi').sum())}) "
                          f"+ BUS-BRA folds 0-2({len(bb_fit)}) = {len(train)} images",
            "calibrated_on": f"BUS-BRA fold {CAL_FOLD} ({len(cal_df)} images)",
            "results": res}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"\nwrote {OUT}/meta.json")


if __name__ == "__main__":
    main()
