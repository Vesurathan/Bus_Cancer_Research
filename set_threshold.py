"""
set_threshold.py -- choose the deployed decision threshold for a target
sensitivity, tuned on the LEAK-FREE in-distribution out-of-fold predictions
(never on the external BUSI test set), and write it into prod/meta.json.

For a cancer detector we set the highest threshold that still meets the target
sensitivity (so specificity is as high as possible without missing cancers).
Tuned on the image-only fusion -- the conservative path (image alone is weaker
than image+descriptors, and is the setting used for a new image with no form).

    python set_threshold.py [target_sensitivity]     # default 0.90
"""
import sys, json, pickle
import numpy as np
from sklearn.metrics import confusion_matrix

STREAMS = "./malig_streams.npz"
PROD = "./prod"


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
    z = np.load(STREAMS, allow_pickle=True)
    y = z["y"].astype(int)
    fi = pickle.load(open(f"{PROD}/fusion_image.pkl", "rb"))
    X = np.column_stack([z[s] for s in fi["order"]])       # OOF image streams
    p = fi["meta"].predict_proba(X)[:, 1]
    # apply the image calibrator if present, so the threshold is on the same
    # (calibrated) scale the deployed predictor uses
    import os
    if os.path.exists(f"{PROD}/calibrator_image.pkl"):
        cal = pickle.load(open(f"{PROD}/calibrator_image.pkl", "rb"))["platt"]
        lz = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        p = cal.predict_proba(lz.reshape(-1, 1))[:, 1]

    best_thr, best_spec = 0.05, -1.0
    for thr in np.linspace(0.01, 0.99, 197):
        yh = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens >= target and spec > best_spec:
            best_spec, best_thr = spec, thr

    # report the in-distribution operating point at the chosen threshold
    yh = (p >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    print(f"target sensitivity {target:.2f} -> threshold {best_thr:.3f}")
    print(f"in-distribution at this threshold: sens {sens:.3f}  spec {spec:.3f} "
          f"(misses {fn} cancers, {fp} false alarms of {len(y)} val cases)")

    meta = json.load(open(f"{PROD}/meta.json"))
    meta["threshold"] = round(float(best_thr), 3)
    meta["threshold_target_sensitivity"] = target
    json.dump(meta, open(f"{PROD}/meta.json", "w"), indent=2)
    print(f"wrote threshold {meta['threshold']} -> {PROD}/meta.json")


if __name__ == "__main__":
    main()
