"""
calibrate_fusion.py -- make the reported confidence trustworthy. The fusion
outputs a probability, but it is not calibrated (ECE ~0.09), so "confidence 36%"
does not literally mean 36%. We fit a Platt calibrator (a 1-parameter logistic on
the fused log-odds) on the LEAK-FREE cross-fitted out-of-fold predictions, for
both the full (image+descriptor) and image-only fusions, and save them for the
production predictor to apply. Reports ECE and Brier before/after.

    python calibrate_fusion.py
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluate_malignancy import fuse_stack, STREAM_CACHE

PROD = "./prod"


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_oof, y):
    """Platt scaling: logistic on the fused log-odds -> calibrated probability."""
    cal = LogisticRegression(C=1e6, solver="lbfgs")  # near-unpenalised 1D fit
    cal.fit(_logit(p_oof).reshape(-1, 1), y)
    return cal


def apply_platt(cal, p):
    return cal.predict_proba(_logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def main():
    z = np.load(STREAM_CACHE, allow_pickle=True)
    y, fold = z["y"].astype(int), z["fold"]
    streams = {k: z[k] for k in ("descriptor", "knn", "vit")}

    for name, keys, out in [
        ("full (image+descriptor)", ("descriptor", "knn", "vit"), "calibrator_full.pkl"),
        ("image-only", ("knn", "vit"), "calibrator_image.pkl"),
    ]:
        sub = {k: streams[k] for k in keys}
        p_oof = fuse_stack(sub, y, fold)               # leak-free fused probs
        cal = fit_platt(p_oof, y)
        p_cal = apply_platt(cal, p_oof)
        print(f"\n{name}:")
        print(f"  before: ECE {ece(y,p_oof):.3f}  Brier {brier(y,p_oof):.3f}")
        print(f"  after : ECE {ece(y,p_cal):.3f}  Brier {brier(y,p_cal):.3f}")
        with open(f"{PROD}/{out}", "wb") as f:
            pickle.dump({"platt": cal}, f)
        print(f"  saved {PROD}/{out}")


if __name__ == "__main__":
    main()
