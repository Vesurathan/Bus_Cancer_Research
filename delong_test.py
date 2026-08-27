"""
delong_test.py -- DeLong test for two correlated ROC AUCs (Sun & Xu, 2014 fast
algorithm). Tests whether the stacked fusion (0.932) differs significantly from
the descriptor stream (0.924) on the same BrEaST cases, and the fusion versus the
image ViT as a sanity check.
"""
import numpy as np
from scipy import stats

from malignancy_data import load_dataset, make_folds
from evaluate_malignancy import fuse_stack


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds, y):
    # preds: (k, n) scores; y binary
    order = np.argsort(-y)
    y = y[order]; preds = preds[:, order]
    m = int(y.sum()); n = len(y) - m
    pos = preds[:, :m]; neg = preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_p(p1, p2, y):
    preds = np.vstack([p1, p2])
    aucs, cov = _fast_delong(preds, y.astype(float))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var + 1e-12)
    pval = 2 * stats.norm.sf(abs(z))
    return aucs, z, pval


def main():
    df = load_dataset(); fold = make_folds(df); y = df["y"].values
    z = np.load("malig_streams.npz", allow_pickle=True)
    streams = {"descriptor": z["descriptor"], "knn": z["knn"], "vit": z["vit"]}
    p_fuse = fuse_stack(streams, y, fold)

    for name, other in [("descriptor", z["descriptor"]), ("ViT", z["vit"])]:
        aucs, zstat, pval = delong_p(p_fuse, other, y)
        sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
        print(f"fusion ({aucs[0]:.3f}) vs {name} ({aucs[1]:.3f}): "
              f"ΔAUC {aucs[0]-aucs[1]:+.3f}, z={zstat:.2f}, p={pval:.3f}  -> {sig}")


if __name__ == "__main__":
    main()
