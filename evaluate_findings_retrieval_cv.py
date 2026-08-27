"""
evaluate_findings_retrieval_cv.py -- robust leak-free 5-fold evaluation of the
retrieval-based findings module (the LDAM replacement). Neighbours for each test
case come only from the same fold's train split. Pools predictions and reports
macro/micro/per-tier F1 against the LDAM baseline.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from vocab import TERMS
from evaluate_malignancy import embed_all
from train_finding_agent import labels_from_report

K, THR = 5, 0.20


def f1(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def main():
    df = pd.read_csv(config.MANIFEST_CSV).reset_index(drop=True)
    E = embed_all(df)
    Y = np.array([labels_from_report(t) for t in df["report_text"]])
    strata = Y.argmax(1)                       # rough stratification by dominant term
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    pred = np.zeros_like(Y)
    for tr, te in skf.split(df, strata):
        sims = E[te] @ E[tr].T
        for ii, i in enumerate(te):
            nn = np.argsort(-sims[ii])[:K]
            w = np.clip(sims[ii, nn], 1e-6, None)
            sc = np.average(Y[tr][nn], axis=0, weights=w)
            pred[i] = (sc >= THR).astype(int)

    tp = ((pred == 1) & (Y == 1)).sum(0).astype(float)
    fp = ((pred == 1) & (Y == 0)).sum(0).astype(float)
    fn = ((pred == 0) & (Y == 1)).sum(0).astype(float)
    per = f1(tp, fp, fn)
    n_j = Y.sum(0)
    tiers = {"head": np.where(n_j >= 10)[0], "medium": np.where((n_j >= 3) & (n_j < 10))[0],
             "tail": np.where((n_j >= 1) & (n_j < 3))[0]}
    macro = per.mean()
    micro = f1(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]))[0]
    print("LDAM-DRW baseline (single split): macro 0.093  micro 0.473  head 0.288  medium 0.000  tail 0.000")
    print(f"Retrieval (k={K}, thr={THR}), leak-free 5-fold pooled:")
    print(f"  macro {macro:.3f}  micro {micro:.3f}  "
          f"head {per[tiers['head']].mean():.3f}  "
          f"medium {per[tiers['medium']].mean():.3f}  "
          f"tail {per[tiers['tail']].mean():.3f}")


if __name__ == "__main__":
    main()
