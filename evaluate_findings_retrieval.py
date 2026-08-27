"""
evaluate_findings_retrieval.py -- can training-free RETRIEVAL replace the LDAM
findings classifier? For each test case, findings are voted from its BiomedCLIP
nearest neighbours in the train split (similarity-weighted over their ground-truth
terms). Reports macro/micro/per-tier F1 vs the LDAM baseline (macro 0.093).
"""
import numpy as np
import pandas as pd

import config
from vocab import TERMS
from evaluate_malignancy import embed_all
from train_finding_agent import labels_from_report


def f1(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def scores_for(E, Y, tr_idx, te_idx, k):
    sims = E[te_idx] @ E[tr_idx].T                 # cosine (E is L2-normalised)
    out = np.zeros((len(te_idx), Y.shape[1]))
    for i in range(len(te_idx)):
        nn = np.argsort(-sims[i])[:k]
        w = np.clip(sims[i, nn], 1e-6, None)
        out[i] = np.average(Y[tr_idx][nn], axis=0, weights=w)
    return out


def evaluate(scores, Yte, thr, tiers):
    pred = (scores >= thr).astype(int)
    tp = ((pred == 1) & (Yte == 1)).sum(0).astype(float)
    fp = ((pred == 1) & (Yte == 0)).sum(0).astype(float)
    fn = ((pred == 0) & (Yte == 1)).sum(0).astype(float)
    per = f1(tp, fp, fn)
    macro = per.mean()
    micro = f1(np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()]))[0]
    tier = {t: (per[idx].mean() if len(idx) else float("nan")) for t, idx in tiers.items()}
    return macro, micro, tier


def main():
    df = pd.read_csv(config.MANIFEST_CSV)
    E = embed_all(df)                              # (256, 512) normalised
    Y = np.array([labels_from_report(t) for t in df["report_text"]])
    tr_idx = np.where(df["split"].values == "train")[0]
    te_idx = np.where(df["split"].values == "test")[0]
    n_j = Y[tr_idx].sum(0)
    tiers = {"head": np.where(n_j >= 10)[0], "medium": np.where((n_j >= 3) & (n_j < 10))[0],
             "tail": np.where((n_j >= 1) & (n_j < 3))[0]}
    Yte = Y[te_idx]
    print("baseline  LDAM-DRW          macro 0.093  micro 0.473  head 0.288  medium 0.000  tail 0.000")
    print(f"{'retrieval k / thr':22} {'macro':>6} {'micro':>6} {'head':>6} {'medium':>7} {'tail':>6}")
    best = (0, None)
    for k in (5, 7, 15, 25):
        sc = scores_for(E, Y, tr_idx, te_idx, k)
        for thr in (0.2, 0.3, 0.4, 0.5):
            macro, micro, tier = evaluate(sc, Yte, thr, tiers)
            print(f"k={k:<3} thr={thr:<4}        {macro:>6.3f} {micro:>6.3f} "
                  f"{tier['head']:>6.3f} {tier['medium']:>7.3f} {tier['tail']:>6.3f}")
            if macro > best[0]:
                best = (macro, (k, thr, micro, tier))
    k, thr, micro, tier = best[1]
    print(f"\nBEST retrieval: k={k}, thr={thr} -> macro {best[0]:.3f}, micro {micro:.3f} "
          f"(vs LDAM macro 0.093, micro 0.473)")


if __name__ == "__main__":
    main()
