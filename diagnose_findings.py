"""
diagnose_findings.py -- explain the Finding-F1 number for a pooled CV run.

    python diagnose_findings.py [row] [--thr 0.60] [--low 0.15]
    # row defaults to "monolithic"; reads ./pred_cv_<row>.csv
    # or pass a CSV path directly:  python diagnose_findings.py pred_cv_full_llm.csv

Macro-F1 over the 28 interpretation terms is low because it averages many
near-unlearnable long-tail terms (each contributes 0) and because the fixed
report-coverage threshold under-fires an under-confident classifier. This script
makes that concrete: a threshold sweep, support-stratified macro-F1, micro-F1,
and a per-term table (support, how often the term is asserted, its probability
mass, and its P/R/F1). Read it alongside the per-tier Table 2 from evaluate_cv.

findings_json in the prediction CSV holds the classifier's per-term probabilities;
gold_terms holds the reference term set. Both are written by run_phase_a.
"""
import sys, json
import numpy as np
import pandas as pd

import config
from vocab import TERMS, TIER


def load(path):
    df = pd.read_csv(path)
    P = np.array([[json.loads(f).get(t, 0.0) for t in TERMS] for f in df["findings_json"]])
    G = np.zeros_like(P)
    for i, g in enumerate(df["gold_terms"]):
        for t in json.loads(g):
            if t in TERMS:
                G[i, TERMS.index(t)] = 1.0
    return df, P, G


def _counts(P, G, j, thr):
    pred, gt = P[:, j] >= thr, G[:, j] > 0
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return tp, fp, fn


def _f1(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def macro_f1(P, G, idx, thr):
    return float(np.mean([_f1(*_counts(P, G, j, thr))[0] for j in idx]))


def micro_f1(P, G, idx, thr):
    pred, gt = P[:, idx] >= thr, G[:, idx] > 0
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return _f1(tp, fp, fn)[0]


def main():
    args = [a for a in sys.argv[1:]]
    thr, low = config.HIGH_CONF, config.LOW_CONF
    if "--thr" in args:
        thr = float(args[args.index("--thr") + 1])
    if "--low" in args:
        low = float(args[args.index("--low") + 1])
    positional = [a for a in args if not a.startswith("--")
                  and a not in (str(thr), str(low))]
    row = positional[0] if positional else "monolithic"
    path = row if row.endswith(".csv") else f"./pred_cv_{row}.csv"

    df, P, G = load(path)
    sup = G.sum(0)
    idx_all = list(range(len(TERMS)))
    idx20 = [j for j in idx_all if sup[j] >= 20]
    idx5 = [j for j in idx_all if sup[j] >= 5]

    print(f"=== finding diagnostics: {path}  (n={len(df)}) ===")
    print(f"gold labels total {int(G.sum())} | avg {G.sum(1).mean():.2f} terms/case | "
          f"terms with support >=20: {len(idx20)}, >=5: {len(idx5)}, "
          f"==1: {int((sup == 1).sum())}, <=2: {int((sup <= 2).sum())}")

    print("\nThreshold sweep (macro-F1, all 28 terms):")
    for t in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        print(f"  thr={t:.2f}  macro-F1 {macro_f1(P, G, idx_all, t):.3f}")

    print(f"\nContext metrics        {'@%.2f' % thr:>8} {'@%.2f' % low:>8}")
    print(f"  macro-F1 (all 28)    {macro_f1(P,G,idx_all,thr):8.3f} {macro_f1(P,G,idx_all,low):8.3f}")
    print(f"  macro-F1 (sup>=20)   {macro_f1(P,G,idx20,thr):8.3f} {macro_f1(P,G,idx20,low):8.3f}")
    print(f"  macro-F1 (sup>=5)    {macro_f1(P,G,idx5,thr):8.3f} {macro_f1(P,G,idx5,low):8.3f}")
    print(f"  micro-F1 (all 28)    {micro_f1(P,G,idx_all,thr):8.3f} {micro_f1(P,G,idx_all,low):8.3f}")
    never = int(((P >= thr).sum(0) == 0).sum())
    print(f"\nterms never asserted at {thr:.2f}: {never}/{len(TERMS)} "
          f"(each = F1 0) -> hard macro ceiling {(len(TERMS)-never)/len(TERMS):.3f}")

    order = sorted(idx_all, key=lambda j: -sup[j])
    print(f"\n{'term':32s}{'tier':7s}{'gold':>5}{'pred':>5}{'maxp':>6}{'meanp+':>7}"
          f"{'prec':>6}{'rec':>6}{'F1':>6}")
    for j in order:
        tp, fp, fn = _counts(P, G, j, thr)
        f1, pr, rc = _f1(tp, fp, fn)
        gt = G[:, j] > 0
        meanp_pos = P[gt, j].mean() if gt.sum() else 0.0
        print(f"{TERMS[j][:31]:32s}{TIER[TERMS[j]]:7s}{int(sup[j]):>5}"
              f"{int((P[:,j]>=thr).sum()):>5}{P[:,j].max():>6.2f}{meanp_pos:>7.2f}"
              f"{pr:>6.2f}{rc:>6.2f}{f1:>6.3f}")


if __name__ == "__main__":
    main()
