"""
findings_improve.py -- can we improve the weak findings section? The 28-term
classifier is data-limited (macro-F1 ~0.10, dragged by unlearnable tail terms).
Here we test whether fusing it with RETRIEVAL (terms mentioned by the most
similar train cases) recovers findings the classifier misses, and we tune the
operating threshold. Evaluated on the leak-free single-split test set; the
retrieval bank is the train split only.

Reports macro-F1 (all 28), micro-F1 (label-weighted), and head-tier macro-F1
for: classifier alone, retrieval alone, and the fusion -- at each method's best
threshold.

    python findings_improve.py
"""
import numpy as np
import pandas as pd

import config
from vocab import TERMS, TIER, SYNONYMS
from finding_model import Predictor
from retrieval_tool import Retriever
from evaluate import finding_f1


def gold_terms(text):
    t = (text or "").lower()
    return {term for term in TERMS if any(s in t for s in SYNONYMS[term])}


def retrieval_scores(exemplars):
    """Per-term score = max similarity among retrieved neighbours mentioning it."""
    s = {t: 0.0 for t in TERMS}
    for e in exemplars:
        rep = (e.get("report_text") or "").lower()
        sim = float(e.get("sim", 0.0))
        for t in TERMS:
            if any(syn in rep for syn in SYNONYMS[t]):
                s[t] = max(s[t], sim)
    return s


def macro_micro(pred_sets, gold_sets):
    macro = finding_f1(pred_sets, gold_sets)
    head = finding_f1(pred_sets, gold_sets, subset=[t for t in TERMS if TIER[t] == "head"])
    # micro
    tp = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    fp = sum(len(p - g) for p, g in zip(pred_sets, gold_sets))
    fn = sum(len(g - p) for p, g in zip(pred_sets, gold_sets))
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    micro = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return macro, micro, head


def best_threshold(prob_rows, gold_sets, grid):
    best = (0.5, -1)
    for thr in grid:
        preds = [{t for t in TERMS if pr[t] >= thr} for pr in prob_rows]
        m, _, _ = macro_micro(preds, gold_sets)
        if m > best[1]:
            best = (thr, m)
    return best[0]


def main():
    te = pd.read_csv(config.TEST_MANIFEST)
    clf = Predictor(config.CLF_CKPT, config.DEVICE)
    retr = Retriever()

    clf_rows, ret_rows, fus_rows, golds = [], [], [], []
    for _, r in te.iterrows():
        cp = clf.predict_proba(r["image_path"])
        ex = retr.retrieve(r["image_path"])
        rp = retrieval_scores(ex)
        fp = {t: max(cp[t], rp[t]) for t in TERMS}      # fusion = max evidence
        clf_rows.append(cp); ret_rows.append(rp); fus_rows.append(fp)
        golds.append(gold_terms(r["report_text"]))

    grid = np.round(np.arange(0.1, 0.85, 0.05), 2)
    print(f"findings on {len(golds)} leak-free test cases\n")
    print(f"  {'method':14s}{'thr':>5}{'macroF1':>9}{'microF1':>9}{'head-F1':>9}")
    for name, rows in [("classifier", clf_rows), ("retrieval", ret_rows), ("fusion", fus_rows)]:
        thr = best_threshold(rows, golds, grid)
        preds = [{t for t in TERMS if pr[t] >= thr} for pr in rows]
        ma, mi, hd = macro_micro(preds, golds)
        print(f"  {name:14s}{thr:>5.2f}{ma:>9.3f}{mi:>9.3f}{hd:>9.3f}")


if __name__ == "__main__":
    main()
