"""
evaluate.py -- fills Table 1 (aggregate) and Table 2 (per-tier Finding-F1) plus
calibration (ECE, Brier). Run after run_phase_a.py has written predictions.

    exec(open('evaluate.py').read())

Metrics:
  BLEU-4, ROUGE-L, METEOR   -- report-quality (sacrebleu, rouge-score, nltk)
  Finding-F1 (macro)        -- predicted vs gold term sets, overall and per tier
  ECE, Brier                -- calibration of the finding probabilities
CIDEr is optional (pycocoevalcap); skipped if not installed.
"""
import json
import numpy as np
import pandas as pd
import config
from vocab import TERMS, TIER

import sacrebleu
from rouge_score import rouge_scorer
import nltk
for pkg in ["wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
from nltk.translate.meteor_score import meteor_score


def _asserted(findings, thresh=None):
    thresh = config.HIGH_CONF if thresh is None else thresh
    return {t for t in TERMS if findings.get(t, 0.0) >= thresh}


def finding_f1(pred_sets, gold_sets, subset=None):
    """Macro F1 over the (optionally tier-restricted) term set."""
    terms = subset if subset is not None else TERMS
    f1s = []
    for t in terms:
        tp = sum((t in p) and (t in g) for p, g in zip(pred_sets, gold_sets))
        fp = sum((t in p) and (t not in g) for p, g in zip(pred_sets, gold_sets))
        fn = sum((t not in p) and (t in g) for p, g in zip(pred_sets, gold_sets))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def calibration(prob_label_pairs, n_bins=10):
    """Flattened multi-label ECE and Brier over all (prob, label) pairs."""
    if not prob_label_pairs:
        return 0.0, 0.0
    probs = np.array([p for p, _ in prob_label_pairs])
    labels = np.array([y for _, y in prob_label_pairs])
    brier = float(np.mean((probs - labels) ** 2))
    ece, bins = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.sum():
            ece += (m.sum() / len(probs)) * abs(probs[m].mean() - labels[m].mean())
    return float(ece), brier


def main():
    df = pd.read_csv(config.PRED_OUT)
    preds = df["pred_report"].fillna("").tolist()
    golds = df["gold_report"].fillna("").tolist()

    # ---- report-quality (aggregate) ------------------------------------- #
    bleu = sacrebleu.corpus_bleu(preds, [golds]).score
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = 100 * np.mean([rs.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)])
    meteor = 100 * np.mean([
        meteor_score([g.split()], p.split()) for g, p in zip(golds, preds)
    ])

    # ---- finding sets --------------------------------------------------- #
    pred_sets = [_asserted(json.loads(f)) for f in df["findings_json"]]
    gold_sets = [set(json.loads(g)) for g in df["gold_terms"]]

    ff1_all = finding_f1(pred_sets, gold_sets)
    by_tier = {
        tier: finding_f1(pred_sets, gold_sets,
                         subset=[t for t in TERMS if TIER[t] == tier])
        for tier in ["head", "medium", "tail"]
    }

    # ---- calibration ---------------------------------------------------- #
    pairs = []
    for f, g in zip(df["findings_json"], gold_sets):
        probs = json.loads(f)
        for t in TERMS:
            pairs.append((float(probs.get(t, 0.0)), 1.0 if t in g else 0.0))
    ece, brier = calibration(pairs)

    print(f"\n=== config: {df['config'].iloc[0]}  (n={len(df)}) ===")
    print("Table 1 (aggregate):")
    print(f"  BLEU-4      {bleu:6.2f}")
    print(f"  ROUGE-L     {rougeL:6.2f}")
    print(f"  METEOR      {meteor:6.2f}")
    print(f"  Finding-F1  {ff1_all:6.3f}")
    print(f"  ECE         {ece:6.3f}   (lower better)")
    print(f"  Brier       {brier:6.3f}   (lower better)")
    print("\nTable 2 (per-tier Finding-F1):")
    for tier in ["head", "medium", "tail"]:
        print(f"  {tier:>6}     {by_tier[tier]:6.3f}")
    print(f"  {'macro':>6}     {np.mean(list(by_tier.values())):6.3f}")


if __name__ == "__main__":
    main()
