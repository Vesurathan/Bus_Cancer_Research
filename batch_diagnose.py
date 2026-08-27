"""
batch_diagnose.py -- sanity-check the DEPLOYED production model's decisions across
many cases: accuracy / sensitivity / specificity, the review-flag rate, and --
crucially -- whether the review gate actually catches the errors (accuracy on
auto-decided vs flagged cases).

    python batch_diagnose.py busi      # all BUSI, image-only (truly unseen)
    python batch_diagnose.py breast    # all BrEaST, image+descriptors (IN-SAMPLE)

Runs only the diagnosis (no PDF / findings / report) so it is fast: the model is
loaded once and reused across cases.
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from malignancy_predictor import MalignancyPredictor
from malignancy_data import load_dataset, DESCRIPTOR_COLS
from busi_data import load_busi
from diagnose import review_gate


def run(df, use_desc, note=""):
    pred = MalignancyPredictor()
    y, p, yhat, flag = [], [], [], []
    for i, (_, r) in enumerate(df.iterrows()):
        desc = None
        if use_desc:
            desc = {c: r[c] for c in DESCRIPTOR_COLS if str(r.get(c, ""))}
            desc["age"] = r.get("age", np.nan)
        dx = pred.predict(r["image_path"], desc)
        f, _ = review_gate(dx)
        y.append(int(r["y"])); p.append(dx["p_malignant"])
        yhat.append(dx["decision"] == "malignant"); flag.append(f)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(df)}", flush=True)
    y = np.array(y); p = np.array(p); yhat = np.array(yhat).astype(int); flag = np.array(flag)

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    acc = (tp + tn) / len(y)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

    print(f"\n=== {note} | n={len(y)} (mal {int(y.sum())}) ===")
    print(f"AUC {auc:.3f} | acc {acc:.3f} | sensitivity {sens:.3f} | specificity {spec:.3f}")
    print(f"confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")

    # operating-point sweep: for a cancer detector we care about sensitivity
    np.savez("./batch_probs.npz", y=y, p=p)
    print("\noperating-point sweep (threshold -> sensitivity / specificity):")
    for thr in (0.50, 0.40, 0.30, 0.25, 0.20, 0.15):
        yh = (p >= thr).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        se = tp2 / (tp2 + fn2) if tp2 + fn2 else 0
        sp = tn2 / (tn2 + fp2) if tn2 + fp2 else 0
        print(f"  thr {thr:.2f}: sens {se:.3f}  spec {sp:.3f}  "
              f"(misses {fn2} cancers, {fp2} false alarms)")

    # review-gate analysis: does flagging catch the errors?
    correct = (yhat == y)
    kept = ~flag
    print(f"\nreview gate: flagged {flag.sum()}/{len(y)} ({100*flag.mean():.0f}%) for human review")
    if kept.sum():
        print(f"  auto-decided (not flagged): {kept.sum()} cases, accuracy {correct[kept].mean():.3f}")
    if flag.sum():
        print(f"  flagged (deferred):        {flag.sum()} cases, accuracy {correct[flag].mean():.3f} "
              f"(these are the hard ones -> lower is expected/good)")
    err = ~correct
    if err.sum():
        print(f"  of {err.sum()} total errors, {int((err & flag).sum())} "
              f"({100*(err & flag).sum()/err.sum():.0f}%) were caught by the flag")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "busi"
    if which == "busi":
        run(load_busi(), use_desc=False, note="BUSI (unseen, image-only)")
    elif which == "breast":
        run(load_dataset(), use_desc=True, note="BrEaST (IN-SAMPLE, image+descriptors)")
    else:
        print("usage: python batch_diagnose.py [busi|breast]")


if __name__ == "__main__":
    main()
