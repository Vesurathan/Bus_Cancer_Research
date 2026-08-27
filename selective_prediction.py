"""
selective_prediction.py -- the core novelty experiment: is CROSS-MODAL
DISAGREEMENT (image evidence vs the radiologist's BI-RADS descriptors) a useful
signal for *when to trust the agent* vs *defer to a human*?

Reads the leak-free out-of-fold stream predictions cached by
evaluate_malignancy.py (./malig_streams.npz: descriptor, knn, vit, y, fold) and:

  1. builds image-only P_img (fuse vit+knn) and descriptor P_desc, and the full
     fused agent probability P_fused (cross-fitted stacked LR over all streams);
  2. defines abstention signals -- cross-modal disagreement |P_img - P_desc|,
     the standard confidence baseline |P_fused-0.5|, plus random/oracle bounds;
  3. reports risk-coverage curves + AURC (does abstaining raise accuracy on the
     auto-decided subset, and is disagreement competitive with / complementary
     to confidence?);
  4. tests a clinically meaningful policy -- "defer flagged (conflicting) cases to
     the descriptor/radiologist" -- and reports the accuracy gain;
  5. reports calibration (ECE) of the fused probability.

    python selective_prediction.py
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from evaluate_malignancy import fuse_stack, STREAM_CACHE, N_FOLDS

THRESH = 0.5
EPI_CACHE = "./malig_epistemic.npz"   # optional: MC-dropout epistemic uncertainty


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def image_prob(streams):
    """Image-only evidence: mean of vit + knn in logit space."""
    img = [s for s in ("vit", "knn") if s in streams]
    return _sig(np.mean([_logit(streams[s]) for s in img], axis=0))


def risk_coverage(correct, reliability):
    """Sort most-reliable-first; cumulative error as coverage grows 0->1."""
    order = np.argsort(-reliability)
    c = correct[order].astype(float)
    n = np.arange(1, len(c) + 1)
    cov = n / len(c)
    err = 1 - np.cumsum(c) / n
    return cov, err


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def aurc(correct, reliability):
    cov, err = risk_coverage(correct, reliability)
    return float(_trapz(err, cov))


def sel_acc_at(correct, reliability, coverage):
    cov, err = risk_coverage(correct, reliability)
    i = min(len(cov) - 1, max(0, int(round(coverage * len(cov))) - 1))
    return 1 - err[i]


def learned_deferral(feats, correct, fold):
    """Cross-fitted 'learning to defer': a meta-model predicts P(the agent is
    correct) from the evidence vector. Higher = keep. Leak-free -- fit on other
    folds, predict the held-out fold. Reliability score = predicted P(correct)."""
    out = np.zeros(len(correct))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if len(set(correct[tr])) < 2:
            out[te] = correct[tr].mean()
            continue
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(feats[tr], correct[tr])
        out[te] = m.predict_proba(feats[te])[:, 1]
    return out


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    z = np.load(STREAM_CACHE, allow_pickle=True)
    y, fold = z["y"].astype(int), z["fold"]
    streams = {k: z[k] for k in z.files if k not in ("y", "fold")}
    n = len(y)
    print(f"loaded streams {list(streams)}  (n={n}, malignant={int(y.sum())})")

    p_desc = streams["descriptor"]
    p_img = image_prob(streams)
    p_fused = fuse_stack(streams, y, fold)         # the agent's decision probability
    yhat = (p_fused >= THRESH).astype(int)
    correct = (yhat == y).astype(int)

    print(f"\nagent (stacked fusion): AUC {roc_auc_score(y, p_fused):.3f}  "
          f"acc {correct.mean():.3f}  ECE {ece(y, p_fused):.3f}")
    print(f"image-only AUC {roc_auc_score(y, p_img):.3f}   "
          f"descriptor AUC {roc_auc_score(y, p_desc):.3f}")

    # ---- abstention signals (higher reliability = keep / auto-decide) -------
    disagree = np.abs(p_img - p_desc)              # cross-modal conflict
    conf = np.abs(p_fused - 0.5)                    # standard confidence baseline
    stream_std = np.std(np.column_stack([streams[s] for s in streams]), axis=1)
    # evidence vector for the learned deferral model
    feats = np.column_stack([
        streams["descriptor"], streams["knn"], streams["vit"],
        p_img, p_fused, disagree, stream_std, conf,
    ])
    learned = learned_deferral(feats, correct, fold)   # P(correct); higher = keep
    rng = np.random.default_rng(0)
    signals = {
        "cross-modal disagreement": -disagree,
        "3-stream variance": -stream_std,
        "fusion confidence (baseline)": conf,
        "learned deferral (meta)": learned,
        "learned + disagreement": learned - 0.5 * disagree,
    }
    # Optional MC-dropout epistemic uncertainty (epistemic_vit.py). Included for a
    # complete, honest comparison -- in-distribution it does NOT beat confidence.
    if os.path.exists(EPI_CACHE):
        e = np.load(EPI_CACHE, allow_pickle=True)
        if np.array_equal(e["y"].astype(int), y):
            signals["epistemic (MC-dropout)"] = -e["u_epi"]
    signals["random"] = rng.random(n)
    signals["oracle (upper bound)"] = correct.astype(float)

    print("\nRisk-coverage — selective accuracy at each coverage (AURC: lower better)")
    print(f"  {'signal':32s}{'AURC':>7}{'cov100':>8}{'cov90':>7}{'cov80':>7}{'cov70':>7}")
    for name, rel in signals.items():
        a = aurc(correct, rel)
        row = [sel_acc_at(correct, rel, c) for c in (1.0, 0.9, 0.8, 0.7)]
        print(f"  {name:32s}{a:7.3f}" + "".join(f"{v:8.3f}" if i == 0 else f"{v:7.3f}"
                                                 for i, v in enumerate(row)))

    # ---- clinical policy: defer flagged (conflicting) cases to descriptor ----
    print("\n'Defer conflicting cases to the radiologist descriptors' policy:")
    for q in (0.10, 0.20, 0.30):
        k = int(round(q * n))
        flag = np.argsort(-disagree)[:k]           # most-conflicting q%
        # (a) auto-decide only the non-flagged; report their accuracy + coverage
        keep = np.ones(n, bool); keep[flag] = False
        auto_acc = correct[keep].mean()
        flag_err = 1 - correct[flag].mean()
        # (b) hybrid: on flagged cases trust the descriptor stream instead of fusion
        yhat_hy = yhat.copy()
        yhat_hy[flag] = (p_desc[flag] >= THRESH).astype(int)
        hy_acc = (yhat_hy == y).mean()
        print(f"  flag top {int(q*100):2d}%: auto-decide {keep.sum()}/{n} at "
              f"acc {auto_acc:.3f} (flagged-subset error {flag_err:.3f}); "
              f"hybrid defer-to-descriptor overall acc {hy_acc:.3f} "
              f"(vs {correct.mean():.3f} fusion-only)")


if __name__ == "__main__":
    main()
