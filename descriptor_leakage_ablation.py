"""
descriptor_leakage_ablation.py -- rebut the "descriptor stream leaks the BI-RADS
category" concern with numbers. The production descriptor stream deliberately EXCLUDES
target-derived fields (BI-RADS category, Interpretation, Diagnosis, Verification,
Classification) and uses only morphological descriptors. This quantifies:

  (a) morphology-only descriptors (the deployed leak-free stream)      -> expect 0.924
  (b) morphology + BI-RADS category (the leaky variant we DID NOT use) -> expect higher
  (c) BI-RADS category alone (a near-label baseline)

Leak-free 5-fold on BrEaST, same folds as evaluate_malignancy.

    python descriptor_leakage_ablation.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset, make_folds, DescriptorFeaturizer, N_FOLDS


def birads_onehot(train_df, test_df):
    """One-hot of the BI-RADS category, categories fitted on train only."""
    cats = sorted(v for v in train_df["birads"].unique() if str(v).strip())
    idx = {c: j for j, c in enumerate(cats)}
    def enc(df):
        X = np.zeros((len(df), len(cats)), dtype="float32")
        for i, v in enumerate(df["birads"].values):
            if v in idx:
                X[i, idx[v]] = 1.0
        return X
    return enc(train_df), enc(test_df)


def oof_auc(df, y, fold, build):
    """build(train_df, test_df) -> (Xtr, Xte); pooled OOF AUC of balanced logistic."""
    p = np.zeros(len(df))
    for k in range(N_FOLDS):
        tr, te = df[fold != k], df[fold == k]
        Xtr, Xte = build(tr, te)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, tr["y"].values)
        p[fold == k] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, p), p


def main():
    df = load_dataset().reset_index(drop=True)
    y = df["y"].values
    fold = make_folds(df)
    print(f"BrEaST n={len(df)} | malignant {int(y.sum())} | {N_FOLDS}-fold "
          f"| BI-RADS values: {sorted(v for v in df['birads'].unique() if str(v).strip())}\n")

    def morph(tr, te):
        f = DescriptorFeaturizer().fit(tr)
        return f.transform(tr), f.transform(te)

    def morph_plus_birads(tr, te):
        f = DescriptorFeaturizer().fit(tr)
        Xtr, Xte = f.transform(tr), f.transform(te)
        Btr, Bte = birads_onehot(tr, te)
        return np.hstack([Xtr, Btr]), np.hstack([Xte, Bte])

    def birads_only(tr, te):
        return birads_onehot(tr, te)

    variants = [
        ("(a) morphology only  [DEPLOYED, leak-free]", morph),
        ("(b) morphology + BI-RADS category  [leaky]", morph_plus_birads),
        ("(c) BI-RADS category alone  [near-label]", birads_only),
    ]
    print(f"{'descriptor variant':<46}  {'AUC':>6}")
    aucs = {}
    for name, build in variants:
        a, _ = oof_auc(df, y, fold, build)
        aucs[name[:3]] = a
        print(f"{name:<46}  {a:>6.3f}")

    print(f"\n  Inflation avoided by excluding BI-RADS category: "
          f"{aucs['(b)'] - aucs['(a)']:+.3f} AUC")
    print(f"  BI-RADS category alone already reaches {aucs['(c)']:.3f} -- i.e. it is close to a")
    print(f"  partial label, which is exactly why the deployed stream excludes it.")


if __name__ == "__main__":
    main()
