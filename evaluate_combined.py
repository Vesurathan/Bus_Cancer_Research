"""
evaluate_combined.py -- does training on a second dataset help the image model
generalise? Held-out protocol so the external number stays honest:

  * hold out a fixed 30% of BUSI as an untouched TEST set (stratified, seeded);
  * train the image streams (ViT + kNN) on BrEaST + the other 70% of BUSI;
  * compare, on the SAME held-out BUSI test split, the new combined model against
    the current BrEaST-only model (which never saw any BUSI).

Image-only (BUSI has no descriptors). Reports AUC and, at each model's ~0.90
sensitivity operating point, the sensitivity/specificity.

    python evaluate_combined.py
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from malignancy_data import load_dataset
from busi_data import load_busi
import malignancy_model
from malignancy_model import Predictor
from evaluate_malignancy import embed_all
from evaluate_external import embed as embed_paths

TEST_FRAC = 0.30


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def _sig(z): return 1 / (1 + np.exp(-z))


def knn_vote(E_bank, y_bank, E_q, k=7):
    sims = E_q @ E_bank.T
    out = np.zeros(len(E_q))
    for i in range(len(E_q)):
        nn = np.argsort(-sims[i])[:k]
        w = np.clip(sims[i, nn], 1e-6, None)
        out[i] = np.average(y_bank[nn], weights=w)
    return out


def sens_at(y, p, target=0.90):
    best_thr, best_spec = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 197):
        yh = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        se = tp / (tp + fn) if tp + fn else 0
        sp = tn / (tn + fp) if tn + fp else 0
        if se >= target and sp > best_spec:
            best_spec, best_thr = sp, thr
    yh = (p >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return best_thr, tp / (tp + fn), tn / (tn + fp)


def main():
    br = load_dataset()
    bu = load_busi()
    bu_tr, bu_te = train_test_split(bu, test_size=TEST_FRAC, stratify=bu["y"],
                                    random_state=config.SEED)
    print(f"BrEaST {len(br)} | BUSI train {len(bu_tr)} | BUSI held-out test {len(bu_te)} "
          f"(mal {int(bu_te['y'].sum())})")

    y_te = bu_te["y"].values

    # --- embeddings (BiomedCLIP) ---
    E_br = embed_all(br)
    E_bu = embed_paths(bu["image_path"].tolist(), config.DEVICE)
    idx = {p: i for i, p in enumerate(bu["image_path"])}
    E_bu_tr = np.vstack([E_bu[idx[p]] for p in bu_tr["image_path"]])
    E_bu_te = np.vstack([E_bu[idx[p]] for p in bu_te["image_path"]])

    # ================= OLD: BrEaST-only image model =================
    old_vit = Predictor("./prod/vit_malignancy.pt", config.DEVICE)
    p_vit_old = np.array([old_vit.predict_proba(p) for p in bu_te["image_path"]])
    p_knn_old = knn_vote(E_br, br["y"].values, E_bu_te)
    p_old = _sig(np.mean([_logit(p_vit_old), _logit(p_knn_old)], axis=0))

    # ================= NEW: BrEaST + 70% BUSI image model =================
    import pandas as pd
    comb = pd.concat([br[["image_path", "y"]], bu_tr[["image_path", "y"]]], ignore_index=True)
    p_vit_new = malignancy_model.train_fold_proba(comb, bu_te, log_prefix="[combined-vit] ")
    E_bank = np.vstack([E_br, E_bu_tr]); y_bank = np.concatenate([br["y"].values, bu_tr["y"].values])
    p_knn_new = knn_vote(E_bank, y_bank, E_bu_te)
    p_new = _sig(np.mean([_logit(p_vit_new), _logit(p_knn_new)], axis=0))

    # ================= compare on the SAME held-out BUSI test =================
    print(f"\n=== held-out BUSI test (n={len(y_te)}, image-only) ===")
    for name, pv, pk, pf in [("OLD  (BrEaST-only)", p_vit_old, p_knn_old, p_old),
                             ("NEW  (BrEaST+BUSI) ", p_vit_new, p_knn_new, p_new)]:
        thr, se, sp = sens_at(y_te, pf, 0.90)
        print(f"{name}: AUC vit {roc_auc_score(y_te,pv):.3f} knn {roc_auc_score(y_te,pk):.3f} "
              f"fused {roc_auc_score(y_te,pf):.3f} | @~90%sens thr {thr:.2f} sens {se:.3f} spec {sp:.3f}")


if __name__ == "__main__":
    main()
