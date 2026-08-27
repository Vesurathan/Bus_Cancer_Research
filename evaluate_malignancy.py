"""
evaluate_malignancy.py -- leak-free 5-fold CV for the agentic malignant-vs-benign
predictor. Trains each evidence stream on every fold's train split and pools the
held-out test predictions, then reports clinical metrics (AUC, sensitivity,
specificity, accuracy, F1) for each stream standalone AND for the fusion.

Streams:
  descriptor -- LogisticRegression over the radiologist's BI-RADS descriptors
                (strong; reported standalone so it never hides inside the fusion).
  knn        -- BiomedCLIP embedding -> cosine kNN over train -> similarity-
                weighted vote of neighbours' known labels (image-only).
  vit        -- (added next) a ViT malignancy head (image-only, learned).
  fusion     -- mean of the available per-case stream probabilities.

    python evaluate_malignancy.py            # descriptor + knn (fast)
    python evaluate_malignancy.py --vit      # + the ViT stream (trains per fold)
    python evaluate_malignancy.py --fusion-only   # reuse cached stream preds; retune fusion
"""
import os, sys, hashlib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from malignancy_data import load_dataset, make_folds, DescriptorFeaturizer, N_FOLDS

EMB_CACHE = "./malig_biomedclip_emb.npy"
STREAM_CACHE = "./malig_streams.npz"
KNN_K = 7


# ----------------------------- evidence streams ---------------------------- #
def descriptor_proba(train_df, test_df):
    feat = DescriptorFeaturizer().fit(train_df)
    Xtr, Xte = feat.transform(train_df), feat.transform(test_df)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, train_df["y"].values)
    return clf.predict_proba(Xte)[:, 1]


def embed_all(df):
    """BiomedCLIP image embeddings for every case (cached by the path list)."""
    key = hashlib.md5("|".join(df["image_path"]).encode()).hexdigest()[:8]
    cache = EMB_CACHE.replace(".npy", f"_{key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import torch, open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_ID)
    model = model.to(config.DEVICE).eval()
    embs = []
    with torch.no_grad():
        for p in df["image_path"]:
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(config.DEVICE)
            e = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
            embs.append(e.cpu().float().numpy()[0])
    E = np.vstack(embs).astype("float32")
    np.save(cache, E)
    return E


def knn_proba(E, y, train_idx, test_idx, k=KNN_K):
    Etr, Ete = E[train_idx], E[test_idx]
    ytr = y[train_idx]
    sims = Ete @ Etr.T                                   # cosine (rows normalised)
    out = np.zeros(len(test_idx))
    for i in range(len(test_idx)):
        nn = np.argsort(-sims[i])[:k]
        w = np.clip(sims[i, nn], 1e-6, None)
        out[i] = np.average(ytr[nn], weights=w)
    return out


# ------------------------------- metrics ----------------------------------- #
def clinical_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / len(y)
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    return dict(auc=auc, sens=sens, spec=spec, prec=prec, acc=acc, f1=f1,
               tp=tp, fp=fp, fn=fn, tn=tn)


def youden_threshold(y, p):
    order = np.argsort(-p)
    best_thr, best_j = 0.5, -1
    for t in np.unique(p):
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        if sens + spec - 1 > best_j:
            best_j, best_thr = sens + spec - 1, t
    return best_thr


def fuse_stack(streams, y, fold):
    """Leak-free learned fusion: the base predictions are already out-of-fold, and
    the meta LogisticRegression is itself cross-fitted (fit on other folds' base
    preds, predict the held-out fold), so no case informs its own fused score."""
    names = list(streams)
    P = np.column_stack([streams[s] for s in names])
    out = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        meta = LogisticRegression(max_iter=2000, class_weight="balanced")
        meta.fit(P[tr], y[tr])
        out[te] = meta.predict_proba(P[te])[:, 1]
    return out


def bootstrap_auc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def report(name, y, p):
    m = clinical_metrics(y, p, 0.5)
    thr = youden_threshold(y, p)
    mj = clinical_metrics(y, p, thr)
    lo, hi = bootstrap_auc_ci(y, p)
    print(f"\n== {name} ==")
    print(f"  AUC {m['auc']:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    print(f"  @0.50 : acc {m['acc']:.3f}  sens {m['sens']:.3f}  spec {m['spec']:.3f}  "
          f"prec {m['prec']:.3f}  F1 {m['f1']:.3f}  (TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']})")
    print(f"  @{thr:.2f}* : acc {mj['acc']:.3f}  sens {mj['sens']:.3f}  spec {mj['spec']:.3f}  "
          f"prec {mj['prec']:.3f}  F1 {mj['f1']:.3f}   (*Youden-J operating point)")
    return m


def main():
    use_vit = "--vit" in sys.argv
    fusion_only = "--fusion-only" in sys.argv

    if fusion_only:
        z = np.load(STREAM_CACHE, allow_pickle=True)
        y, fold = z["y"], z["fold"]
        streams = {k: z[k] for k in z.files if k not in ("y", "fold")}
        print(f"loaded cached streams {list(streams)} (n={len(y)})")
    else:
        df = load_dataset()
        y = df["y"].values
        fold = make_folds(df)
        n = len(df)
        print(f"cases {n} | malignant {int(y.sum())} | benign {int((y==0).sum())} | {N_FOLDS}-fold CV")

        E = embed_all(df)
        print(f"embedded {E.shape[0]} images (dim {E.shape[1]}) for kNN")

        streams = {"descriptor": np.zeros(n), "knn": np.zeros(n)}
        if use_vit:
            import malignancy_model
            streams["vit"] = np.zeros(n)

        for k in range(N_FOLDS):
            tr_idx = np.where(fold != k)[0]
            te_idx = np.where(fold == k)[0]
            tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
            streams["descriptor"][te_idx] = descriptor_proba(tr_df, te_df)
            streams["knn"][te_idx] = knn_proba(E, y, tr_idx, te_idx)
            if use_vit:
                streams["vit"][te_idx] = malignancy_model.train_fold_proba(
                    tr_df, te_df, log_prefix=f"[vit][fold {k}] ")
            print(f"  fold {k} done ({len(te_idx)} test)")

        np.savez(STREAM_CACHE, y=y, fold=fold, **streams)
        print(f"cached stream predictions -> {STREAM_CACHE}")

    for s in streams:
        report(s, y, streams[s])
    report("FUSION (mean)", y, np.mean([streams[s] for s in streams], axis=0))
    report("FUSION (stacked LR, cross-fitted)", y, fuse_stack(streams, y, fold))


if __name__ == "__main__":
    main()
