"""
compare_backbones.py -- leak-free 5-fold comparison of image backbones for the
malignancy stream: current ImageNet ViT-B/16 vs BiomedCLIP (medical) at several
fine-tuning depths. Uses the SAME folds as evaluate_malignancy.py so the AUCs are
directly comparable. Pools out-of-fold predictions and reports AUC + sens/spec.

    python compare_backbones.py last2 last4        # sweep these freeze modes
    python compare_backbones.py imagenet           # re-run the ImageNet ViT baseline
"""
import sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

import config
from malignancy_data import load_dataset, make_folds, N_FOLDS

OUT = "./backbone_compare.npz"


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / len(y)
    auc = roc_auc_score(y, p)
    return auc, sens, spec, acc


def run_backbone(mode, df, y, fold):
    """mode: 'imagenet' or a BiomedCLIP freeze mode (head/last1/last2/last4/full)."""
    n = len(df)
    oof = np.zeros(n)
    if mode == "imagenet":
        import malignancy_model as mm
        trainer = mm.train_fold_proba
        kw = {}
    elif mode.startswith("timm:"):
        import backbone_zoo as bz
        name = mode.split(":", 1)[1]
        trainer = bz.train_fold_proba
        kw = {"timm_name": name}
    else:
        import os
        os.environ["CLIP_FREEZE"] = mode
        import importlib, malignancy_model_clip as mc
        importlib.reload(mc)
        trainer = mc.train_fold_proba
        kw = {"freeze": mode}
    t0 = time.time()
    for k in range(N_FOLDS):
        tr_idx = np.where(fold != k)[0]
        te_idx = np.where(fold == k)[0]
        oof[te_idx] = trainer(df.iloc[tr_idx], df.iloc[te_idx],
                              log_prefix=f"[{mode}][fold {k}] ", **kw)
        a, se, sp, ac = metrics(y[te_idx], oof[te_idx])
        print(f"  [{mode}] fold {k}: AUC {a:.3f}  ({len(te_idx)} test)  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return oof


def main():
    modes = sys.argv[1:] or ["last2", "last4"]
    df = load_dataset(); y = df["y"].values; fold = make_folds(df)
    print(f"cases {len(df)} | malignant {int(y.sum())} | benign {int((y==0).sum())} | "
          f"{N_FOLDS}-fold\n", flush=True)

    # load any existing results to append to
    try:
        prev = dict(np.load(OUT))
    except Exception:
        prev = {}

    results = {}
    for mode in modes:
        print(f"===== backbone: {mode} =====", flush=True)
        oof = run_backbone(mode, df, y, fold)
        prev[mode] = oof
        results[mode] = oof
        np.savez(OUT, y=y, fold=fold, **{k: v for k, v in prev.items()
                                          if k not in ("y", "fold")})

    print("\n================ SUMMARY (pooled out-of-fold, n=256) ================")
    print(f"{'backbone':<12}  {'AUC':>6}  {'sens':>6}  {'spec':>6}  {'acc':>6}")
    print(f"{'kNN (ref)':<12}  {'0.713':>6}")
    print(f"{'descr (ref)':<12}  {'0.924':>6}")
    for mode in modes:
        a, se, sp, ac = metrics(y, results[mode])
        tag = "ImageNet ViT" if mode == "imagenet" else f"BiomedCLIP {mode}"
        print(f"{tag:<12}  {a:>6.3f}  {se:>6.3f}  {sp:>6.3f}  {ac:>6.3f}")
    print(f"\nsaved OOF predictions -> {OUT}")


if __name__ == "__main__":
    main()
