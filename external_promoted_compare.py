"""
external_promoted_compare.py -- final deployed external evaluation for the promoted
system. Replicates the headline protocol (promote_combined.py / external_roi_eval.py):
train the image stream on BrEaST + 70% BUSI, form the image system as
mean-logit(ViT, BiomedCLIP-kNN), and evaluate on the held-out 30% BUSI partition and
all BUS-BRA -- for the OLD ImageNet ViT vs the promoted BiomedCLIP-full backbone.

Same seeded split (config.SEED, 30% test) as the dissertation, so numbers are
comparable to the reported 0.945 (BUSI held-out) / 0.865 (BUS-BRA).

    python external_promoted_compare.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset
from busi_data import load_busi
from busbra_data import load_busbra
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if len(set(y[i])) > 1]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def train_predict(mode, train_df, test_df):
    if mode == "imagenet":
        import malignancy_model as mm
        return mm.train_fold_proba(train_df, test_df, log_prefix=f"[{mode}] ")
    freeze = mode.split(":", 1)[1] if ":" in mode else "full"
    import os; os.environ["CLIP_FREEZE"] = freeze
    import importlib, malignancy_model_clip as mc; importlib.reload(mc)
    return mc.train_fold_proba(train_df, test_df, log_prefix=f"[{mode}] ", freeze=freeze)


def main():
    dev = config.DEVICE
    br = load_dataset()[["image_path", "y"]]
    bu = load_busi()[["image_path", "y"]]
    bb = load_busbra()[["image_path", "y"]]
    bu_tr, bu_te = train_test_split(bu, test_size=0.30, stratify=bu["y"],
                                    random_state=config.SEED)
    comb = pd.concat([br, bu_tr], ignore_index=True)
    print(f"train {len(comb)} (BrEaST {len(br)} + 70% BUSI {len(bu_tr)}) | "
          f"test: held-out BUSI {len(bu_te)}, BUS-BRA {len(bb)}\n", flush=True)

    ext = pd.concat([bu_te[["image_path", "y"]], bb], ignore_index=True)
    n_te = len(bu_te)
    y_busi, y_bb = bu_te["y"].values, bb["y"].values

    # kNN stream (BiomedCLIP embeddings) -- identical for both backbones
    print("embedding kNN bank + external sets ...", flush=True)
    E_bank = embed_paths(comb["image_path"].tolist(), dev); y_bank = comb["y"].values
    E_ext = embed_paths(ext["image_path"].tolist(), dev)
    p_knn = knn_vote(E_bank, y_bank, E_ext)

    import sys
    modes = sys.argv[1:] or ["imagenet", "clip:full"]
    rows = []
    for mode in modes:
        print(f"\n===== image backbone: {mode} =====", flush=True)
        t0 = time.time()
        p_vit = train_predict(mode, comb, ext)
        p_img = _mean_logit(p_vit, p_knn)                 # the deployed image system
        a_bu = roc_auc_score(y_busi, p_img[:n_te])
        a_bb = roc_auc_score(y_bb, p_img[n_te:])
        lo, hi = boot_ci(y_bb, p_img[n_te:])
        # ViT-only reference too
        a_bu_v = roc_auc_score(y_busi, p_vit[:n_te]); a_bb_v = roc_auc_score(y_bb, p_vit[n_te:])
        print(f"  [{mode}] image-system: BUSI {a_bu:.3f} | BUS-BRA {a_bb:.3f} "
              f"[{lo:.3f}-{hi:.3f}]   (ViT-only: BUSI {a_bu_v:.3f}, BUS-BRA {a_bb_v:.3f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        rows.append((mode, a_bu, a_bb, lo, hi))

    print("\n=========== DEPLOYED external evaluation (train BrEaST+70%BUSI) ===========")
    print(f"{'image system':<20}  {'BUSI held-out':>13}  {'BUS-BRA [95% CI]':>22}")
    print(f"{'(diss. ImageNet)':<20}  {'0.945':>13}  {'0.865 [0.825-0.904]':>22}")
    print(f"{'ImageNet ViT (prev)':<20}  {'0.934':>13}  {'0.839 [0.818-0.859]':>22}")
    print(f"{'BiomedCLIP-full (prev)':<20}  {'0.924':>13}  {'0.772 [0.748-0.795]':>22}")
    for mode, a_bu, a_bb, lo, hi in rows:
        tag = "ImageNet ViT" if mode == "imagenet" else f"BiomedCLIP {mode.split(':')[-1]}"
        print(f"{tag:<20}  {a_bu:>13.3f}  {f'{a_bb:.3f} [{lo:.3f}-{hi:.3f}]':>22}")


if __name__ == "__main__":
    main()
