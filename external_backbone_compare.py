"""
external_backbone_compare.py -- the decisive test of medical pretraining: train the
image malignancy stream ONLY on BrEaST (no target data), then measure cross-site
transfer to BUSI and BUS-BRA for the ImageNet ViT vs the BiomedCLIP backbone.

Pure transfer (no target leakage), so it isolates how well each backbone's features
generalise to unseen scanners/institutions -- exactly the cross-site gap.

    python external_backbone_compare.py imagenet last4
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from malignancy_data import load_dataset
from busi_data import load_busi
from busbra_data import load_busbra


def train_and_predict(mode, train_df, test_df, epochs=12):
    """Train the image model on train_df, return P(malignant) for test_df rows."""
    if mode == "imagenet":
        import malignancy_model as mm
        return mm.train_fold_proba(train_df, test_df, epochs=epochs, log_prefix=f"[{mode}] ")
    import os
    os.environ["CLIP_FREEZE"] = mode
    import importlib, malignancy_model_clip as mc
    importlib.reload(mc)
    return mc.train_fold_proba(train_df, test_df, epochs=epochs, log_prefix=f"[{mode}] ", freeze=mode)


def main():
    modes = sys.argv[1:] or ["imagenet", "last4"]
    br = load_dataset()[["image_path", "y"]]
    bu = load_busi()[["image_path", "y"]]
    bb = load_busbra()[["image_path", "y"]]
    print(f"train BrEaST n={len(br)} (mal {int(br['y'].sum())}) | "
          f"test BUSI n={len(bu)} (mal {int(bu['y'].sum())}) | "
          f"BUS-BRA n={len(bb)} (mal {int(bb['y'].sum())})\n", flush=True)

    # concat external tests so each backbone trains ONCE
    ext = pd.concat([bu, bb], ignore_index=True)
    n_bu = len(bu)

    rows = []
    for mode in modes:
        print(f"===== backbone: {mode} =====", flush=True)
        t0 = time.time()
        p = train_and_predict(mode, br, ext)
        auc_bu = roc_auc_score(bu["y"].values, p[:n_bu])
        auc_bb = roc_auc_score(bb["y"].values, p[n_bu:])
        print(f"  [{mode}] BUSI AUC {auc_bu:.3f} | BUS-BRA AUC {auc_bb:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        rows.append((mode, auc_bu, auc_bb))
        np.save(f"./ext_backbone_{mode}.npy", p)

    print("\n=========== BrEaST-only -> external transfer (image-only) ===========")
    print(f"{'backbone':<16}  {'BUSI AUC':>9}  {'BUS-BRA AUC':>12}")
    print(f"{'(diss. ImageNet)':<16}  {'0.835':>9}  {'~':>12}   <- reference")
    for mode, a_bu, a_bb in rows:
        tag = "ImageNet ViT" if mode == "imagenet" else f"BiomedCLIP {mode}"
        print(f"{tag:<16}  {a_bu:>9.3f}  {a_bb:>12.3f}")


if __name__ == "__main__":
    main()
