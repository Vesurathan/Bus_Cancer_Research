"""
combined_corpus.py -- assemble the expanded multi-source training corpus.

Sources
    BrEaST    256 cases   Poland   (descriptors + BI-RADS)   internal
    BUSI      771 images  Egypt    (deduplicated)            external site 1
    BUS-BRA  1875 images  Brazil   (BI-RADS, 4 scanners)     external site 2

Split policy -- nothing that tunes the model may touch a held-out set:
    * BUSI     : the SAME seeded 70/30 split as promote_combined.py, so the
                 held-out 30% stays byte-identical to the previously reported
                 external test set and the before/after comparison is fair.
    * BUS-BRA  : official PATIENT-LEVEL folds; folds 0-3 train, fold 4 held out.
                 Grouping by patient is essential -- 1,875 images come from only
                 1,064 patients (mean 1.76 views each).

    python combined_corpus.py     # counts + leakage assertions
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from malignancy_data import load_dataset
from busi_data import load_busi
from busbra_data import load_busbra

BUSI_CLEAN_CSV = "busi_clean.csv"
TEST_FRAC = 0.30
BUSBRA_HOLDOUT_FOLD = 4


def load_busi_clean():
    """Deduplicated BUSI if dedupe_busi.py has been run, else raw (with a warning)."""
    if os.path.exists(BUSI_CLEAN_CSV):
        return pd.read_csv(BUSI_CLEAN_CSV)
    print("WARNING: busi_clean.csv missing -- run dedupe_busi.py; using raw BUSI")
    return load_busi()


def build():
    br = load_dataset()
    bu = load_busi_clean()
    bb = load_busbra()

    bu_tr, bu_te = train_test_split(bu, test_size=TEST_FRAC, stratify=bu["y"],
                                    random_state=config.SEED)
    bb_tr = bb[bb["fold"] != BUSBRA_HOLDOUT_FOLD]
    bb_te = bb[bb["fold"] == BUSBRA_HOLDOUT_FOLD]

    def cols(d, src):
        out = d[["image_path", "y"]].copy()
        out["source"] = src
        return out

    train = pd.concat([cols(br, "breast"), cols(bu_tr, "busi"), cols(bb_tr, "busbra")],
                      ignore_index=True)
    return {"train": train,
            "busi_holdout": cols(bu_te, "busi"),
            "busbra_holdout": cols(bb_te, "busbra"),
            "breast": br, "busi_full": bu, "busbra_full": bb,
            "busbra_train_df": bb_tr, "busbra_holdout_df": bb_te}


def main():
    d = build()
    tr = d["train"]
    print("=" * 62)
    print("EXPANDED TRAINING CORPUS")
    print("=" * 62)
    for src, n in tr["source"].value_counts().sort_index().items():
        sub = tr[tr["source"] == src]
        print(f"  {src:8} {n:5} images   malignant {int(sub['y'].sum()):4}")
    print(f"  {'TOTAL':8} {len(tr):5} images   malignant {int(tr['y'].sum()):4} "
          f"/ benign {int((tr['y'] == 0).sum())}")

    print("\nHELD-OUT EXTERNAL TEST SETS (never trained or calibrated on)")
    for k in ("busi_holdout", "busbra_holdout"):
        h = d[k]
        print(f"  {k:15} {len(h):5} images   malignant {int(h['y'].sum()):4}")

    prev = 256 + 546
    print(f"\nprevious training corpus: {prev} images")
    print(f"expanded  training corpus: {len(tr)} images  "
          f"({len(tr) / prev:.1f}x increase)")

    # ---- leakage assertions ------------------------------------------------- #
    print("\nLEAKAGE CHECKS")
    tr_paths = set(tr["image_path"])
    for k in ("busi_holdout", "busbra_holdout"):
        overlap = tr_paths & set(d[k]["image_path"])
        print(f"  train n {k}: {len(overlap)} overlapping images (must be 0)")
    bb_tr, bb_te = d["busbra_train_df"], d["busbra_holdout_df"]
    shared = set(bb_tr["patient_id"]) & set(bb_te["patient_id"])
    print(f"  BUS-BRA patients in both train and held-out: {len(shared)} (must be 0)")


if __name__ == "__main__":
    main()
