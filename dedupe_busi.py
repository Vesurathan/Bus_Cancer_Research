"""
dedupe_busi.py -- detect and remove duplicate / near-duplicate images in BUSI.

BUSI is documented to contain duplicated images (sometimes with inconsistent
labels across copies), which leaks between train and test splits and inflates
reported performance. This script finds them two ways:

  1. exact duplicates      -- md5 of the raw file bytes
  2. near-duplicates       -- 16x16 dHash of the grayscale image (Hamming <= 4)

Cross-class duplicate groups (the same lesion appearing under two different
class folders) are reported separately: those are label inconsistencies, not
just redundancy.

Writes busi_clean.csv -- one row per KEPT image.

    python dedupe_busi.py
"""
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from busi_data import load_busi

OUT_CSV = "busi_clean.csv"
HAMMING_MAX = 4          # near-duplicate threshold on a 256-bit dHash


def md5_bytes(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def dhash(path, size=16):
    """Difference hash: compare adjacent pixels of a resized grayscale image."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def main():
    df = load_busi()
    print(f"BUSI raw images: {len(df)}")
    print("by class:", df["cls"].value_counts().to_dict(), "\n")

    df["md5"] = [md5_bytes(p) for p in df["image_path"]]
    hashes = [dhash(p) for p in df["image_path"]]

    # ---- 1. exact duplicates ------------------------------------------------ #
    by_md5 = defaultdict(list)
    for i, m in enumerate(df["md5"]):
        by_md5[m].append(i)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    n_exact_extra = sum(len(g) - 1 for g in exact_groups)
    print(f"exact-duplicate groups: {len(exact_groups)}  (redundant copies: {n_exact_extra})")

    # ---- 2. near-duplicates (union-find over hamming distance) -------------- #
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if hamming(hashes[i], hashes[j]) <= HAMMING_MAX:
                union(i, j)

    groups = defaultdict(list)
    for i in range(len(df)):
        groups[find(i)].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    n_extra = sum(len(g) - 1 for g in dup_groups)
    print(f"near-duplicate groups (dHash<={HAMMING_MAX}): {len(dup_groups)}  "
          f"(redundant copies: {n_extra})")

    # ---- 3. cross-class groups = label inconsistencies ---------------------- #
    cross = [g for g in dup_groups if df["cls"].iloc[g].nunique() > 1]
    print(f"cross-class duplicate groups (LABEL CONFLICTS): {len(cross)}")
    for g in cross[:10]:
        for i in g:
            print(f"    [{df['cls'].iloc[i]:9}] {df['image_path'].iloc[i].split('/')[-1]}")
        print("    ---")

    # ---- 4. keep one representative per group ------------------------------- #
    # drop whole groups that conflict on label (unusable), keep first of the rest
    conflict_ids = {i for g in cross for i in g}
    keep = []
    for g in groups.values():
        g_sorted = sorted(g)
        if g_sorted[0] in conflict_ids:
            continue                      # ambiguous label -> exclude entirely
        keep.append(g_sorted[0])
    keep = sorted(keep)

    clean = df.iloc[keep].drop(columns=["md5"]).reset_index(drop=True)
    clean.to_csv(OUT_CSV, index=False)

    print(f"\nremoved {len(df) - len(clean)} images "
          f"({n_extra} redundant + {len(conflict_ids)} label-conflicting)")
    print(f"BUSI clean: {len(clean)} images -> {OUT_CSV}")
    print("by class:", clean["cls"].value_counts().to_dict())
    print(f"malignant {int(clean['y'].sum())} / non-malignant {int((clean['y']==0).sum())}")


if __name__ == "__main__":
    main()
