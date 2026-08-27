"""
busi_data.py -- loader for the external BUSI dataset (Al-Dhabyani et al., "Dataset
of breast ultrasound images", 2020) used for cross-dataset validation.

Expected layout (place the unzipped dataset here):
    ./busi/
        benign/     benign (1).png, benign (1)_mask.png, ...
        malignant/  malignant (1).png, ...
        normal/     normal (1).png, ...

We use it image-only (BUSI has no BI-RADS descriptor columns), so cross-dataset
validation exercises the IMAGE streams (ViT + kNN) and the shift-detection /
deferral mechanism -- exactly the setting where "know when to abstain" matters.

    python busi_data.py     # sanity: counts + label balance
"""
import os, glob
import pandas as pd

BUSI_DIR = "./busi"
CLASS_DIRS = {"benign": 0, "malignant": 1, "normal": 0}   # malignant vs non-malignant


def load_busi(root=BUSI_DIR):
    """Return DataFrame: image_path, y, cls. Masks (*_mask*.png) are skipped.
    Raises a clear error if the dataset folder is missing."""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"BUSI not found at {root}. Download 'Dataset_BUSI_with_GT' and unzip "
            f"so that {root}/benign, {root}/malignant, {root}/normal exist.")
    rows = []
    # tolerate case / naming variants of the class subfolders
    subdirs = {d.lower(): d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))}
    for cls, y in CLASS_DIRS.items():
        d = subdirs.get(cls)
        if not d:
            continue
        for p in glob.glob(os.path.join(root, d, "*.png")) + \
                 glob.glob(os.path.join(root, d, "*.jpg")):
            if "_mask" in os.path.basename(p).lower():
                continue
            rows.append({"image_path": p, "y": y, "cls": cls})
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No BUSI images found under {root} (check the layout).")
    return df


if __name__ == "__main__":
    df = load_busi()
    print(f"BUSI images: {len(df)}")
    print("by class:", df["cls"].value_counts().to_dict())
    print(f"malignant {int(df['y'].sum())} / non-malignant {int((df['y']==0).sum())}")
