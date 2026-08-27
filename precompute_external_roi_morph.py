"""Precompute ROI crops + morphology features for BUSI and BUS-BRA (both ship
masks), for external validation of the ROI + automated-morphology enhancements."""
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

from roi_crop import roi_crop
from morphology_features import get_features, FEATURE_NAMES


def busi_pairs():
    rows = []
    for cls, y in (("benign", 0), ("malignant", 1), ("normal", 0)):
        for p in sorted(glob.glob(f"busi/{cls}/*.png")):
            if "_mask" in p:
                continue
            masks = sorted(glob.glob(f"{p[:-4]}_mask*.png"))
            if masks:
                rows.append({"image_path": p, "mask_path": masks[0], "y": y, "src": "busi"})
    return pd.DataFrame(rows)


def busbra_pairs():
    meta = pd.read_csv("busbra/BUSBRA/bus_data.csv")
    rows = []
    for _, r in meta.iterrows():
        img = f"busbra/BUSBRA/Images/{r['ID']}.png"
        mask = f"busbra/BUSBRA/Masks/mask_{r['ID'][4:]}.png"  # bus_0001-l -> mask_0001-l
        if os.path.exists(img) and os.path.exists(mask):
            y = 1 if str(r["Pathology"]).strip().lower() == "malignant" else 0
            rows.append({"image_path": img, "mask_path": mask, "y": y, "src": "busbra"})
    return pd.DataFrame(rows)


def process(df, roi_dir, tag):
    os.makedirs(roi_dir, exist_ok=True)
    X = []
    roi_paths = []
    for i, r in enumerate(df.itertuples()):
        crop = roi_crop(r.image_path, r.mask_path)
        out = os.path.join(roi_dir, f"{tag}_{i:04d}.png")
        crop.save(out)
        roi_paths.append(out)
        X.append(get_features(r.image_path, r.mask_path))
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i+1}/{len(df)}", flush=True)
    df = df.copy()
    df["roi_path"] = roi_paths
    np.save(f"{tag}_morph.npy", np.vstack(X))
    df.to_csv(f"{tag}_pairs.csv", index=False)
    print(f"{tag}: {len(df)} images, morph {np.vstack(X).shape}")


if __name__ == "__main__":
    bu = busbra_pairs()
    process(bu, "busbra/roi", "busbra")
    b = busi_pairs()
    process(b, "busi/roi", "busi")
