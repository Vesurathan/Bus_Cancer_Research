"""
roi_crop.py -- crop a BUS image to a padded, square lesion ROI from its mask, so
the image classifier focuses on the lesion instead of speckle/background. Falls
back to the full frame when no lesion is present. Precomputes cropped BrEaST
images into dataset/roi/ for the ROI-ViT experiment.
"""
import os

import numpy as np
from PIL import Image

MARGIN = 0.30            # pad the lesion bbox by 30% of its size on each side


def roi_crop(image_path, mask_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    if not os.path.exists(mask_path):
        return img
    m = np.asarray(Image.open(mask_path).convert("L").resize((W, H))) > 127
    ys, xs = np.where(m)
    if ys.size == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = y1 - y0, x1 - x0
    side = int(max(bh, bw) * (1 + 2 * MARGIN))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    l, r = max(0, cx - half), min(W, cx + half)
    t, b = max(0, cy - half), min(H, cy + half)
    return img.crop((l, t, r, b))


def precompute_breast(out_dir="dataset/roi"):
    import glob
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    f = glob.glob("dataset/*clinical*")[0]
    df = pd.read_excel(f)
    n = 0
    for _, row in df.iterrows():
        img = os.path.join("dataset/images", str(row["Image_filename"]))
        mask = os.path.join("dataset/images", str(row["Mask_tumor_filename"]))
        if not os.path.exists(img):
            continue
        crop = roi_crop(img, mask)
        crop.save(os.path.join(out_dir, os.path.basename(img)))
        n += 1
    print(f"wrote {n} ROI crops to {out_dir}")


if __name__ == "__main__":
    precompute_breast()
