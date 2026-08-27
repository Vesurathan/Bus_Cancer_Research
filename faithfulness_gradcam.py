"""
faithfulness_gradcam.py -- quantitative faithfulness of the malignancy
classifier's Grad-CAM explanations against expert lesion masks. Following the
faithfulness-evaluation protocol of Shivakumar, Mahmood & Khatoon (2026), we
report the Intersection-over-Union (IoU), Dice coefficient and Pointing-Game
accuracy between the Grad-CAM heatmap and the ground-truth segmentation mask.

Run from bus/. Uses the production ViT and the BUSI masks.
"""
import glob

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from explain import GradCAMViT

CKPT = "./prod/vit_malignancy.pt"
SIGMA = 8               # Gaussian smoothing (matches the overlay used in Fig 4.3)


def _otsu(x):
    """Otsu threshold on a [0,1] map (256-bin histogram)."""
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0, 1))
    hist = hist.astype(float)
    w = np.cumsum(hist)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(hist * mids)
    tot_w, tot_mu = w[-1], mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        wb = w
        wf = tot_w - w
        mb = mu / np.where(wb == 0, 1, wb)
        mf = (tot_mu - mu) / np.where(wf == 0, 1, wf)
        between = wb * wf * (mb - mf) ** 2
    return mids[int(np.nanargmax(between))]


def masks_for(image_path):
    """All mask files that belong to one BUSI image (union of multi-masks)."""
    stem = image_path[:-4]                       # strip .png
    return sorted(glob.glob(f"{stem}_mask*.png"))


def load_mask(paths, size=224):
    m = np.zeros((size, size), dtype=bool)
    for p in paths:
        img = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
        m |= (np.asarray(img) > 127)
    return m


def metrics(cam, mask):
    # smooth into a blob (as visualised) then Otsu-binarise for region overlap
    sm = gaussian_filter(cam, sigma=SIGMA)
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
    cam_bin = sm >= _otsu(sm)
    inter = np.logical_and(cam_bin, mask).sum()
    union = np.logical_or(cam_bin, mask).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (cam_bin.sum() + mask.sum()) if (cam_bin.sum() + mask.sum()) else 0.0
    # pointing game: does the smoothed CAM's peak fall inside the lesion?
    peak = np.unravel_index(np.argmax(sm), sm.shape)
    hit = bool(mask[peak])
    # energy-based localisation: fraction of raw attribution mass inside the mask
    energy = cam[mask].sum() / (cam.sum() + 1e-8)
    return iou, dice, hit, energy


def main():
    g = GradCAMViT(CKPT)
    rows = {"malignant": [], "benign": []}
    for cls in ("malignant", "benign"):
        imgs = [p for p in sorted(glob.glob(f"busi/{cls}/*.png")) if "_mask" not in p]
        for p in imgs:
            mpaths = masks_for(p)
            if not mpaths:
                continue
            mask = load_mask(mpaths)
            if mask.sum() == 0:
                continue
            _, cam = g.heatmap(p)
            rows[cls].append(metrics(cam, mask))
        arr = np.array(rows[cls]) if rows[cls] else np.zeros((0, 4))
        if len(arr):
            print(f"{cls:10} n={len(arr):3}  IoU {arr[:,0].mean():.3f}  "
                  f"Dice {arr[:,1].mean():.3f}  Pointing-Game {arr[:,2].mean():.3f}  "
                  f"Energy {arr[:,3].mean():.3f}", flush=True)

    allrows = np.array(rows["malignant"] + rows["benign"])
    print("-" * 72)
    print(f"{'OVERALL':10} n={len(allrows):3}  IoU {allrows[:,0].mean():.3f}  "
          f"Dice {allrows[:,1].mean():.3f}  Pointing-Game {allrows[:,2].mean():.3f}  "
          f"Energy {allrows[:,3].mean():.3f}")
    np.save("gradcam_faithfulness.npy", allrows)


if __name__ == "__main__":
    main()
