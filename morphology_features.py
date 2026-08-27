"""
morphology_features.py -- automated, image-derived quantitative descriptors for
a breast-ultrasound lesion, computed from its segmentation mask (+ the greyscale
image). These are designed to be an *automated* analogue of the radiologist's
BI-RADS descriptors, so the descriptor evidence stream no longer depends on
manual annotation.

Feature groups (all computable without a radiologist):
  shape        area fraction, equivalent diameter, extent, solidity, eccentricity
  orientation  |cos(orientation)|  -> "taller-than-wide" malignancy sign
  aspect       major/minor axis ratio (width-to-height)
  margin       circularity 4*pi*A/P^2, and convexity (area / convex area) as a
               proxy for an irregular / spiculated margin
  echo         mean & std intensity inside the lesion, and lesion-vs-surround
               contrast (hypoechoic lesions are darker than surrounding tissue)
  posterior    ratio of mean intensity in the region just below the lesion to the
               lesion itself -> posterior shadowing (<1) vs enhancement (>1)

get_features(image_path, mask_path) -> 1-D float32 vector (FEATURE_NAMES order).
"""
import numpy as np
from PIL import Image
from skimage.measure import regionprops, label

FEATURE_NAMES = [
    "area_frac", "equiv_diam", "extent", "solidity", "eccentricity",
    "vertical_orient", "aspect_ratio", "circularity", "convexity",
    "lesion_mean", "lesion_std", "lesion_surround_contrast", "posterior_ratio",
]


def _largest_region(mask):
    lab = label(mask.astype(int))
    if lab.max() == 0:
        return None
    props = regionprops(lab)
    return max(props, key=lambda p: p.area)


def get_features(image_path, mask_path):
    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    H, W = m.shape
    if img.shape != m.shape:
        img = np.asarray(Image.open(image_path).convert("L").resize((W, H)),
                         dtype=np.float32) / 255.0

    r = _largest_region(m)
    if r is None or r.area < 10:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    area_frac = r.area / (H * W)
    equiv_diam = r.equivalent_diameter_area / np.sqrt(H * W)
    extent = r.extent
    solidity = r.solidity
    ecc = r.eccentricity
    # orientation: regionprops angle is between the major axis and the ROW axis;
    # |cos| near 1 => major axis vertical => taller-than-wide (suspicious)
    vertical_orient = abs(np.cos(r.orientation))
    minor = max(r.axis_minor_length, 1e-3)
    aspect = r.axis_major_length / minor
    perim = max(r.perimeter, 1e-3)
    circularity = 4 * np.pi * r.area / (perim ** 2)
    convexity = r.area / max(r.area_convex, 1)

    ys, xs = np.where(m)
    lesion_px = img[ys, xs]
    lesion_mean = float(lesion_px.mean())
    lesion_std = float(lesion_px.std())
    # surrounding ring: dilate bbox and take non-lesion pixels around it
    y0, x0, y1, x1 = r.bbox
    py, px = int(0.5 * (y1 - y0)), int(0.5 * (x1 - x0))
    ry0, ry1 = max(0, y0 - py), min(H, y1 + py)
    rx0, rx1 = max(0, x0 - px), min(W, x1 + px)
    ring = img[ry0:ry1, rx0:rx1]
    ring_mask = m[ry0:ry1, rx0:rx1]
    surround = ring[~ring_mask]
    surround_mean = float(surround.mean()) if surround.size else lesion_mean
    contrast = lesion_mean - surround_mean          # negative => hypoechoic
    # posterior region: a band directly below the lesion bbox
    pb0, pb1 = y1, min(H, y1 + (y1 - y0))
    post = img[pb0:pb1, x0:x1]
    posterior_ratio = float(post.mean()) / (lesion_mean + 1e-6) if post.size else 1.0

    return np.array([
        area_frac, equiv_diam, extent, solidity, ecc, vertical_orient,
        aspect, circularity, convexity, lesion_mean, lesion_std,
        contrast, posterior_ratio], dtype=np.float32)


if __name__ == "__main__":
    f = get_features("dataset/images/case001.png", "dataset/images/case001_tumor.png")
    for n, v in zip(FEATURE_NAMES, f):
        print(f"  {n:24} {v: .4f}")
