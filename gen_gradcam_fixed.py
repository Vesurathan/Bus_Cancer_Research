"""Regenerate the Grad-CAM overlay figure with enough top margin so the
two-line titles are not clipped."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from explain import GradCAMViT, save_overlay

OUT = "../dissertation/figures/fig4_3_gradcam.png"
g = GradCAMViT("./prod/vit_malignancy.pt")

picks = []
for cls in ("malignant", "benign"):
    imgs = [p for p in sorted(glob.glob(f"busi/{cls}/*.png")) if "_mask" not in p]
    for p in imgs[:2]:
        picks.append((cls, p))

fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.5),
                         constrained_layout=True)
for ax, (cls, path) in zip(axes, picks):
    prob, cam = g.heatmap(path)
    tmp = "/tmp/_cam.png"; save_overlay(path, cam, tmp)
    ax.imshow(Image.open(tmp)); ax.axis("off")
    ax.set_title(f"true: {cls}\nP(malignant) = {prob:.2f}", fontsize=10, pad=8)
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print("wrote", OUT)
