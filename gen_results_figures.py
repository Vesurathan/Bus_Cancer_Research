"""Generate Chapter 4 results figures from the real cached predictions/models.
Run from the bus/ directory. Outputs to ../dissertation/figures/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from evaluate_malignancy import fuse_stack
from selective_prediction import risk_coverage, image_prob

OUT = "../dissertation/figures"
os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, ORANGE, GREY, RED = "#1F3A5F", "#2C7A7B", "#C77D3B", "#9AA3B2", "#B23A48"

z = np.load("malig_streams.npz", allow_pickle=True)
y, fold = z["y"].astype(int), z["fold"]
streams = {k: z[k] for k in ("descriptor", "knn", "vit")}
p_fused = fuse_stack(streams, y, fold)


# ---- Figure 4.1: ROC curves (internal 5-fold CV) ------------------------- #
def fig_roc():
    plt.figure(figsize=(6.2, 5.4))
    curves = [
        ("Stacked fusion", p_fused, TEAL, 2.6),
        ("Descriptor", streams["descriptor"], NAVY, 1.8),
        ("ViT (image)", streams["vit"], ORANGE, 1.8),
        ("k-NN (image)", streams["knn"], "#7C6BAF", 1.8),
    ]
    for name, p, c, lw in curves:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, color=c, lw=lw, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    plt.xlabel("1 − Specificity (false-positive rate)", fontsize=11)
    plt.ylabel("Sensitivity (true-positive rate)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_1_roc.png", dpi=200, facecolor="white"); plt.close()


# ---- Figure 4.2: risk-coverage (selective prediction) -------------------- #
def fig_riskcoverage():
    yhat = (p_fused >= 0.5).astype(int)
    correct = (yhat == y).astype(int)
    conf = np.abs(p_fused - 0.5)
    p_img = image_prob(streams)
    disagree = np.abs(p_img - streams["descriptor"])
    rng = np.random.default_rng(0)
    sigs = [
        ("Oracle (upper bound)", correct.astype(float), "#2E7D32", "--", 1.6),
        ("Confidence (best)", conf, TEAL, "-", 2.4),
        ("Cross-modal disagreement", -disagree, ORANGE, "-", 1.8),
        ("Random", rng.random(len(y)), GREY, ":", 1.6),
    ]
    plt.figure(figsize=(6.2, 5.0))
    for name, rel, c, ls, lw in sigs:
        cov, err = risk_coverage(correct, rel)
        plt.plot(cov * 100, (1 - err) * 100, color=c, ls=ls, lw=lw, label=name)
    plt.xlabel("Coverage — cases auto-decided (%)", fontsize=11)
    plt.ylabel("Selective accuracy (%)", fontsize=11)
    plt.legend(loc="lower left", fontsize=9.5, frameon=True)
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_2_riskcoverage.png", dpi=200, facecolor="white"); plt.close()


# ---- Figure 4.3: Grad-CAM examples --------------------------------------- #
def fig_gradcam():
    import glob
    from explain import GradCAMViT, save_overlay
    from PIL import Image
    g = GradCAMViT("./prod/vit_malignancy.pt")
    picks = []
    for cls in ("malignant", "benign"):
        imgs = [p for p in sorted(glob.glob(f"busi/{cls}/*.png")) if "_mask" not in p]
        for p in imgs[:2]:
            picks.append((cls, p))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 3.2))
    for ax, (cls, path) in zip(axes, picks):
        prob, cam = g.heatmap(path)
        tmp = "/tmp/_cam.png"; save_overlay(path, cam, tmp)
        ax.imshow(Image.open(tmp)); ax.axis("off")
        ax.set_title(f"true: {cls}\nP(malignant) = {prob:.2f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_3_gradcam.png", dpi=200, facecolor="white"); plt.close()


if __name__ == "__main__":
    fig_roc(); print("fig4_1_roc")
    fig_riskcoverage(); print("fig4_2_riskcoverage")
    fig_gradcam(); print("fig4_3_gradcam")
