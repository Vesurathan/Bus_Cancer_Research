"""Evaluation figures from the real cached predictions (malig_streams.npz):
confusion matrix, precision-recall curves, and F1-vs-threshold.
Run from bus/. Outputs to ../dissertation/figures/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix, f1_score)
from evaluate_malignancy import fuse_stack

OUT = "../dissertation/figures"
os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, ORANGE, PURPLE, RED = "#1F3A5F", "#2C7A7B", "#C77D3B", "#7C6BAF", "#B23A48"

z = np.load("malig_streams.npz", allow_pickle=True)
y = z["y"].astype(int)
streams = {k: z[k] for k in ("descriptor", "knn", "vit")}
p_fused = fuse_stack(streams, y, z["fold"])


# ---- Confusion matrix (fusion @ 0.5) ------------------------------------- #
def fig_confusion():
    yhat = (p_fused >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Benign", "Malignant"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    thr = cm.max() / 2
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > thr else "#1F2937",
                    fontweight="bold")
    ax.set_title("Confusion matrix — fused model (threshold 0.5)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_confusion_matrix.png", dpi=200, facecolor="white"); plt.close()


# ---- Precision-recall curves --------------------------------------------- #
def fig_pr():
    plt.figure(figsize=(6.2, 5.2))
    curves = [("Stacked fusion", p_fused, TEAL, 2.6),
              ("Descriptor", streams["descriptor"], NAVY, 1.8),
              ("ViT (image)", streams["vit"], ORANGE, 1.8),
              ("k-NN (image)", streams["knn"], PURPLE, 1.8)]
    for name, p, c, lw in curves:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rec, prec, color=c, lw=lw, label=f"{name} (AP = {ap:.3f})")
    plt.axhline(y.mean(), ls="--", color="#9AA3B2", lw=1, label=f"Prevalence ({y.mean():.2f})")
    plt.xlabel("Recall (sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision–recall curves (internal 5-fold CV)", fontsize=11.5)
    plt.legend(loc="lower left", fontsize=9.5); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_pr_curve.png", dpi=200, facecolor="white"); plt.close()


# ---- F1 / precision / recall vs threshold (fusion) ----------------------- #
def fig_f1_threshold():
    ts = np.linspace(0.02, 0.98, 97)
    f1s, precs, recs = [], [], []
    for t in ts:
        yh = (p_fused >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        precs.append(tp / (tp + fp) if tp + fp else 0)
        recs.append(tp / (tp + fn) if tp + fn else 0)
        f1s.append(f1_score(y, yh, zero_division=0))
    best_t = ts[int(np.argmax(f1s))]
    plt.figure(figsize=(6.2, 5.0))
    plt.plot(ts, f1s, color=TEAL, lw=2.4, label="F1")
    plt.plot(ts, precs, color=NAVY, lw=1.6, ls="-", label="Precision")
    plt.plot(ts, recs, color=RED, lw=1.6, ls="-", label="Recall (sensitivity)")
    plt.axvline(best_t, ls=":", color="#9AA3B2", lw=1.2,
                label=f"Best-F1 threshold = {best_t:.2f}")
    plt.xlabel("Decision threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("F1, precision and recall versus threshold (fused model)", fontsize=11.5)
    plt.legend(loc="lower center", fontsize=9.5); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_f1_threshold.png", dpi=200, facecolor="white"); plt.close()


if __name__ == "__main__":
    fig_confusion(); print("fig_confusion_matrix")
    fig_pr(); print("fig_pr_curve")
    fig_f1_threshold(); print("fig_f1_threshold")
