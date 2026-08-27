"""Two-panel confusion matrix for the fused model: balanced (0.5) and
sensitivity-first (0.32) operating points. Overwrites fig_confusion_matrix.png."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from evaluate_malignancy import fuse_stack

OUT = "../dissertation/figures/fig_confusion_matrix.png"
z = np.load("malig_streams.npz", allow_pickle=True)
y = z["y"].astype(int)
streams = {k: z[k] for k in ("descriptor", "knn", "vit")}
p = fuse_stack(streams, y, z["fold"])

panels = [(0.5, "(a) Balanced — threshold 0.5"),
          (0.32, "(b) Sensitivity-first — threshold 0.32")]
labels = ["Benign", "Malignant"]
names = [["TN", "FP"], ["FN", "TP"]]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))
for ax, (t, title) in zip(axes, panels):
    yhat = (p >= t).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=10.5)
    ax.set_ylabel("True label", fontsize=10.5)
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > thr else "#1F2937",
                    fontweight="bold")
    ax.set_title(f"{title}\nsensitivity {sens:.2f} · specificity {spec:.2f}", fontsize=10.5)
fig.suptitle("Confusion matrices — fused model (leak-free 5-fold CV, BrEaST n = 256)",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
