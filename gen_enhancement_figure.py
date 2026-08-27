"""Figure: baseline vs enhanced (ROI + automated morphology) autonomous pipeline
on the two external datasets, mean +/- std over 3 seeds."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../dissertation/figures/fig_enhancement.png"
GREY, TEAL = "#9AA3B2", "#2C7A7B"

sites = ["BUSI\n(Egypt, n = 234)", "BUS-BRA\n(Brazil, n = 1875)"]
base_m, base_s = [0.983, 0.809], [0.001, 0.017]
enh_m, enh_s = [0.980, 0.884], [0.003, 0.002]

x = np.arange(len(sites)); w = 0.36
fig, ax = plt.subplots(figsize=(6.8, 4.6))
b1 = ax.bar(x - w / 2, base_m, w, yerr=base_s, capsize=5, color=GREY,
            edgecolor="white", label="Baseline (ViT + k-NN)")
b2 = ax.bar(x + w / 2, enh_m, w, yerr=enh_s, capsize=5, color=TEAL,
            edgecolor="white", label="Enhanced (ROI-ViT + k-NN + morphology)")
for bars, m in ((b1, base_m), (b2, enh_m)):
    for b, v in zip(bars, m):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sites)
ax.set_ylabel("Area under ROC curve (AUC)", fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_title("Autonomous image-only pipeline: effect of ROI focus and\nautomated "
             "morphology (mean ± std, 3 seeds)", fontsize=11.5)
ax.legend(loc="lower left", fontsize=9.5)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
