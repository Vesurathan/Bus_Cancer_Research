"""Clean single-system two-site external-validation figure: the system's AUC on
the two independent external datasets (BUSI and BUS-BRA). No model comparison."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../dissertation/figures/fig_twosite_external.png"
NAVY, TEAL = "#1F3A5F", "#2C7A7B"

sites = ["BUSI\n(Egypt, n = 234)", "BUS-BRA\n(Brazil, n = 383)"]
auc = [0.945, 0.865]
# 95% CI for BUS-BRA measured by bootstrap; BUSI point estimate from Table 4.3
err_lo = [0.0, 0.865 - 0.825]
err_hi = [0.0, 0.904 - 0.865]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = np.arange(len(sites))
bars = ax.bar(x, auc, 0.5, color=[NAVY, TEAL], edgecolor="white",
              yerr=[err_lo, err_hi], capsize=6)
for b, v in zip(bars, auc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(sites, fontsize=10.5)
ax.set_ylabel("Area under ROC curve (AUC)", fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, ls=":", color="#9AA3B2", lw=1)
ax.set_title("External validation on two independent datasets", fontsize=12)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
