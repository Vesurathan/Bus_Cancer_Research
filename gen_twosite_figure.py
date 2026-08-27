"""Two-site external-validation figure: original vs expanded model on the SAME
clean held-out sets (BUSI 232 + BUS-BRA 383), AUC with bootstrap 95% CI.
Outputs fig_twosite_external.png and prints the table."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import config
from combined_corpus import build
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote
from promote_expanded import boot_ci

OUT = "../dissertation/figures/fig_twosite_external.png"
NAVY, TEAL, GREY = "#1F3A5F", "#2C7A7B", "#9AA3B2"
dev = config.DEVICE
d = build()


def auc_ci(model_dir, df):
    vit = Predictor(f"{model_dir}/vit_malignancy.pt", dev)
    bank = np.load(f"{model_dir}/knn_bank.npz")
    p_v = np.array([vit.predict_proba(p) for p in df["image_path"]])
    p_k = knn_vote(bank["E"], bank["y"], embed_paths(df["image_path"].tolist(), dev))
    p = _mean_logit(p_v, p_k)
    y = df["y"].values
    lo, hi = boot_ci(y, p, roc_auc_score)
    return roc_auc_score(y, p), lo, hi


sites = [("BUSI\n(site 1, Egypt)", d["busi_holdout"]),
         ("BUS-BRA\n(site 2, Brazil)", d["busbra_holdout"])]
models = [("Original (2 sites, 802 img)", "./prod", GREY),
          ("Expanded (3 sites, 1,922 img)", "./prod_expanded", TEAL)]

vals = {}
print(f"{'model':32}{'site':12}{'AUC':>8}  95% CI")
for mname, mdir, _ in models:
    for sname, df in sites:
        a, lo, hi = auc_ci(mdir, df)
        vals[(mname, sname)] = (a, lo, hi)
        print(f"{mname:32}{sname.splitlines()[0]:12}{a:8.3f}  [{lo:.3f}, {hi:.3f}]")

fig, ax = plt.subplots(figsize=(7.4, 4.8))
x = np.arange(len(sites)); w = 0.36
for j, (mname, mdir, c) in enumerate(models):
    a = [vals[(mname, s[0])][0] for s in sites]
    lo = [vals[(mname, s[0])][0] - vals[(mname, s[0])][1] for s in sites]
    hi = [vals[(mname, s[0])][2] - vals[(mname, s[0])][0] for s in sites]
    bars = ax.bar(x + (j - 0.5) * w, a, w, yerr=[lo, hi], capsize=5,
                  color=c, label=mname, edgecolor="white")
    for b, v in zip(bars, a):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sites])
ax.set_ylabel("Area under ROC curve (AUC)", fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_title("External validation across two independent sites", fontsize=12)
ax.axhline(0.5, ls=":", color=GREY, lw=1)
ax.legend(loc="lower left", fontsize=9.5, frameon=True)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("\nwrote", OUT)
