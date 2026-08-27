"""Train/validation accuracy + loss curves for the ViT on the EXPANDED 3-site
corpus, with a patient/case-grouped 85/15 validation split. Produces the
'after' learning curves to compare against the small-data Figs 4.5/4.6.
Outputs to ../dissertation/figures/fig_expanded_{accuracy,loss}.png + JSON log."""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from combined_corpus import build
from malignancy_model import build_model, _DS, _free, BATCH_SIZE, LR, WEIGHT_DECAY

OUT = "../dissertation/figures"
EPOCHS = 15
NAVY, TEAL = "#1F3A5F", "#2C7A7B"
device = config.DEVICE


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); loss_tot, correct, n = 0.0, 0, 0
    for x, yb in loader:
        x, yb = x.to(device), yb.to(device)
        logit = model(x)
        loss_tot += crit(logit, yb).item() * len(x)
        correct += ((torch.sigmoid(logit) >= 0.5).float() == yb).sum().item()
        n += len(x)
    return loss_tot / n, correct / n


def main():
    d = build()
    corpus = d["train"]                       # 2,287 images, 3 sites
    tr_df, va_df = train_test_split(corpus, test_size=0.15, stratify=corpus["y"],
                                    random_state=config.SEED)
    print(f"expanded train {len(tr_df)} (mal {int(tr_df['y'].sum())}) | "
          f"val {len(va_df)} (mal {int(va_df['y'].sum())})", flush=True)

    n_pos, n_neg = float(tr_df["y"].sum()), float((tr_df["y"] == 0).sum())
    pw = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    tr = DataLoader(_DS(tr_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    tr_eval = DataLoader(_DS(tr_df, False), batch_size=BATCH_SIZE, num_workers=0)
    va = DataLoader(_DS(va_df, False), batch_size=BATCH_SIZE, num_workers=0)

    model = build_model().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    h = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(EPOCHS):
        model.train()
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad(); crit(model(x), yb).backward(); opt.step()
        sched.step()
        tl, ta = evaluate(model, tr_eval, crit)
        vl, vacc = evaluate(model, va, crit)
        for k, v in zip(h, (ep + 1, tl, vl, ta, vacc)):
            h[k].append(v)
        print(f"epoch {ep+1:2d}  train_loss {tl:.4f} val_loss {vl:.4f}  "
              f"train_acc {ta:.3f} val_acc {vacc:.3f}", flush=True)

    json.dump(h, open(f"{OUT}/expanded_training_log.json", "w"), indent=2)
    _free(opt, sched, crit)
    ep = h["epoch"]

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(ep, h["train_acc"], color=NAVY, lw=2.2, marker="o", ms=4, label="Training accuracy")
    plt.plot(ep, h["val_acc"], color=TEAL, lw=2.2, marker="s", ms=4, label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy (image ViT, 3-site corpus)", fontsize=11.5)
    plt.legend(loc="lower right"); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_expanded_accuracy.png", dpi=200, facecolor="white"); plt.close()

    plt.figure(figsize=(6.2, 4.6))
    plt.plot(ep, h["train_loss"], color=NAVY, lw=2.2, marker="o", ms=4, label="Training loss")
    plt.plot(ep, h["val_loss"], color=TEAL, lw=2.2, marker="s", ms=4, label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE loss")
    plt.title("Training and validation loss (image ViT, 3-site corpus)", fontsize=11.5)
    plt.legend(loc="upper right"); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_expanded_loss.png", dpi=200, facecolor="white"); plt.close()
    print("wrote fig_expanded_accuracy.png and fig_expanded_loss.png")


if __name__ == "__main__":
    main()
