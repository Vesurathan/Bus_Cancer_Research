"""Train the malignancy ViT on a real stratified 80/20 train/validation split,
logging per-epoch training/validation loss and accuracy, then plot the learning
curves. These are genuine curves from an actual run (not recorded originally).
Run from bus/. Outputs curves + a JSON log to ../dissertation/figures/."""
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
from malignancy_data import load_dataset
from malignancy_model import build_model, _DS, _free, BATCH_SIZE, LR, WEIGHT_DECAY

OUT = "../dissertation/figures"
EPOCHS = 15
NAVY, TEAL = "#1F3A5F", "#2C7A7B"
device = config.DEVICE


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    loss_tot, correct, n = 0.0, 0, 0
    for x, yb in loader:
        x, yb = x.to(device), yb.to(device)
        logit = model(x)
        loss_tot += crit(logit, yb).item() * len(x)
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == yb).sum().item()
        n += len(x)
    return loss_tot / n, correct / n


def main():
    df = load_dataset()
    tr_df, va_df = train_test_split(
        df, test_size=0.20, stratify=df["y"].values, random_state=config.SEED)
    print(f"train {len(tr_df)} (mal {int(tr_df['y'].sum())}) | "
          f"val {len(va_df)} (mal {int(va_df['y'].sum())})", flush=True)

    n_pos, n_neg = float(tr_df["y"].sum()), float((tr_df["y"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    tr = DataLoader(_DS(tr_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    tr_eval = DataLoader(_DS(tr_df, False), batch_size=BATCH_SIZE, num_workers=0)
    va = DataLoader(_DS(va_df, False), batch_size=BATCH_SIZE, num_workers=0)

    model = build_model().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(x), yb)
            loss.backward(); opt.step()
        sched.step()
        tl, ta = evaluate(model, tr_eval, crit, device)
        vl, va_acc = evaluate(model, va, crit, device)
        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta); hist["val_acc"].append(va_acc)
        print(f"epoch {epoch+1:2d}  train_loss {tl:.4f} val_loss {vl:.4f}  "
              f"train_acc {ta:.3f} val_acc {va_acc:.3f}", flush=True)

    json.dump(hist, open(f"{OUT}/training_log.json", "w"), indent=2)
    _free(opt, sched, crit)

    ep = hist["epoch"]
    # ---- Accuracy curve --------------------------------------------------- #
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(ep, hist["train_acc"], color=NAVY, lw=2.2, marker="o", ms=4, label="Training accuracy")
    plt.plot(ep, hist["val_acc"], color=TEAL, lw=2.2, marker="s", ms=4, label="Validation accuracy")
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("Accuracy", fontsize=11)
    plt.title("Training and validation accuracy (image ViT stream)", fontsize=11.5)
    plt.legend(loc="lower right", fontsize=10); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_train_val_accuracy.png", dpi=200, facecolor="white")
    plt.close()

    # ---- Loss curve ------------------------------------------------------- #
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(ep, hist["train_loss"], color=NAVY, lw=2.2, marker="o", ms=4, label="Training loss")
    plt.plot(ep, hist["val_loss"], color=TEAL, lw=2.2, marker="s", ms=4, label="Validation loss")
    plt.xlabel("Epoch", fontsize=11); plt.ylabel("BCE loss", fontsize=11)
    plt.title("Training and validation loss (image ViT stream)", fontsize=11.5)
    plt.legend(loc="upper right", fontsize=10); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_train_val_loss.png", dpi=200, facecolor="white")
    plt.close()
    print("wrote fig_train_val_accuracy.png and fig_train_val_loss.png")


if __name__ == "__main__":
    main()
