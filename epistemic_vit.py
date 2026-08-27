"""
epistemic_vit.py -- MC-dropout epistemic uncertainty for the ViT visual stream,
the principled "stronger deferral signal" candidate: uncertainty the fused point
confidence cannot see (what the *image model* itself does not know).

Per fold: train a ViT (with dropout), then run T stochastic forward passes with
dropout kept ON at inference. Per test case we get the MC-mean P(malignant) and
the epistemic std across passes. Writes ./malig_epistemic.npz (p_vit_mc, u_epi,
y, fold) for selective_prediction.py to test as an abstention signal.

    python epistemic_vit.py            # ~ one extra per-fold ViT training pass
"""
import numpy as np
import torch
import torch.nn as nn
import timm

import config
from malignancy_data import load_dataset, make_folds, N_FOLDS
from malignancy_model import _DS, _free, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
from torch.utils.data import DataLoader

DROP_RATE = 0.2
MC_PASSES = 20


def build_dropout_model():
    return timm.create_model("vit_base_patch16_224", pretrained=True,
                             num_classes=1, drop_rate=DROP_RATE)


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_proba(model, loader, device, passes=MC_PASSES):
    model.eval()
    _enable_dropout(model)                      # keep dropout stochastic at inference
    all_p = []
    for _ in range(passes):
        ps = []
        for x, _y in loader:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy().ravel())
        all_p.append(np.concatenate(ps))
    P = np.stack(all_p)                         # (passes, n)
    return P.mean(0), P.std(0)


def train_fold(train_df, test_df, device, log_prefix=""):
    tr = DataLoader(_DS(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te = DataLoader(_DS(test_df, False), batch_size=BATCH_SIZE, num_workers=0)
    n_pos = float(train_df["y"].sum()); n_neg = float((train_df["y"] == 0).sum())
    pos_w = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    model = build_dropout_model().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(EPOCHS):
        model.train(); tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)
    mean, std = mc_proba(model, te, device)
    _free(model, opt, sched, crit)
    return mean, std


def main():
    df = load_dataset(); y = df["y"].values; fold = make_folds(df); n = len(df)
    p_mc, u_epi = np.zeros(n), np.zeros(n)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold != k)[0]; te_idx = np.where(fold == k)[0]
        m, s = train_fold(df.iloc[tr_idx], df.iloc[te_idx], config.DEVICE,
                          log_prefix=f"[mc-vit][fold {k}] ")
        p_mc[te_idx] = m; u_epi[te_idx] = s
        print(f"  fold {k} done: mean epistemic std {s.mean():.3f}", flush=True)
    np.savez("./malig_epistemic.npz", p_vit_mc=p_mc, u_epi=u_epi, y=y, fold=fold)
    from sklearn.metrics import roc_auc_score
    print(f"\nMC-ViT AUC {roc_auc_score(y, p_mc):.3f} | "
          f"epistemic std range {u_epi.min():.3f}-{u_epi.max():.3f}")
    print("wrote ./malig_epistemic.npz")


if __name__ == "__main__":
    main()
