"""
ablation_findings.py -- does the long-tail loss actually help the findings module?
Trains the same ViT on the same split with three losses and compares macro/micro-F1
and per-frequency-tier F1:
  1. plain BCE                (vanilla multi-label)
  2. class-weighted BCE       (inverse-frequency reweighting only)
  3. LDAM-DRW                 (margin + curriculum + deferred reweighting -- proposed)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from vocab import TERMS
from finding_model import build_model
from train_finding_agent import (labels_from_report, BUSReportDS, LDAMDRWLoss,
                                  EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
                                  MARGIN_WARMUP_FRAC, DRW_START_FRAC)

dev = config.DEVICE


def plain_bce(n_j):
    return lambda logits, y, ep: F.binary_cross_entropy_with_logits(logits, y)


def weighted_bce(n_j):
    n = n_j.sum() if hasattr(n_j, "sum") else sum(n_j)
    pw = torch.tensor([(len(TERMS) and (max(1.0, (n - nj)) / max(1.0, nj)))
                       for nj in n_j], dtype=torch.float32, device=dev).clamp(max=50)
    return lambda logits, y, ep: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def ldam(n_j):
    warm = max(1, round(EPOCHS * MARGIN_WARMUP_FRAC)); drw = max(1, round(EPOCHS * DRW_START_FRAC))
    return LDAMDRWLoss(n_j, dev, warm, drw)


@torch.no_grad()
def per_term_f1(model, loader):
    model.eval()
    tp = np.zeros(len(TERMS)); fp = np.zeros(len(TERMS)); fn = np.zeros(len(TERMS))
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).cpu().numpy()
        y = y.numpy()
        tp += ((p == 1) & (y == 1)).sum(0)
        fp += ((p == 1) & (y == 0)).sum(0)
        fn += ((p == 0) & (y == 1)).sum(0)
    return tp, fp, fn


def f1_from(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return f1


def run(name, loss_fn, tr, te, n_j, tiers):
    tr_ld = DataLoader(BUSReportDS(tr, True), batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(BUSReportDS(te, False), batch_size=BATCH_SIZE)
    torch.manual_seed(config.SEED)
    model = build_model().to(dev)
    crit = loss_fn(n_j)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); crit(model(x), y, ep).backward(); opt.step()
        sched.step()
    tp, fp, fn = per_term_f1(model, te_ld)
    f1 = f1_from(tp, fp, fn)
    macro = f1.mean()
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro = f1_from(np.array([micro_tp]), np.array([micro_fp]), np.array([micro_fn]))[0]
    tier_f1 = {t: (f1[idx].mean() if len(idx) else float("nan")) for t, idx in tiers.items()}
    print(f"{name:22} macro {macro:.3f}  micro {micro:.3f}  "
          f"head {tier_f1['head']:.3f}  medium {tier_f1['medium']:.3f}  tail {tier_f1['tail']:.3f}",
          flush=True)
    return macro


def main():
    df = pd.read_csv(config.MANIFEST_CSV)
    tr = df[df["split"] == "train"]; te = df[df["split"] == "test"]
    n_j = np.sum([labels_from_report(t) for t in tr["report_text"]], axis=0)
    tiers = {"head": np.where(n_j >= 10)[0],
             "medium": np.where((n_j >= 3) & (n_j < 10))[0],
             "tail": np.where((n_j >= 1) & (n_j < 3))[0]}
    print(f"train {len(tr)} test {len(te)} | tier sizes: "
          f"head {len(tiers['head'])} medium {len(tiers['medium'])} tail {len(tiers['tail'])}\n")
    print(f"{'Loss':22} {'macro':>6} {'micro':>6} {'head':>6} {'medium':>7} {'tail':>6}")
    for name, fn in [("plain BCE", plain_bce), ("class-weighted BCE", weighted_bce),
                     ("LDAM-DRW (proposed)", ldam)]:
        run(name, fn, tr, te, n_j, tiers)


if __name__ == "__main__":
    main()
