"""
train_finding_agent.py -- train the 28-term interpretation classifier with a
multi-label LDAM-DRW loss + curriculum margins (your contribution).

    python train_finding_agent.py

LDAM-DRW, multi-label form:
  - margin per term  m_j = C / n_j^(1/4), scaled so the largest margin = MAX_MARGIN.
    The margin is SUBTRACTED from the logit of a term when it is PRESENT, forcing
    the model to be more confident on rare terms (Cao et al., 2019, adapted to the
    per-term binary setting).
  - curriculum: the margin is annealed from 0 -> full over the first half of training.
  - DRW: after ~62.5% of training, per-term class-balanced weights up-weight the
    rare terms in the BCE (deferred re-weighting).

train_and_eval() is the reusable entry point: evaluate_cv.py calls it once per
fold so each fold retrains on its own train split and is evaluated leak-free.
main() below is the single-split learning check + first standalone checkpoint.
"""
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config
from vocab import TERMS, SYNONYMS
from finding_model import build_model, get_transform


def _free(*objs):
    """Release model/optimizer memory and empty the MPS/CUDA cache so long
    multi-fold sweeps don't accumulate allocations until the host OOMs."""
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

EPOCHS             = 40
BATCH_SIZE         = 16
LR                 = 2e-5
WEIGHT_DECAY       = 0.05
MAX_MARGIN         = 0.5
MARGIN_WARMUP_FRAC = 0.5    # curriculum reaches full strength at this fraction of training
DRW_START_FRAC     = 0.625  # deferred re-weighting kicks in at this fraction of training
THRESH             = 0.5


def labels_from_report(text):
    text = str(text).lower()
    return [1.0 if any(s in text for s in SYNONYMS[t]) else 0.0 for t in TERMS]


class BUSReportDS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.tf = get_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        y = torch.tensor(labels_from_report(r["report_text"]))
        return self.tf(img), y


class LDAMDRWLoss(torch.nn.Module):
    def __init__(self, n_j, device, margin_warmup, drw_start):
        super().__init__()
        n = np.maximum(n_j, 1).astype("float32")
        m = 1.0 / np.power(n, 0.25)
        m = m / m.max() * MAX_MARGIN
        self.margins = torch.tensor(m, device=device)
        beta = 0.9999
        w = (1 - beta) / (1 - np.power(beta, n))
        w = w / w.mean()
        self.drw_w = torch.tensor(w.astype("float32"), device=device)
        self.ones = torch.ones_like(self.drw_w)
        self.margin_warmup = margin_warmup
        self.drw_start = drw_start

    def forward(self, logits, targets, epoch):
        alpha = min(1.0, epoch / max(1, self.margin_warmup))      # curriculum
        z = logits - alpha * self.margins.unsqueeze(0) * targets  # margin on positives
        w = self.drw_w if epoch >= self.drw_start else self.ones
        loss = F.binary_cross_entropy_with_logits(z, targets, reduction="none")
        return (loss * w.unsqueeze(0)).mean()


def macro_f1(model, loader, device):
    model.eval()
    P, G = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            P.append(p); G.append(y.numpy())
    P, G = np.vstack(P) >= THRESH, np.vstack(G) >= 0.5
    f1s = []
    for j in range(len(TERMS)):
        tp = np.sum(P[:, j] & G[:, j]); fp = np.sum(P[:, j] & ~G[:, j]); fn = np.sum(~P[:, j] & G[:, j])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def train_and_eval(df, epochs, ckpt_path, device, log_prefix=""):
    """Train on df's train split, eval on its test split, save to ckpt_path.
    Shared by the single-split check (main()) and evaluate_cv.py, which calls
    this once per fold so every fold gets its own leak-free checkpoint and
    every term is trained wherever its examples land."""
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print(f"{log_prefix}train {len(tr)} | test {len(te)} | device {device}")

    tr_ds = BUSReportDS(tr, train=True)
    te_ds = BUSReportDS(te, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, num_workers=0)

    n_j = np.sum([labels_from_report(t) for t in tr["report_text"]], axis=0)
    print(f"{log_prefix}positives per term (min/max):", int(n_j.min()), int(n_j.max()))

    margin_warmup = max(1, round(epochs * MARGIN_WARMUP_FRAC))
    drw_start = max(1, round(epochs * DRW_START_FRAC))

    model = build_model().to(device)
    crit = LDAMDRWLoss(n_j, device, margin_warmup, drw_start)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    f1 = 0.0
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y, epoch)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            f1 = macro_f1(model, te_ld, device)
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr_ds):.4f}  test macro-F1 {f1:.3f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"{log_prefix}saved checkpoint -> {ckpt_path}")
    _free(model, opt, sched, crit)
    return f1


def main():
    df = pd.read_csv(config.MANIFEST_CSV)
    train_and_eval(df, EPOCHS, config.CLF_CKPT, config.DEVICE)
    print("\nNow set USE_CLASSIFIER=True in config.py and rerun the pipeline.")


if __name__ == "__main__":
    main()
