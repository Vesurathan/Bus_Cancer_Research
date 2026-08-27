"""
evaluate_external.py -- cross-dataset validation on BUSI. Train the IMAGE streams
on all of BrEaST, test on all of BUSI (a different institution/scanner), and ask
the question that makes the agent's deferral mechanism actually novel:

  Under dataset shift, (1) how much does the image predictor degrade, and (2) does
  the uncertainty / out-of-distribution signal RISE on the shifted data so the
  agent abstains on exactly the cases it should?

Streams here are image-only (BUSI has no BI-RADS descriptors):
  vit-mc  -- ViT trained on BrEaST, MC-dropout mean P(malignant) on BUSI + epistemic std
  knn     -- BiomedCLIP kNN vote over BrEaST labels
  ood     -- embedding distance to the nearest BrEaST case (1 - max cosine sim): a
             training-free distribution-shift detector

Reports BUSI AUC vs the in-distribution image AUC, whether epistemic/OOD signals
are elevated on BUSI, and risk-coverage (does abstaining on high-uncertainty BUSI
cases recover accuracy?).

    python evaluate_external.py        # needs ./busi present + a free GPU
"""
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch

import config
from malignancy_data import load_dataset
from malignancy_model import _DS, _free, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY
from epistemic_vit import build_dropout_model, mc_proba
from busi_data import load_busi
from selective_prediction import risk_coverage, aurc, sel_acc_at
import torch.nn as nn

KNN_K = 7


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def train_full_vit(df, device, log_prefix="[ext-vit] "):
    tr = DataLoader(_DS(df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    n_pos = float(df["y"].sum()); n_neg = float((df["y"] == 0).sum())
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
    return model


def embed(paths, device):
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(config.MODEL_ID)
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for p in paths:
            x = pre(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            e = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
            out.append(e.cpu().float().numpy()[0])
    _free(model)
    return np.vstack(out).astype("float32")


def knn_and_ood(E_bank, y_bank, E_query, k=KNN_K):
    sims = E_query @ E_bank.T
    p_knn = np.zeros(len(E_query)); ood = np.zeros(len(E_query))
    for i in range(len(E_query)):
        nn_idx = np.argsort(-sims[i])[:k]
        w = np.clip(sims[i, nn_idx], 1e-6, None)
        p_knn[i] = np.average(y_bank[nn_idx], weights=w)
        ood[i] = 1.0 - float(sims[i].max())        # distance to nearest train case
    return p_knn, ood


def report_selative(name, y, p_img, signals):
    yhat = (p_img >= 0.5).astype(int)
    correct = (yhat == y).astype(int)
    print(f"\n{name}: image-only AUC {roc_auc_score(y,p_img):.3f}  acc {correct.mean():.3f}")
    print(f"  {'signal':28s}{'AURC':>7}{'cov100':>8}{'cov80':>7}{'cov60':>7}")
    for sname, rel in signals.items():
        print(f"  {sname:28s}{aurc(correct,rel):7.3f}"
              f"{sel_acc_at(correct,rel,1.0):8.3f}{sel_acc_at(correct,rel,0.8):7.3f}"
              f"{sel_acc_at(correct,rel,0.6):7.3f}")


def main():
    dev = config.DEVICE
    br = load_dataset()
    bu = load_busi()
    print(f"BrEaST train {len(br)} (mal {int(br['y'].sum())}) | "
          f"BUSI test {len(bu)} (mal {int(bu['y'].sum())})")

    # --- visual stream: train on all BrEaST, MC-dropout inference on BUSI ---
    model = train_full_vit(br, dev)
    bu_loader = DataLoader(_DS(bu, False), batch_size=BATCH_SIZE, num_workers=0)
    p_vit, u_epi = mc_proba(model, bu_loader, dev)
    _free(model)

    # --- kNN + OOD over BiomedCLIP embeddings ---
    E_br = embed(br["image_path"].tolist(), dev)
    E_bu = embed(bu["image_path"].tolist(), dev)
    p_knn, ood = knn_and_ood(E_br, br["y"].values, E_bu)
    # in-distribution OOD reference: BrEaST-to-BrEaST nearest-OTHER distance
    sims_bb = E_br @ E_br.T; np.fill_diagonal(sims_bb, -1)
    ood_indist = 1.0 - sims_bb.max(1)

    y = bu["y"].values
    p_img = _sig(np.mean([_logit(p_vit), _logit(p_knn)], axis=0))

    print(f"\nBUSI image AUC: vit-mc {roc_auc_score(y,p_vit):.3f} | "
          f"knn {roc_auc_score(y,p_knn):.3f} | fused {roc_auc_score(y,p_img):.3f}")
    print("(in-distribution BrEaST image AUC was ~0.83 -> the drop is the shift)")
    print(f"\nShift detection (higher on shifted data = good):")
    print(f"  epistemic std : BUSI mean {u_epi.mean():.3f}")
    print(f"  OOD distance  : BUSI mean {ood.mean():.3f}  vs  BrEaST mean {ood_indist.mean():.3f}")

    conf = np.abs(p_img - 0.5)
    signals = {
        "epistemic (MC-dropout)": -u_epi,
        "OOD embedding distance": -ood,
        "confidence (baseline)": conf,
        "epistemic + OOD": -(u_epi + ood),
        "random": np.random.default_rng(0).random(len(y)),
    }
    report_selative("BUSI selective prediction", y, p_img, signals)


if __name__ == "__main__":
    main()
