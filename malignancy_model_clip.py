"""
malignancy_model_clip.py -- BiomedCLIP-backboned visual stream (drop-in alternative
to malignancy_model.py). Uses the medical-pretrained BiomedCLIP image encoder with a
linear malignancy head, fine-tuned with block-freezing to control overfitting on the
small (256-case) internal set.

Same interface as malignancy_model.py so evaluate_malignancy.py / the harness can swap
it in:
    train_fold_proba(train_df, test_df) -> P(malignant) for the test rows
    Predictor(ckpt).predict_proba(image_path)

Configuration via env (so the freeze depth / schedule can be swept without edits):
    CLIP_FREEZE = head | last1 | last2 | last4 | full   (default last2)
    CLIP_EPOCHS, CLIP_LR_ENC, CLIP_LR_HEAD, CLIP_WD, CLIP_DROPOUT
"""
import gc
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import open_clip

import config

# BiomedCLIP (OpenAI-CLIP) normalisation — NOT ImageNet stats.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

FREEZE = os.environ.get("CLIP_FREEZE", "last2")
EPOCHS = int(os.environ.get("CLIP_EPOCHS", "12"))
LR_ENC = float(os.environ.get("CLIP_LR_ENC", "1e-5"))
LR_HEAD = float(os.environ.get("CLIP_LR_HEAD", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("CLIP_WD", "0.05"))
DROPOUT = float(os.environ.get("CLIP_DROPOUT", "0.2"))
BATCH_SIZE = int(os.environ.get("CLIP_BATCH", "16"))


def get_transform(train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class BiomedCLIPClassifier(nn.Module):
    """BiomedCLIP image encoder (encode_image -> 512-d) + linear malignancy head."""
    def __init__(self, freeze=FREEZE, dropout=DROPOUT):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(config.MODEL_ID)
        # drop the text tower to save memory; we only use the image encoder
        if hasattr(self.clip, "text"):
            self.clip.text = None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(512, 1)
        self._apply_freeze(freeze)

    def _apply_freeze(self, freeze):
        blocks = self.clip.visual.trunk.blocks
        n = len(blocks)
        if freeze == "full":
            unfreeze = set(range(n))
        else:
            k = {"head": 0, "last1": 1, "last2": 2, "last4": 4}.get(freeze, 2)
            unfreeze = set(range(n - k, n))
        for name, p in self.clip.named_parameters():
            if not name.startswith("visual"):
                p.requires_grad = False
                continue
            train = False
            if name.startswith("visual.head") or "trunk.norm" in name:
                train = True                      # projection + final norm
            else:
                mo = re.search(r"visual\.trunk\.blocks\.(\d+)\.", name)
                if mo and int(mo.group(1)) in unfreeze:
                    train = True
            p.requires_grad = train

    def trainable_param_groups(self):
        enc, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith("head") else enc).append(p)
        groups = [{"params": head, "lr": LR_HEAD}]
        if enc:
            groups.append({"params": enc, "lr": LR_ENC})
        return groups

    def forward(self, x):
        feats = self.clip.encode_image(x)         # (B, 512)
        return self.head(self.drop(feats))        # (B, 1) logit


def build_model(freeze=FREEZE):
    return BiomedCLIPClassifier(freeze=freeze)


class _DS(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True)
        self.tf = get_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.tf(img), torch.tensor([float(r["y"])], dtype=torch.float32)


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _proba(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy().ravel())
    return np.concatenate(out)


def train_fold_proba(train_df, test_df, epochs=EPOCHS, device=None, log_prefix="", freeze=FREEZE):
    device = device or config.DEVICE
    n_pos = float(train_df["y"].sum())
    n_neg = float((train_df["y"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    tr = DataLoader(_DS(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te = DataLoader(_DS(test_df, False), batch_size=BATCH_SIZE, num_workers=0)

    model = build_model(freeze).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.trainable_param_groups(), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for x, yb in tr:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(x), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr.dataset):.4f}", flush=True)

    p = _proba(model, te, device)
    _free(model, opt, sched, crit)
    return p


class Predictor:
    def __init__(self, ckpt_path, device=None, freeze=FREEZE):
        self.device = device or config.DEVICE
        self.model = build_model(freeze).to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.tf = get_transform(train=False)

    @torch.no_grad()
    def predict_proba(self, image_path):
        x = self.tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(x)).item())
