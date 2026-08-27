"""
train_generator.py -- train the ViT+Transformer-decoder report generator via
teacher forcing (next-token cross-entropy over gen_vocab.py's word vocabulary).

    python train_generator.py

train_and_eval() is the reusable entry point: evaluate_cv.py can call it once
per fold later, mirroring train_finding_agent.py's leak-free CV pattern. main()
below is the single-split learning check + first standalone checkpoint.
"""
import gc
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sacrebleu

import config
from gen_vocab import PAD_ID, encode, detokenize
from generator_model import ReportGenerator, get_transform


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

EPOCHS       = 60
BATCH_SIZE   = 16
LR           = 3e-4
WEIGHT_DECAY = 0.01


class BUSGenDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tf = get_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        ids = encode(r["report_text"], config.MAX_LEN)
        return self.tf(img), torch.tensor(ids, dtype=torch.long)


def _batch_loss(model, images, ids, device, criterion):
    images, ids = images.to(device), ids.to(device)
    logits = model(images, ids[:, :-1])
    return criterion(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))


@torch.no_grad()
def _test_bleu(model, loader, device):
    model.eval()
    hyps, refs = [], []
    for images, ids in loader:
        gen_ids = model.greedy_generate(images.to(device), device)
        hyps.extend(detokenize(row) for row in gen_ids.tolist())
        refs.extend(detokenize(row) for row in ids.tolist())
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def train_and_eval(df, epochs, ckpt_path, device, log_prefix=""):
    """Train on df's train split, eval BLEU-4 on its test split, save to
    ckpt_path. Shared by the single-split check (main()) and, later, a
    leak-free per-fold CV loop."""
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print(f"{log_prefix}train {len(tr)} | test {len(te)} | device {device}")

    tr_ld = DataLoader(BUSGenDS(tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(BUSGenDS(te), batch_size=BATCH_SIZE, num_workers=0)

    model = ReportGenerator(max_len=config.MAX_LEN).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bleu = 0.0
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for images, ids in tr_ld:
            opt.zero_grad()
            loss = _batch_loss(model, images, ids, device, criterion)
            loss.backward(); opt.step()
            tot += loss.item() * images.size(0)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            bleu = _test_bleu(model, te_ld, device)
            print(f"{log_prefix}epoch {epoch+1:2d}  loss {tot/len(tr_ld.dataset):.4f}  "
                  f"test BLEU-4 {bleu:.2f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"{log_prefix}saved checkpoint -> {ckpt_path}")
    _free(model, opt, sched)
    return bleu


def main():
    df = pd.read_csv(config.MANIFEST_CSV)
    train_and_eval(df, EPOCHS, config.GEN_CKPT, config.DEVICE)
    print("\nNow set DRAFT_MODE='external' in config.py and rerun the pipeline.")


if __name__ == "__main__":
    main()
