"""
predicted_mask_eval.py -- deployment test: use the trained U-Net segmenter (never
trained on BUS-BRA) to PREDICT masks for BUS-BRA, recompute the ROI crops and
morphology features from those predicted masks, and re-run the enhanced autonomous
pipeline. If it still beats the baseline, the enhancement works without any
ground-truth masks -- i.e. it is deployable end-to-end.
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp

import config
from malignancy_data import load_dataset
from morphology_features import get_features, FEATURE_NAMES
from roi_crop import roi_crop
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote
from train_segmenter import SIZE, IMAGENET_MEAN, IMAGENET_STD

dev = config.DEVICE
PRED_MASK_DIR = "busbra/pred_masks"
PRED_ROI_DIR = "busbra/pred_roi"


def load_segmenter():
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    m.load_state_dict(torch.load("prod/segmenter.pt", map_location=dev))
    return m.eval()


@torch.no_grad()
def predict_mask(model, image_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x = (np.asarray(img.resize((SIZE, SIZE)), np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (Image.fromarray((p > 0.5).astype(np.uint8) * 255).resize((W, H), Image.NEAREST))


def build_busbra_predicted():
    os.makedirs(PRED_MASK_DIR, exist_ok=True)
    os.makedirs(PRED_ROI_DIR, exist_ok=True)
    seg = load_segmenter()
    bb = pd.read_csv("busbra_pairs.csv")
    X, rois = [], []
    for i, r in enumerate(bb.itertuples()):
        mp = f"{PRED_MASK_DIR}/{i:04d}.png"
        rp = f"{PRED_ROI_DIR}/{i:04d}.png"
        try:
            predict_mask(seg, r.image_path).convert("L").save(mp)
            roi = roi_crop(r.image_path, mp).convert("RGB")
            if roi.size[0] < 8 or roi.size[1] < 8:           # degenerate crop
                roi = Image.open(r.image_path).convert("RGB")
            roi.save(rp)
            feat = get_features(r.image_path, mp)
        except Exception as e:                                # bad image -> full frame
            print(f"  [fallback {i}] {type(e).__name__}", flush=True)
            Image.open(r.image_path).convert("RGB").save(rp)
            feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        X.append(feat)
        rois.append(rp)
        if (i + 1) % 300 == 0:
            print(f"  predicted {i+1}/{len(bb)}", flush=True)
    bb["pred_roi"] = rois
    np.save("busbra_pred_morph.npy", np.vstack(X))
    bb.to_csv("busbra_pred_pairs.csv", index=False)
    print(f"predicted masks + ROI + morph for {len(bb)} BUS-BRA images")


def main():
    from external_roi_eval import breast_frame
    build_busbra_predicted()

    # train the autonomous pipeline on BrEaST + 70% BUSI (GT masks for train is fine)
    br, br_m = breast_frame()
    bu = pd.read_csv("busi_pairs.csv"); bu_m = np.load("busi_morph.npy")
    from sklearn.model_selection import train_test_split
    tr_i, _ = train_test_split(np.arange(len(bu)), test_size=0.30,
                               stratify=bu["y"].values, random_state=config.SEED)
    train = pd.concat([br[["image_path", "roi_path", "y"]],
                       bu.iloc[tr_i][["image_path", "roi_path", "y"]]], ignore_index=True)
    trainM = np.vstack([br_m, bu_m[tr_i]]); y_tr = train["y"].values

    vr = train_vit_all(train[["roi_path", "y"]].rename(columns={"roi_path": "image_path"}).copy(), dev)
    torch.save(vr, "/tmp/vr_pred.pt")
    vf = train_vit_all(train[["image_path", "y"]].copy(), dev)
    torch.save(vf, "/tmp/vf_pred.pt")
    pr, pf = Predictor("/tmp/vr_pred.pt", dev), Predictor("/tmp/vf_pred.pt", dev)
    E_roi = embed_paths(train["roi_path"].tolist(), dev)
    E_full = embed_paths(train["image_path"].tolist(), dev)
    sc = StandardScaler().fit(trainM)
    morph_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(trainM), y_tr)

    bb = pd.read_csv("busbra_pred_pairs.csv"); bb_pm = np.load("busbra_pred_morph.npy")
    y = bb["y"].values
    # baseline (no masks needed): full-frame ViT + kNN
    base = _mean_logit(np.array([pf.predict_proba(p) for p in bb["image_path"]]),
                       knn_vote(E_full, y_tr, embed_paths(bb["image_path"].tolist(), dev)))
    # enhanced with PREDICTED masks: ROI-ViT + kNN(roi) + morph(predicted)
    p_vr = np.array([pr.predict_proba(p) for p in bb["pred_roi"]])
    p_kr = knn_vote(E_roi, y_tr, embed_paths(bb["pred_roi"].tolist(), dev))
    p_mo = morph_lr.predict_proba(sc.transform(bb_pm))[:, 1]
    enh = _mean_logit(_mean_logit(p_vr, p_kr), p_mo)

    print(f"\nBUS-BRA (n={len(y)}) with PREDICTED masks (unseen domain):")
    print(f"  baseline (no masks)          AUC {roc_auc_score(y, base):.3f}")
    print(f"  enhanced (predicted masks)   AUC {roc_auc_score(y, enh):.3f}")
    print(f"  (enhanced with GT masks was 0.884)")


if __name__ == "__main__":
    main()
