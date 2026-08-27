"""
predicted_mask_eval_v2.py -- does the stronger segmenter (v2) make the descriptor-
free enhancement transfer with PREDICTED masks? Uses segmenter_v2 to predict masks
for the held-out BUS-BRA images (unseen by the segmenter), recomputes ROI + morph,
and compares baseline vs enhanced (predicted masks) and vs enhanced (GT masks).
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp

import config
from morphology_features import get_features, FEATURE_NAMES
from roi_crop import roi_crop
from train_production import train_vit_all
from malignancy_model import Predictor
from evaluate_external import embed as embed_paths
from promote_combined import _mean_logit, knn_vote
from train_segmenter import SIZE, IMAGENET_MEAN, IMAGENET_STD
from external_roi_eval import breast_frame

dev = config.DEVICE


def load_seg(path):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev)); return m.eval()


@torch.no_grad()
def predict_mask(model, image_path):
    img = Image.open(image_path).convert("RGB"); W, H = img.size
    x = (np.asarray(img.resize((SIZE, SIZE)), np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    p = torch.sigmoid(model(torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)))[0, 0].cpu().numpy()
    return Image.fromarray((p > 0.5).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)


def main():
    seg = load_seg("prod/segmenter_v2.pt")
    hold = pd.read_csv("busbra_seg_holdout.csv")          # image_path, mask_path (GT)
    bb = pd.read_csv("busbra_pairs.csv")[["image_path", "y"]]
    test = hold.merge(bb, on="image_path", how="left").dropna(subset=["y"])
    print(f"BUS-BRA held-out test (unseen by segmenter): {len(test)}", flush=True)

    os.makedirs("busbra/predv2", exist_ok=True)
    pred_roi, pred_m, gt_roi, gt_m = [], [], [], []
    for i, r in enumerate(test.itertuples()):
        pm = f"busbra/predv2/{i:04d}.png"
        predict_mask(seg, r.image_path).convert("L").save(pm)
        pr = f"busbra/predv2/roi_{i:04d}.png"
        roi_crop(r.image_path, pm).convert("RGB").save(pr)
        pred_roi.append(pr); pred_m.append(get_features(r.image_path, pm))
        gr = f"busbra/predv2/gtroi_{i:04d}.png"
        roi_crop(r.image_path, r.mask_path).convert("RGB").save(gr)
        gt_roi.append(gr); gt_m.append(get_features(r.image_path, r.mask_path))
    pred_m = np.vstack(pred_m); gt_m = np.vstack(gt_m); y = test["y"].values.astype(int)

    # train classifier on BrEaST + 70% BUSI (GT masks) -- never sees BUS-BRA
    br, br_m = breast_frame()
    bu = pd.read_csv("busi_pairs.csv"); bu_m = np.load("busi_morph.npy")
    tr_i, _ = train_test_split(np.arange(len(bu)), test_size=0.30, stratify=bu["y"].values,
                               random_state=config.SEED)
    train = pd.concat([br[["image_path", "roi_path", "y"]],
                       bu.iloc[tr_i][["image_path", "roi_path", "y"]]], ignore_index=True)
    trainM = np.vstack([br_m, bu_m[tr_i]]); y_tr = train["y"].values

    vf = train_vit_all(train[["image_path", "y"]].copy(), dev); torch.save(vf, "/tmp/vf2.pt")
    vr = train_vit_all(train[["roi_path", "y"]].rename(columns={"roi_path": "image_path"}).copy(), dev)
    torch.save(vr, "/tmp/vr2.pt")
    pf, pr_ = Predictor("/tmp/vf2.pt", dev), Predictor("/tmp/vr2.pt", dev)
    E_full = embed_paths(train["image_path"].tolist(), dev)
    E_roi = embed_paths(train["roi_path"].tolist(), dev)
    sc = StandardScaler().fit(trainM)
    morph_lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(trainM), y_tr)

    base = _mean_logit(np.array([pf.predict_proba(p) for p in test["image_path"]]),
                       knn_vote(E_full, y_tr, embed_paths(test["image_path"].tolist(), dev)))

    def enhanced(rois, mX):
        p_vr = np.array([pr_.predict_proba(p) for p in rois])
        p_kr = knn_vote(E_roi, y_tr, embed_paths(list(rois), dev))
        p_mo = morph_lr.predict_proba(sc.transform(mX))[:, 1]
        return _mean_logit(_mean_logit(p_vr, p_kr), p_mo)

    print(f"\nBUS-BRA held-out (n={len(y)}), autonomous image-only:")
    print(f"  baseline (no masks)             AUC {roc_auc_score(y, base):.3f}")
    print(f"  enhanced (segmenter-v2 masks)   AUC {roc_auc_score(y, enhanced(pred_roi, pred_m)):.3f}")
    print(f"  enhanced (ground-truth masks)   AUC {roc_auc_score(y, enhanced(gt_roi, gt_m)):.3f}")


if __name__ == "__main__":
    main()
