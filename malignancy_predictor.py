"""
malignancy_predictor.py -- deployable malignancy predictor. Loads the artifacts
saved by train_production.py and scores a NEW image (+ optional BI-RADS
descriptors): ViT visual stream + BiomedCLIP kNN vote (+ descriptor stream if
descriptors are given) -> stacked-fusion P(malignant). Uses the full fusion when
descriptors are present, and the image-only fusion otherwise.
"""
import os, json, pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch

import config
from malignancy_model import build_model, get_transform
from malignancy_data import DESCRIPTOR_COLS

PROD = "./prod"


class MalignancyPredictor:
    def __init__(self, prod_dir=PROD, device=None):
        self.device = device or config.DEVICE
        self.meta = json.load(open(f"{prod_dir}/meta.json"))
        self.k = self.meta.get("knn_k", 7)

        # visual stream
        self.vit = build_model().to(self.device).eval()
        self.vit.load_state_dict(torch.load(f"{prod_dir}/vit_malignancy.pt",
                                            map_location=self.device))
        self.tf = get_transform(train=False)

        # descriptor stream
        with open(f"{prod_dir}/descriptor.pkl", "rb") as f:
            d = pickle.load(f)
        self.feat, self.desc_lr = d["featurizer"], d["logreg"]

        # kNN bank
        bank = np.load(f"{prod_dir}/knn_bank.npz")
        self.E_bank, self.y_bank = bank["E"], bank["y"]

        # Platt calibrator for the image fusion (calibrate on held-out data) so the
        # reported P(malignant) is trustworthy.
        self.cal_img = self._load_opt(f"{prod_dir}/calibrator_image.pkl")

        # optional real BI-RADS predictor (birads_model.py)
        self.birads = self._load_opt(f"{prod_dir}/birads.pkl")

        self._clip = None  # lazy BiomedCLIP for embedding the query

    @staticmethod
    def _load_opt(path):
        return pickle.load(open(path, "rb")) if os.path.exists(path) else None

    @staticmethod
    def _calibrate(cal, p):
        if cal is None:
            return p
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return float(cal["platt"].predict_proba([[z]])[0, 1])

    # -- streams ------------------------------------------------------------ #
    @torch.no_grad()
    def _vit_proba(self, image_path):
        x = self.tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.vit(x)).item())

    def _embed(self, image_path):
        if self._clip is None:
            import open_clip
            m, _, pre = open_clip.create_model_and_transforms(config.MODEL_ID)
            self._clip = (m.to(self.device).eval(), pre)
        model, pre = self._clip
        with torch.no_grad():
            x = pre(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
            e = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
        return e.cpu().float().numpy()[0]

    def _knn(self, image_path):
        q = self._embed(image_path)
        sims = self.E_bank @ q
        nn = np.argsort(-sims)[:self.k]
        w = np.clip(sims[nn], 1e-6, None)
        p = float(np.average(self.y_bank[nn], weights=w))
        ood = float(1.0 - sims.max())          # distance to nearest train case
        neigh = [{"sim": float(sims[j]), "label": int(self.y_bank[j])} for j in nn]
        return p, ood, neigh, q

    def _birads(self, emb, descriptors):
        """Predicted BI-RADS category + whether descriptors informed it."""
        if self.birads is None:
            return None, False
        E = np.asarray(emb).reshape(1, -1)
        if descriptors:
            row = {c: "" for c in DESCRIPTOR_COLS}
            row["age"] = descriptors.get("age", np.nan)
            for c in DESCRIPTOR_COLS:
                if c in descriptors and descriptors[c]:
                    row[c] = str(descriptors[c]).lower().strip()
            D = self.birads["featurizer"].transform(pd.DataFrame([row]))
            k = int(self.birads["full"].predict(np.hstack([E, D]))[0])
            return self.birads["order"][k], True
        k = int(self.birads["img"].predict(E)[0])
        return self.birads["order"][k], False

    def _descriptor_proba(self, descriptors):
        row = {c: "" for c in DESCRIPTOR_COLS}
        row["age"] = descriptors.get("age", np.nan) if descriptors else np.nan
        for c in DESCRIPTOR_COLS:
            if descriptors and c in descriptors and descriptors[c]:
                row[c] = str(descriptors[c]).lower().strip()
        X = self.feat.transform(pd.DataFrame([row]))
        return float(self.desc_lr.predict_proba(X)[0, 1])

    # -- fuse --------------------------------------------------------------- #
    @staticmethod
    def _mean_logit(a, b):
        la = np.log(np.clip(a, 1e-6, 1 - 1e-6) / (1 - np.clip(a, 1e-6, 1 - 1e-6)))
        lb = np.log(np.clip(b, 1e-6, 1 - 1e-6) / (1 - np.clip(b, 1e-6, 1 - 1e-6)))
        return float(1 / (1 + np.exp(-(la + lb) / 2)))

    def predict(self, image_path, descriptors=None):
        p_vit = self._vit_proba(image_path)
        p_knn, ood, neigh, emb = self._knn(image_path)
        birads, birads_from_desc = self._birads(emb, descriptors)
        streams = {"vit": p_vit, "knn": p_knn}

        # image evidence: mean-logit fusion of ViT + kNN, then calibrate
        p_img = self._calibrate(self.cal_img, self._mean_logit(p_vit, p_knn))
        if descriptors:
            p_desc = self._descriptor_proba(descriptors)
            streams["descriptor"] = p_desc
            p = 0.5 * (p_img + p_desc)                  # blend image + descriptors
        else:
            p = p_img

        thr = self.meta["threshold"]
        # confidence = how decisively the calibrated probability sits on its side
        # of the DECISION THRESHOLD (0 at the boundary, 1 at the extreme), so it
        # stays meaningful even though the threshold is not 0.5.
        conf = (p - thr) / max(1e-6, 1 - thr) if p >= thr else (thr - p) / max(1e-6, thr)
        return {
            "p_malignant": p,
            "decision": "malignant" if p >= thr else "benign",
            "confidence": max(0.0, min(1.0, conf)),
            "used_descriptors": bool(descriptors),
            "streams": streams,
            "ood_distance": ood,
            "neighbours": neigh,
            "threshold": thr,
            "birads": birads,
            "birads_from_descriptors": birads_from_desc,
        }
