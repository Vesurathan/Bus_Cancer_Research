"""
retrieval_tool.py -- Phase A step 2. Wraps the FAISS index as a query tool.
Loads BiomedCLIP once; encodes a query image; returns the top-k most similar
TRAIN-corpus records (with their exemplar reports) for RAG grounding.
"""
import os, pickle
import numpy as np
from PIL import Image
import torch, open_clip, faiss
import config


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(os.path.join(config.RETRIEVAL_DIR, "biomedclip_faiss.index"))
        with open(os.path.join(config.RETRIEVAL_DIR, "corpus_metadata.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        model, _, self.preprocess = open_clip.create_model_and_transforms(config.MODEL_ID)
        self.model = model.to(config.DEVICE).eval()

    @torch.no_grad()
    def _encode(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(config.DEVICE)
        e = torch.nn.functional.normalize(self.model.encode_image(x), dim=-1)
        return e.cpu().float().numpy().astype("float32")

    def retrieve(self, image_path, k=None):
        """Return list of dicts: {report_text, dataset, sim}. Reports may be empty
        for aux-dataset neighbours; callers should prefer non-empty reports."""
        k = k or config.TOP_K
        sims, ids = self.index.search(self._encode(image_path), k)
        out = []
        for j, s in zip(ids[0], sims[0]):
            if j < 0:
                continue
            rec = dict(self.meta[j]); rec["sim"] = float(s)
            out.append(rec)
        return out
