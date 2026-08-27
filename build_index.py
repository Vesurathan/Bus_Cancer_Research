"""
build_index.py -- Phase A step 1. BiomedCLIP + FAISS index over the TRAIN split only.
Leakage guard: val/test images are never indexed. Run once before the graph.

    exec(open('build_index.py').read())

Manifest schema (CSV): image_path, report_text, dataset, split
"""
import os, json, pickle
import numpy as np, pandas as pd
from PIL import Image
import torch, open_clip, faiss
import config


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


@torch.no_grad()
def _embed(model, preprocess, paths, device, bs=32):
    feats, kept, buf, pos = [], [], [], []
    def flush():
        if not buf: return
        e = model.encode_image(torch.stack(buf).to(device))
        e = torch.nn.functional.normalize(e, dim=-1)
        feats.append(e.cpu().float().numpy()); kept.extend(pos)
        buf.clear(); pos.clear()
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as err:
            print(f"[skip] {p}: {err}"); continue
        buf.append(preprocess(img)); pos.append(i)
        if len(buf) == bs: flush()
    flush()
    if not feats: raise RuntimeError("No images embedded -- check paths.")
    return np.vstack(feats).astype("float32"), kept


def main():
    _set_seed(config.SEED)
    os.makedirs(config.RETRIEVAL_DIR, exist_ok=True)
    print("Device:", config.DEVICE)

    df = pd.read_csv(config.MANIFEST_CSV)
    need = {"image_path", "report_text", "dataset", "split"}
    if need - set(df.columns): raise ValueError(f"Manifest missing: {need - set(df.columns)}")

    corpus = df[df["split"] == "train"].reset_index(drop=True)
    corpus["report_text"] = corpus["report_text"].fillna("")
    print(f"Corpus (train): {len(corpus)} | with reports: "
          f"{(corpus['report_text'].str.len() > 0).sum()}")

    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_ID)
    model = model.to(config.DEVICE).eval()
    emb, kept = _embed(model, preprocess, corpus["image_path"].tolist(), config.DEVICE)
    corpus = corpus.iloc[kept].reset_index(drop=True)
    print(f"Embedded {emb.shape[0]} images, dim={emb.shape[1]}")

    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    faiss.write_index(index, os.path.join(config.RETRIEVAL_DIR, "biomedclip_faiss.index"))
    with open(os.path.join(config.RETRIEVAL_DIR, "corpus_metadata.pkl"), "wb") as f:
        pickle.dump(corpus[["image_path", "report_text", "dataset"]].to_dict("records"), f)
    with open(os.path.join(config.RETRIEVAL_DIR, "index_info.json"), "w") as f:
        json.dump({"model_id": config.MODEL_ID, "n_vectors": int(index.ntotal),
                   "dim": int(emb.shape[1]), "indexed_split": "train",
                   "seed": config.SEED}, f, indent=2)
    print("Index written to", config.RETRIEVAL_DIR)


if __name__ == "__main__":
    main()
