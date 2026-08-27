"""
compute_cider.py -- add the CIDEr metric to the report-generation ablation,
complementing the existing BLEU-4 / ROUGE-L / METEOR scores. CIDEr (Vedantam
et al., 2015) uses TF-IDF-weighted n-gram consensus and is the fourth standard
report-generation metric used by Shivakumar, Mahmood & Khatoon (2026).
"""
import re

import pandas as pd
from pycocoevalcap.cider.cider import Cider

# ablation config -> its cached CV predictions
CONFIGS = [
    ("Generator (monolithic)", "pred_cv_monolithic.csv"),
    ("+ Retrieval (RAG)", "pred_cv_rag.csv"),
    ("+ Verifier (rule-based repair)", "pred_cv_full.csv"),
    ("+ Verifier (LLM refiner, proposed)", "pred_cv_full_llm.csv"),
]


def tok(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cider(csv):
    d = pd.read_csv(csv).fillna("")
    gts = {i: [tok(g)] for i, g in enumerate(d["gold_report"])}
    res = {i: [tok(p)] for i, p in enumerate(d["pred_report"])}
    score, _ = Cider().compute_score(gts, res)
    return score


if __name__ == "__main__":
    print(f"{'Configuration':40} {'CIDEr':>7}")
    for name, csv in CONFIGS:
        print(f"{name:40} {cider(csv):7.3f}")
