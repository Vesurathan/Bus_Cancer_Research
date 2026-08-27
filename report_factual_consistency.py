"""
report_factual_consistency.py -- a factual (not lexical) evaluation of the
generated reports. Unlike BLEU/ROUGE/METEOR/CIDEr, this checks whether the report
states clinically correct content by extracting structured fields and comparing
them to the ground-truth report and findings:

  BI-RADS agreement   exact and within-one-category match of the stated BI-RADS
  findings precision  fraction of asserted findings that are truly present
  findings recall     fraction of true findings that are asserted

Reported for each report-generation configuration.
"""
import json
import re

import numpy as np
import pandas as pd

CONFIGS = [
    ("Generator (monolithic)", "pred_cv_monolithic.csv"),
    ("+ Retrieval (RAG)", "pred_cv_rag.csv"),
    ("+ Verifier (rule-based)", "pred_cv_full.csv"),
    ("+ Verifier (LLM refiner)", "pred_cv_full_llm.csv"),
]


def birads(text):
    m = re.search(r"bi-?rads\s*([0-6])", str(text).lower())
    return int(m.group(1)) if m else None


def findings(text):
    m = re.search(r"findings?:\s*(.*)", str(text).lower())
    if not m:
        return set()
    seg = re.split(r"\bbi-?rads\b", m.group(1))[0]
    return {t.strip(" .") for t in seg.split(",") if t.strip(" .")}


def evaluate(csv):
    d = pd.read_csv(csv).fillna("")
    present = bex = bw1 = nb = 0
    precs, recs = [], []
    for _, r in d.iterrows():
        bp, bg = birads(r["pred_report"]), birads(r["gold_report"])
        if bp is not None:
            present += 1
        if bp is not None and bg is not None:
            nb += 1
            bex += (bp == bg)
            bw1 += (abs(bp - bg) <= 1)
        fp = findings(r["pred_report"])
        fg = set(json.loads(r["gold_terms"])) if r["gold_terms"] else findings(r["gold_report"])
        if fp:
            precs.append(len(fp & fg) / len(fp))
        if fg:
            recs.append(len(fp & fg) / len(fg))
    ex = bex / nb if nb else 0.0
    w1 = bw1 / nb if nb else 0.0
    return (present / len(d), ex, w1,
            np.mean(precs) if precs else 0.0, np.mean(recs) if recs else 0.0)


if __name__ == "__main__":
    print(f"{'Configuration':26}{'BI-RADS stated':>15}{'exact*':>8}{'within-1*':>10}"
          f"{'Find-prec':>11}{'Find-rec':>10}")
    for name, csv in CONFIGS:
        pres, ex, w1, pr, rc = evaluate(csv)
        print(f"{name:26}{pres:>15.3f}{ex:>8.3f}{w1:>10.3f}{pr:>11.3f}{rc:>10.3f}")
    print("* exact / within-1 computed only over reports that state a BI-RADS category")
