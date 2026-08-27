"""
busbra_data.py -- loader for BUS-BRA (Gomez-Flores et al., "BUS-BRA: A breast
ultrasound dataset for assessing computer-aided diagnosis systems", Medical
Physics, 51, 3110-3123, 2024). Used to expand the training corpus and as a
second external validation site.

Layout (as unzipped from Zenodo record 8231412):
    ./busbra/BUSBRA/
        Images/       bus_0001-l.png, bus_0001-r.png, ...
        Masks/        mask_0001-l.png, ...
        bus_data.csv  ID, Case, Histology, Pathology, BIRADS, Device, ...
        5-fold-cv.csv official PATIENT-LEVEL folds

1,875 images from 1,064 patients (607 malignant / 1,268 benign), four scanners,
National Institute of Cancer, Rio de Janeiro. Crucially, several patients
contribute TWO views (mean 1.76 images/case), so splits must be grouped by
`Case` or the same lesion leaks across train and test. The official kFold
column already respects this and is used by default.

BUS-BRA carries BI-RADS but NOT the eight BrEaST radiologist descriptor fields,
so it exercises the IMAGE streams and the BI-RADS head -- the descriptor stream
remains BrEaST-only.

    python busbra_data.py    # sanity: counts, balance, group integrity
"""
import os

import numpy as np
import pandas as pd

BUSBRA_DIR = "./busbra/BUSBRA"


def load_busbra(root=BUSBRA_DIR):
    """Return DataFrame: image_path, case_id, patient_id, y, birads, histology,
    device, fold. y = 1 for malignant. Only rows whose image exists are kept."""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"BUS-BRA not found at {root}. Download BUSBRA.zip from "
            f"https://zenodo.org/records/8231412 and unzip so that "
            f"{root}/Images and {root}/bus_data.csv exist.")

    meta = pd.read_csv(os.path.join(root, "bus_data.csv"))
    folds = pd.read_csv(os.path.join(root, "5-fold-cv.csv"))[["ID", "kFold"]]
    df = meta.merge(folds, on="ID", how="left")

    rows = []
    for _, r in df.iterrows():
        path = os.path.join(root, "Images", f"{r['ID']}.png")
        rows.append({
            "image_path": path,
            "case_id": str(r["ID"]),
            "patient_id": int(r["Case"]),          # group key -- prevents leakage
            "y": 1 if str(r["Pathology"]).strip().lower() == "malignant" else 0,
            "birads": str(r["BIRADS"]).strip(),
            "histology": str(r["Histology"]).strip().lower(),
            "device": str(r["Device"]).strip(),
            "fold": int(r["kFold"]) - 1 if pd.notna(r["kFold"]) else -1,
            "source": "busbra",
        })
    out = pd.DataFrame(rows)
    out = out[out["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return out


def patient_folds(df, n_folds=5, seed=0):
    """Fallback grouped folds if the official split is unavailable: stratified by
    patient-level label so no patient spans two folds."""
    from sklearn.model_selection import StratifiedKFold
    pat = df.groupby("patient_id")["y"].max().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pat["fold"] = -1
    for k, (_, te) in enumerate(skf.split(pat, pat["y"])):
        pat.loc[te, "fold"] = k
    return df["patient_id"].map(dict(zip(pat["patient_id"], pat["fold"]))).values


if __name__ == "__main__":
    d = load_busbra()
    print(f"BUS-BRA images: {len(d)} | patients: {d['patient_id'].nunique()}")
    print(f"malignant {int(d['y'].sum())} / benign {int((d['y'] == 0).sum())}")
    print("BI-RADS:", d["birads"].value_counts().sort_index().to_dict())
    print("devices:", d["device"].nunique())
    print("fold sizes:", np.bincount(d["fold"].values[d["fold"] >= 0]).tolist())

    # leakage guard: no patient may appear in more than one fold
    spans = d.groupby("patient_id")["fold"].nunique()
    print("patients spanning >1 fold:", int((spans > 1).sum()), "(must be 0)")
    mixed = d.groupby("patient_id")["y"].nunique()
    print("patients with mixed labels:", int((mixed > 1).sum()), "(must be 0)")
