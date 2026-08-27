"""
make_manifest_breast.py  (v2) -- build the manifest from raw BrEaST with the
correct three-field report: tissue / shape+margin / interpretation. The third
field is the multi-label 'Interpretation' column ('&'-separated), normalised to
match the vocabulary written by build_vocab.py.

    exec(open('make_manifest_breast.py').read())

Run order: build_vocab.py -> (Restart) -> this -> build_index -> run_phase_a -> evaluate
"""
import os, glob
import pandas as pd
import config
from vocab import FIELD_TEMPLATE, norm_interp

DATA_DIR   = "./dataset"
IMAGES_DIR = "./dataset/images"
CLINICAL_GLOB = "*clinical*"
TEST_FOLD, N_FOLDS = 0, 5


def load_table(path):
    for reader in (lambda p: pd.read_excel(p, engine="openpyxl"),
                   lambda p: pd.read_excel(p),
                   lambda p: pd.read_csv(p)):
        try:
            return reader(path)
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")


def col(df, name):
    lut = {c.lower().replace(" ", "_"): c for c in df.columns}
    return lut.get(name.lower().replace(" ", "_"))


def main():
    clinical = glob.glob(os.path.join(DATA_DIR, CLINICAL_GLOB))[0]
    df = load_table(clinical)

    c_img   = col(df, "Image_filename")
    c_tis   = col(df, "Tissue_composition")
    c_shape = col(df, "Shape")
    c_marg  = col(df, "Margin")
    c_int   = col(df, "Interpretation")
    c_bir   = col(df, "BIRADS")
    c_cls   = col(df, "Classification")

    rows = []
    for _, r in df.iterrows():
        tissue = str(r[c_tis]).lower().strip() if c_tis and pd.notna(r[c_tis]) else "not specified"
        shape_parts = [str(r[x]).lower().strip() for x in (c_shape, c_marg)
                       if x and pd.notna(r[x])]
        shape = ", ".join(shape_parts) if shape_parts else "not specified"

        if c_int and pd.notna(r[c_int]) and str(r[c_int]).strip():
            interps = [norm_interp(x) for x in str(r[c_int]).split("&") if x.strip()]
        else:
            interps = ["normal"]
        findings = ", ".join(interps)
        if c_bir and pd.notna(r[c_bir]):
            findings += f", bi-rads {str(r[c_bir]).strip().lower()}"

        rows.append({
            "image_path": os.path.join(IMAGES_DIR, str(r[c_img])),
            "report_text": FIELD_TEMPLATE.format(tissue=tissue, shape=shape, findings=findings),
            "dataset": "BrEaST",
            "_class": str(r[c_cls]) if c_cls else "unknown",
        })
    man = pd.DataFrame(rows)

    exists = man["image_path"].apply(os.path.exists)
    if (~exists).any():
        print(f"WARNING: {(~exists).sum()} images not found (dropped).")
    man = man[exists].reset_index(drop=True)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    man["split"] = "train"
    for fold, (_, test_idx) in enumerate(skf.split(man, man["_class"])):
        if fold == TEST_FOLD:
            man.loc[test_idx, "split"] = "test"
    man = man.drop(columns=["_class"])

    os.makedirs(os.path.dirname(config.MANIFEST_CSV) or ".", exist_ok=True)
    man.to_csv(config.MANIFEST_CSV, index=False)
    man[man["split"] == "test"].to_csv(config.TEST_MANIFEST, index=False)
    print("Split counts:\n", man["split"].value_counts().to_string())
    print(f"\nWrote {config.MANIFEST_CSV} ({len(man)}) and {config.TEST_MANIFEST} "
          f"({(man['split']=='test').sum()})")
    print("\nExample report:\n ", man.iloc[0]["report_text"])


if __name__ == "__main__":
    main()
