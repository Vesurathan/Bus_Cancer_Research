# Agentic Multi-Evidence Breast-Ultrasound Malignancy Classifier

An agentic, multi-evidence pipeline that classifies breast-ultrasound (BUS) lesions
as **benign or malignant**, calibrates its confidence, explains its decision, flags
uncertain cases for human review, and generates a one-page structured report.
Developed as an MSc dissertation project (University of East London).

> **Research prototype — not a medical device.** Trained and evaluated
> retrospectively on public datasets; not validated for clinical use.

## Key results

| Setting | Metric | Result |
|---|---|---|
| Internal (BrEaST, leak-free 5-fold CV) | Fusion AUC | **0.932** |
| External — source-held-out BUSI | AUC | 0.945 |
| External — unseen-source BUS-BRA | AUC | 0.865 |
| Calibration | ECE (before → after Platt) | 0.088 → 0.030 |
| Descriptor-free enhancement (autonomous, predicted masks, Dice 0.90) | BUS-BRA AUC | 0.809 → 0.839 |
| Report refinement (factual) | BI-RADS stated / findings precision | 0% → 58% / 0.29 → 0.46 |

Fusion vs descriptor stream is *not* statistically significant (DeLong p = 0.31);
fusion vs the image stream is (p < 0.001). See the dissertation for full analysis.

## System overview

A `retrieve → diagnose → find → draft → verify → refine → report` workflow:

- **Retrieve** — BiomedCLIP embedding + FAISS k-NN over prior cases (vote + OOD score)
- **Diagnose** — cross-fitted stacked fusion of a fine-tuned ViT, the k-NN vote and a
  descriptor / automated-morphology model, with Platt calibration
- **Find** — 28-term long-tailed findings classifier (LDAM-DRW)
- **Draft / Verify / Refine** — ViT encoder + Transformer decoder, checked against the
  findings and refined by a local LLM (Llama 3.1 via Ollama)
- **Review gate** — flags low-confidence / out-of-distribution / disagreeing cases
- **Enhancement** — a U-Net segmenter enables ROI-cropping + automated morphology,
  reducing dependence on radiologist descriptors

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional (report refinement): install Ollama and pull the model
#   brew install ollama && ollama pull llama3.1:8b
```

Runs on CPU, CUDA or Apple-silicon MPS (auto-selected in `config.py`).

## Data (not included — download separately)

The datasets are **not** in this repo (size and licence restrictions). Download each
from its source and place it as shown; then run the manifest/vocab scripts.

| Dataset | Source | Expected path |
|---|---|---|
| BrEaST-Lesions-USG | Pawłowska et al. 2024 (Zenodo/TCIA) | `dataset/images/`, `dataset/*clinical*.xlsx` |
| BUSI | Al-Dhabyani et al. 2020 | `busi/{benign,malignant,normal}/` |
| BUS-BRA | Gómez-Flores et al. 2024 (Zenodo 8231412, CC-BY-4.0) | `busbra/BUSBRA/{Images,Masks}/`, `bus_data.csv` |

Respect each dataset's licence; **do not redistribute** the images.

## Running

```bash
python malignancy_data.py          # sanity-check BrEaST loading
python evaluate_malignancy.py      # leak-free 5-fold CV of the fusion (internal)
python evaluate_external.py        # external validation on BUSI / BUS-BRA
python train_production.py         # train + save the deployable model to prod/
python diagnose.py <image.png>     # run the full agentic pipeline on one image
```

Enhancement / analysis experiments: `evaluate_morphology.py`, `evaluate_roi_vit.py`,
`external_roi_multiseed.py`, `train_segmenter_v2.py`, `predicted_mask_eval_v2.py`,
`delong_test.py`, `report_factual_consistency.py`.

## Repository layout (key modules)

```
malignancy_data.py / malignancy_model.py   data + ViT image stream
evaluate_malignancy.py                      leak-free fusion CV
retrieval_tool.py                           BiomedCLIP + FAISS retrieval
morphology_features.py                      automated lesion descriptors
train_segmenter*.py                         U-Net lesion segmenter
diagnosis_agent.py / diagnose.py            agentic orchestration
finding_agent.py / generator_model.py       findings + report generation
report_pdf.py / explain.py                  PDF output + Grad-CAM
config.py                                   paths, device, hyperparameters
```

## Licence

Source code is released under the **MIT Licence** (see `LICENSE`). The datasets
are **not** included and remain under their own licences — obtain them from the
original sources and comply with those terms.

## Acknowledgements / citation

If you use this code, please cite the dissertation and the underlying datasets
(BrEaST, BUSI, BUS-BRA) and models (ViT, BiomedCLIP). See the dissertation's
reference list for full details.
