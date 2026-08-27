"""
config.py -- shared settings for the agentic BUS report-generation pipeline (Phase A).
Edit paths/flags here; every module imports from this file so settings stay consistent.
"""
import torch

# ---- paths --------------------------------------------------------------- #
MANIFEST_CSV   = "/content/bus_manifest.csv"          # train corpus (indexed)
TEST_MANIFEST  = "/content/bus_manifest_test.csv"     # eval set (image_path, report_text, dataset)
RETRIEVAL_DIR  = "/content/agentic_bus/retrieval"     # output of build_index.py
GEN_CKPT       = "/content/breast_report_baseline.pt" # your ViT+decoder generator checkpoint
CLF_CKPT       = "/content/interp_ldam_drw.pt"        # your LDAM-DRW interpretation classifier
PRED_OUT       = "/content/agentic_bus/predictions.csv"

# ---- models -------------------------------------------------------------- #
MODEL_ID   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEVICE     = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
MAX_LEN    = 48
SEED       = 42

# ---- agent behaviour ----------------------------------------------------- #
TOP_K          = 5       # retrieved exemplars
USE_CLASSIFIER = False   # True once CLF_CKPT is trained; else exemplar-parse fallback
DRAFT_MODE     = "template"  # "external" (your generator) | "template" (findings->report)
HIGH_CONF      = 0.60    # finding asserted at/above this must appear in the report
LOW_CONF       = 0.15    # report asserting a term below this is a contradiction
MAX_ATTEMPTS   = 2       # verify->refine loop cap
