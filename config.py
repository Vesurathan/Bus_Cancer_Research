"""
config.py -- shared settings for the agentic BUS report-generation pipeline (Phase A).
Edit paths/flags here; every module imports from this file so settings stay consistent.
"""
import os
import torch

# ---- paths --------------------------------------------------------------- #
MANIFEST_CSV   = "./bus_manifest.csv"          # train corpus (indexed)
TEST_MANIFEST  = "./bus_manifest_test.csv"     # eval set (image_path, report_text, dataset)
RETRIEVAL_DIR  = "./retrieval"                 # output of build_index.py
GEN_CKPT       = "./breast_report_baseline.pt" # your ViT+decoder generator checkpoint
CLF_CKPT       = "./interp_ldam_drw.pt"        # your LDAM-DRW interpretation classifier
PRED_OUT       = "./predictions.csv"

# ---- models -------------------------------------------------------------- #
MODEL_ID   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEVICE     = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
MAX_LEN    = 48
SEED       = 42

# ---- agent behaviour ----------------------------------------------------- #
TOP_K          = 5       # retrieved exemplars
USE_CLASSIFIER = False  # retrieval-based findings (ablation showed the trained
                        # LDAM classifier does not beat grounded retrieval)
DRAFT_MODE     = "external"  # "external" (your generator) | "template" (findings->report)
HIGH_CONF      = 0.20    # finding asserted at/above this must appear in the report
                         # (tuned to the retrieval-vote score distribution)
LOW_CONF       = 0.05    # report asserting a term below this is a contradiction
MAX_ATTEMPTS   = 2       # verify->refine loop cap

# ---- Phase B: refiner ---------------------------------------------------- #
# The verify->refine loop's repair step. "llm" uses a local Ollama model to
# rewrite the draft so it is clinically consistent while staying fluent;
# "rule" reproduces the Phase A deterministic repair (and is the automatic
# fallback whenever the LLM is unreachable or returns an unusable report).
REFINE_MODE    = os.environ.get("REFINE_MODE", "llm")   # "llm" | "rule" (env override for hosted/CPU deploys with no Ollama)
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_MODEL   = "llama3.1:8b"
OLLAMA_TIMEOUT = 60                          # seconds per refine call
OLLAMA_NUM_PREDICT = 160                     # max tokens for the rewritten report
