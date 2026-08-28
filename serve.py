"""
serve.py -- FastAPI wrapper around the agentic BUS pipeline for a live web demo.

Exposes the SAME logic as diagnose.py (multimodal fusion -> findings -> report
-> Grad-CAM -> review gate) but returns JSON instead of a PDF, so a hosted
frontend (e.g. on Vercel) can upload an image and render the result.

    uvicorn serve:app --host 0.0.0.0 --port 7860

Endpoints
    GET  /health   -> {"status": "ok", ...}   liveness + which artifacts loaded
    POST /predict  -> multipart image (+ optional descriptors JSON) -> result JSON

Only ./prod (ViT + kNN bank + calibrator) is REQUIRED. Findings, the generated
report and retrieval exemplars are best-effort: if their checkpoints/indexes are
absent the endpoint still returns the core decision and Grad-CAM.

NOT A MEDICAL DEVICE. Research prototype only.
"""
import base64
import io
import json
import os
import tempfile
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config

# diagnose.py holds the reusable pipeline helpers; importing it does not run main()
from diagnose import (
    bi_rads_estimate,
    get_findings,
    get_possible_findings,
    get_report,
    review_gate,
)
from malignancy_predictor import MalignancyPredictor

PROD_DIR = os.environ.get("PROD_DIR", "./prod")

app = FastAPI(title="Agentic BUS Reporting API", version="1.0")

# CORS: allow the deployed frontend. Set ALLOWED_ORIGINS to a comma-separated
# list (e.g. "https://your-app.vercel.app") in production; "*" is fine for a demo.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else
    [o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------- #
# Load the heavy artifacts once, at startup, and reuse across requests.
# ----------------------------------------------------------------------------- #
_predictor = None
_gradcam = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = MalignancyPredictor(prod_dir=PROD_DIR)
    return _predictor


def _get_gradcam():
    """Lazy Grad-CAM; returns None if the explainer cannot be built."""
    global _gradcam
    if _gradcam is None:
        try:
            from explain import GradCAMViT
            _gradcam = GradCAMViT(os.path.join(PROD_DIR, "vit_malignancy.pt"))
        except Exception as e:  # noqa: BLE001
            print(f"[gradcam unavailable: {e}]")
            _gradcam = False
    return _gradcam or None


def _b64_png(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


@app.on_event("startup")
def _warm():
    # Load the required predictor so the first request isn't slow / doesn't 500.
    try:
        _get_predictor()
        print("[startup] MalignancyPredictor loaded from", PROD_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"[startup] WARNING: predictor not loaded ({e}) -- "
              f"is {PROD_DIR} present with the trained artifacts?")


@app.get("/health")
def health():
    ok = os.path.isfile(os.path.join(PROD_DIR, "vit_malignancy.pt"))
    return {
        "status": "ok" if ok else "missing_artifacts",
        "prod_dir": PROD_DIR,
        "device": config.DEVICE,
        "has_report_generator": os.path.isfile(config.GEN_CKPT),
        "findings_mode": "retrieval" if not config.USE_CLASSIFIER else "classifier",
        "refine_mode": config.REFINE_MODE,
    }


def _parse_descriptors(raw):
    """Accept descriptors as a JSON object string; return dict or None."""
    if not raw:
        return None
    try:
        d = json.loads(raw)
        # keep only non-empty values
        d = {k: v for k, v in d.items() if v not in (None, "", [])}
        return d or None
    except (json.JSONDecodeError, TypeError):
        return None


@app.post("/predict")
async def predict(image: UploadFile = File(...), descriptors: str = Form(None)):
    predictor = _get_predictor()
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image upload")

    # persist to a temp file (the pipeline works on file paths)
    suffix = os.path.splitext(image.filename or "")[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()
    case_id = os.path.splitext(os.path.basename(image.filename or "upload"))[0]
    desc = _parse_descriptors(descriptors)
    heatmap_path = None

    try:
        # 1) DIAGNOSE ------------------------------------------------------- #
        dx = predictor.predict(tmp.name, desc)

        # 2) FINDINGS (best-effort) ---------------------------------------- #
        findings = get_findings(tmp.name)                 # [(term, prob), ...]
        findings_dict = {t: p for t, p in findings}
        possible = get_possible_findings(tmp.name, findings)

        # 3) REPORT (best-effort) ------------------------------------------ #
        report_text = get_report(tmp.name, findings_dict)

        # 4) RATIONALE ----------------------------------------------------- #
        try:
            from diagnosis_agent import DiagnosisAgent
            rationale = DiagnosisAgent(threshold=dx["threshold"]).diagnose(
                dx["streams"], p=dx["p_malignant"])["rationale"]
        except Exception:  # noqa: BLE001
            rationale = ""

        # 5) REVIEW GATE --------------------------------------------------- #
        flag, reason = review_gate(dx)

        # BI-RADS (real predictor if available, else estimate from probability)
        if dx.get("birads"):
            suffix_b = "" if dx.get("birads_from_descriptors") else " (from image)"
            bi_rads = f"{dx['birads']}{suffix_b}"
        else:
            bi_rads = bi_rads_estimate(dx["p_malignant"])

        # 6) GRAD-CAM overlay (best-effort) -------------------------------- #
        heatmap_b64 = None
        gc = _get_gradcam()
        if gc is not None:
            try:
                from explain import save_overlay
                _, cam = gc.heatmap(tmp.name)
                heatmap_path = tmp.name + "_cam.png"
                save_overlay(tmp.name, cam, heatmap_path)
                heatmap_b64 = _b64_png(heatmap_path)
            except Exception as e:  # noqa: BLE001
                print(f"[heatmap skipped: {e}]")

        result = {
            "id": case_id,
            "decision": dx["decision"],
            "pMalignant": round(dx["p_malignant"], 4),
            "confidence": round(dx["confidence"], 4),
            "biRads": bi_rads,
            "usedDescriptors": dx["used_descriptors"],
            "streams": {k: round(v, 4) for k, v in dx["streams"].items()},
            "oodDistance": round(dx["ood_distance"], 4),
            "threshold": round(dx["threshold"], 4),
            "neighbours": dx["neighbours"],
            "rationale": rationale,
            "review": {"flagged": flag, "reason": reason},
            "findings": [{"term": t, "prob": round(p, 3)} for t, p in findings],
            "possibleFindings": [{"term": t, "sim": round(s, 3)} for t, s in possible],
            "report": report_text,
            "heatmap": heatmap_b64,      # data URI or null
            "image": "data:image/png;base64," + base64.b64encode(raw).decode(),
            "disclaimer": "Research prototype. Not a medical device. Not for clinical use.",
        }
        return JSONResponse(result)

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
    finally:
        for p in (tmp.name, heatmap_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("serve:app", host="0.0.0.0", port=port)
