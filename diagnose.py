"""
diagnose.py -- the end-to-end agentic flow and product entry point.

    python diagnose.py path/to/image.png [--out report.pdf]
                       [--desc Shape=irregular Margin="not circumscribed - spiculated" age=54]

Pipeline (one image, optional BI-RADS descriptors):
  1. DIAGNOSE  -- multimodal fusion -> benign/malignant + P(malignant) + confidence
                 (uses image + descriptors when given, image-only otherwise)
  2. FIND      -- long-tail interpretation classifier -> structured findings
  3. DRAFT     -- ViT+decoder generator -> report text
  4. VERIFY/REFINE (optional) -- LLM consistency refine if Ollama is up
  5. REVIEW GATE -- flag low-confidence / out-of-distribution / conflicting cases
  6. REPORT    -- render a one-page PDF

Requires production artifacts from train_production.py (./prod). Findings and
report generation reuse the existing CLF_CKPT / GEN_CKPT checkpoints.
"""
import os, sys, json, argparse

import config
from malignancy_predictor import MalignancyPredictor

OOD_FLAG = 0.15        # kNN distance above this = looks out-of-distribution
LOWCONF_FLAG = 0.30    # fused confidence below this = uncertain


def bi_rads_estimate(p):
    if p < 0.10:  return "2 (benign, est.)"
    if p < 0.40:  return "3 (probably benign, est.)"
    if p < 0.70:  return "4 (suspicious, est.)"
    return "5 (highly suspicious, est.)"


def get_findings(image_path, top=5):
    """Confidently ASSERTED interpretation terms from the findings classifier.
    High-precision (the classifier is reliable on the common terms)."""
    try:
        from finding_model import Predictor
        if not os.path.exists(config.CLF_CKPT):
            return []
        probs = Predictor(config.CLF_CKPT, config.DEVICE).predict_proba(image_path)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return [(t, p) for t, p in ranked[:top] if p >= 0.30]
    except Exception as e:
        print(f"[findings skipped: {e}]")
        return []


def get_possible_findings(image_path, asserted, top=4, min_sim=0.5):
    """POSSIBLE findings surfaced from the most similar train cases (retrieval).
    Lower precision -> shown separately as context, not asserted. Helps catch
    rarer terms the classifier alone misses."""
    try:
        from retrieval_tool import Retriever
        from vocab import TERMS, SYNONYMS
        have = {t for t, _ in asserted}
        scores = {}
        for e in Retriever().retrieve(image_path):
            rep = (e.get("report_text") or "").lower()
            sim = float(e.get("sim", 0.0))
            if sim < min_sim:
                continue
            for t in TERMS:
                if t not in have and any(s in rep for s in SYNONYMS[t]):
                    scores[t] = max(scores.get(t, 0.0), sim)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    except Exception as e:
        print(f"[possible-findings skipped: {e}]")
        return []


def get_report(image_path, findings_dict):
    """Generate the report text via the drafter, optionally LLM-refined."""
    try:
        from draft_agent import DraftAgent
        text = DraftAgent().draft(image_path, findings_dict)
    except Exception as e:
        print(f"[draft skipped: {e}]")
        return ""
    if config.REFINE_MODE == "llm":
        try:
            from verifier import Verifier
            from refiner import LLMRefiner
            v = Verifier(refiner=LLMRefiner())
            issues = v.check(text, findings_dict)
            if issues:
                fixed = v.repair(text, findings_dict, issues)
                if fixed:
                    text = fixed
        except Exception:
            pass
    return text


def review_gate(dx):
    reasons = []
    if dx["confidence"] < LOWCONF_FLAG:
        reasons.append("low decision confidence")
    if dx["ood_distance"] > OOD_FLAG:
        reasons.append("image looks out-of-distribution vs training data")
    s = dx["streams"]
    if "descriptor" in s and abs(
            (s["vit"] + s["knn"]) / 2 - s["descriptor"]) > 0.4:
        reasons.append("image and descriptors disagree")
    return (len(reasons) > 0, "; ".join(reasons))


def parse_descriptors(items):
    if not items:
        return None
    d = {}
    for kv in items:
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.strip()] = v.strip()
    return d or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--desc", nargs="*", default=None,
                    help='optional descriptors, e.g. Shape=irregular Margin=spiculated age=54')
    args = ap.parse_args()

    image = args.image
    out = args.out or f"./report_{os.path.splitext(os.path.basename(image))[0]}.pdf"
    descriptors = parse_descriptors(args.desc)
    case_id = os.path.splitext(os.path.basename(image))[0]

    print(f"[1/6] diagnosing {case_id} "
          f"({'image + descriptors' if descriptors else 'image only'}) ...")
    dx = MalignancyPredictor().predict(image, descriptors)
    print(f"      -> {dx['decision'].upper()}  P(malignant)={dx['p_malignant']:.2f}  "
          f"conf={dx['confidence']*100:.0f}%  OOD={dx['ood_distance']:.2f}")

    print("[2/6] findings ...")
    findings = get_findings(image)
    findings_dict = {t: p for t, p in findings}
    possible = get_possible_findings(image, findings)

    print("[3/6] drafting report ...")
    report_text = get_report(image, findings_dict)

    flag, reason = review_gate(dx)
    if flag:
        print(f"[5/6] REVIEW FLAG: {reason}")

    from diagnosis_agent import DiagnosisAgent
    # use the production fused probability so the rationale matches the decision
    rationale = DiagnosisAgent(threshold=dx["threshold"]).diagnose(
        dx["streams"], p=dx["p_malignant"])["rationale"]

    # Grad-CAM lesion localisation (where the ViT looked)
    heatmap_path = None
    try:
        from explain import GradCAMViT, save_overlay
        gc = GradCAMViT(f"./prod/vit_malignancy.pt")
        _, cam = gc.heatmap(image)
        heatmap_path = f"./_cam_{case_id}.png"
        save_overlay(image, cam, heatmap_path)
    except Exception as e:
        print(f"[heatmap skipped: {e}]")

    print("[6/6] rendering PDF ...")
    from report_pdf import render_report
    # real BI-RADS predictor (birads_model.py) with graceful fallback
    if dx.get("birads"):
        suffix = "" if dx.get("birads_from_descriptors") else " (from image)"
        bi_rads = f"{dx['birads']}{suffix}"
    else:
        bi_rads = bi_rads_estimate(dx["p_malignant"])
    render_report(
        out, case_id=case_id, image_path=image, diagnosis=dx,
        findings=findings, report_text=report_text,
        bi_rads=bi_rads, possible_findings=possible,
        rationale=rationale, review_flag=flag, review_reason=reason,
        heatmap_path=heatmap_path,
    )
    if heatmap_path and os.path.exists(heatmap_path):
        try:
            os.remove(heatmap_path)
        except OSError:
            pass
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
