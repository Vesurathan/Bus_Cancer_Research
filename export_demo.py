"""
export_demo.py -- run the REAL agentic pipeline once, capturing every node's
output per case, and dump web/public/demo_data.json (+ the images) for the
Next.js replay UI. Nothing is faked: findings come from the trained classifier,
exemplars from BiomedCLIP+FAISS, the draft from the generator, the issues from
the verifier, the refined report from the LLM. The UI just replays this.

    python export_demo.py            # ~16 demo cases from the leak-free test split

Prereqs (run once, cheap): make_manifest_breast.py + build_index.py so the split
and FAISS index are consistent, single-split checkpoints present (CLF_CKPT,
GEN_CKPT), Ollama up for the refine step (falls back to rule repair if not).
"""
import os, re, json, shutil
import pandas as pd

import config
from vocab import TERMS, TIER
from retrieval_tool import Retriever
from finding_agent import FindingAgent
from draft_agent import DraftAgent
from verifier import Verifier
from refiner import LLMRefiner
from graph import build_app

N_CASES = 16
OUT_DIR = "./web/public"
CASES_DIR = os.path.join(OUT_DIR, "cases")
DATA_JSON = os.path.join(OUT_DIR, "demo_data.json")


def bi_rads(text):
    m = re.search(r"bi[\s-]*rads\s*([0-9][a-c]?)", str(text).lower())
    return m.group(1) if m else None


def gold_term_set(text):
    from vocab import SYNONYMS
    t = (text or "").lower()
    return [term for term in TERMS if any(s in t for s in SYNONYMS[term])]


def copy_image(src, dst_name):
    dst = os.path.join(CASES_DIR, dst_name)
    try:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
        return f"cases/{dst_name}" if os.path.exists(dst) else None
    except OSError:
        return None


def run_case(app, retriever, image_path):
    """Stream the graph and pull out each node's contribution in order."""
    draft_report, refined_report, issues, findings, exemplars = None, None, [], {}, []
    final = {}
    for update in app.stream({"image_path": image_path}):
        for node, state in update.items():
            if node == "retrieve":
                exemplars = state.get("exemplars", [])
            elif node == "find":
                findings = state.get("findings", {})
            elif node == "draft":
                if draft_report is None:
                    draft_report = state.get("report")   # first draft (pre-refine)
                final = state
            elif node == "verify":
                if not issues:
                    issues = state.get("issues", [])     # issues from the first check
                final = {**final, **state}
            elif node == "refine":
                refined_report = state.get("report")     # LLM-rewritten report
                final = {**final, **state}
    return draft_report, refined_report, issues, findings, exemplars, final


def main():
    os.makedirs(CASES_DIR, exist_ok=True)
    config.DRAFT_MODE = "external"
    config.USE_CLASSIFIER = True

    retriever = Retriever()
    finder = FindingAgent()
    drafter = DraftAgent()
    refiner = LLMRefiner() if config.REFINE_MODE == "llm" else None
    verifier = Verifier(refiner=refiner)
    app = build_app(retriever, finder, drafter, verifier,
                    use_retrieval=True, use_verifier=True)

    test = pd.read_csv(config.TEST_MANIFEST)
    # Prefer variety: keep a spread of BI-RADS categories; cap at N_CASES.
    test = test.drop_duplicates("image_path").reset_index(drop=True)
    picks = test.head(N_CASES)

    cases = []
    for i, r in picks.iterrows():
        img = r["image_path"]
        cid = os.path.splitext(os.path.basename(img))[0]
        print(f"[{i+1}/{len(picks)}] {cid} ...", flush=True)
        draft, refined, issues, findings, exemplars, final = run_case(app, retriever, img)

        findings_sorted = sorted(
            ({"term": t, "prob": round(float(findings.get(t, 0.0)), 4),
              "tier": TIER[t]} for t in TERMS),
            key=lambda d: -d["prob"])
        ex_out = []
        for k, e in enumerate(exemplars[:3]):
            ex_img = copy_image(e.get("image_path", ""), f"{cid}_ex{k}.png")
            ex_out.append({"image": ex_img,
                           "report": e.get("report_text", ""),
                           "sim": round(float(e.get("sim", 0.0)), 3)})

        gold_report = r["report_text"]
        cases.append({
            "id": cid,
            "image": copy_image(img, f"{cid}.png"),
            "biRads": bi_rads(gold_report),
            "goldReport": gold_report,
            "goldTerms": gold_term_set(gold_report),
            "steps": {
                "retrieve": {"exemplars": ex_out},
                "find": {"findings": findings_sorted},
                "draft": {"report": draft or ""},
                "verify": {
                    "verified": bool(final.get("verified", False)) if draft == final.get("report") else False,
                    "issues": [{"kind": k_, "term": t_} for (k_, t_, _p) in issues],
                },
                "refine": {"report": refined or "", "applied": refined is not None},
            },
            "final": {
                "report": final.get("report", ""),
                "attempts": int(final.get("attempts", 1)),
                "verified": bool(final.get("verified", False)),
            },
        })

    payload = {
        "meta": {
            "dataset": "BrEaST-Lesions-USG",
            "config": "full agent (retrieve + find + draft + verify + LLM refine)",
            "model": config.OLLAMA_MODEL if refiner else "rule-based repair",
            "nTerms": len(TERMS),
        },
        "tiers": {tier: [t for t in TERMS if TIER[t] == tier]
                  for tier in ("head", "medium", "tail")},
        "cases": cases,
    }
    with open(DATA_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {DATA_JSON} with {len(cases)} cases; images in {CASES_DIR}")


if __name__ == "__main__":
    main()
