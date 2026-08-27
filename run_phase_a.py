"""
run_phase_a.py -- Phase A entry point. Builds the agents, compiles the graph,
runs every test image through it, and writes predictions for evaluate.py.

    exec(open('run_phase_a.py').read())

Set CONFIG_NAME below to reproduce ablation rows:
  "monolithic" -> draft only (no retrieval, no verifier)
  "rag"        -> retrieve + draft
  "full"       -> retrieve + draft + verify/refine (the full agent)
"""
import json
import pandas as pd
import config
from vocab import TERMS, SYNONYMS
from retrieval_tool import Retriever
from finding_agent import FindingAgent
from draft_agent import DraftAgent
from verifier import Verifier
from refiner import LLMRefiner
from graph import build_app

CONFIG_NAME = "full"
_ABLATION = {
    "monolithic": dict(use_retrieval=False, use_verifier=False),
    "rag":        dict(use_retrieval=True,  use_verifier=False),
    "full":       dict(use_retrieval=True,  use_verifier=True),
}


def gold_terms(report_text):
    """Extract the gold term set from a reference report for Finding-F1."""
    text = (report_text or "").lower()
    hits = []
    for t in TERMS:
        if any(s in text for s in SYNONYMS[t]):
            hits.append(t)
    return hits


def main():
    ablation = _ABLATION[CONFIG_NAME]
    retriever = Retriever() if ablation["use_retrieval"] else _Stub()
    finder    = FindingAgent()
    drafter   = DraftAgent()
    # Attach the LLM refiner only when the verifier is active and REFINE_MODE=="llm".
    refiner = (LLMRefiner()
               if ablation["use_verifier"] and config.REFINE_MODE == "llm"
               else None)
    verifier  = Verifier(refiner=refiner)
    app = build_app(retriever, finder, drafter, verifier, **ablation)

    test = pd.read_csv(config.TEST_MANIFEST)
    rows = []
    for _, r in test.iterrows():
        out = app.invoke({"image_path": r["image_path"]})
        rows.append({
            "image_path": r["image_path"],
            "dataset": r.get("dataset", ""),
            "gold_report": r["report_text"],
            "pred_report": out.get("report", ""),
            "findings_json": json.dumps(out.get("findings", {})),
            "gold_terms": json.dumps(gold_terms(r["report_text"])),
            "attempts": out.get("attempts", 0),
            "config": CONFIG_NAME,
        })
    pd.DataFrame(rows).to_csv(config.PRED_OUT, index=False)
    print(f"[{CONFIG_NAME}] wrote {len(rows)} predictions -> {config.PRED_OUT}")


class _Stub:
    """No-op retriever for the monolithic ablation (retrieval disabled)."""
    def retrieve(self, *a, **k):
        return []


if __name__ == "__main__":
    main()
