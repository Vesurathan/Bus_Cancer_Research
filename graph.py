"""
graph.py -- Phase A step 6. LangGraph state machine wiring the agents:

    retrieve -> find -> draft -> verify --(pass)--> END
                                   ^                  |
                                   |               (fail, attempts<MAX)
                                   +----- refine <----+

'refine' is the rule-based repair in Phase A; swap it for an LLM refiner in
Phase B and this control flow is unchanged. Toggle nodes on/off to reproduce the
Table 1 ablation (monolithic = draft only; +RAG = retrieve+draft; full = all).
"""
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import config


class ReportState(TypedDict, total=False):
    image_path: str
    exemplars: List[Dict]
    findings: Dict[str, float]
    report: str
    issues: List
    attempts: int
    verified: bool


def build_app(retriever, finder, drafter, verifier,
              use_retrieval=True, use_verifier=True):
    g = StateGraph(ReportState)

    def retrieve_node(s):
        return {"exemplars": retriever.retrieve(s["image_path"]) if use_retrieval else []}

    def find_node(s):
        return {"findings": finder.predict(s["image_path"], s.get("exemplars", []))}

    def draft_node(s):
        r = drafter.draft(s["image_path"], s.get("findings", {}))
        return {"report": r, "attempts": s.get("attempts", 0) + 1}

    def verify_node(s):
        if not use_verifier:
            return {"verified": True, "issues": []}
        ok, issues = verifier.verify(s["report"], s["findings"])
        return {"verified": ok, "issues": issues}

    def refine_node(s):
        # Count each refine against MAX_ATTEMPTS so the verify->refine loop always
        # terminates. (Phase A's rule repair converged in one pass and hid this;
        # an LLM rewrite may not exactly satisfy the string-match gate, which would
        # otherwise loop forever since attempts was only bumped in draft_node.)
        fixed = verifier.repair(s["report"], s["findings"], s["issues"])
        return {"report": fixed, "attempts": s.get("attempts", 0) + 1}

    g.add_node("retrieve", retrieve_node)
    g.add_node("find", find_node)
    g.add_node("draft", draft_node)
    g.add_node("verify", verify_node)
    g.add_node("refine", refine_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "find")
    g.add_edge("find", "draft")
    g.add_edge("draft", "verify")

    def route(s):
        if s.get("verified") or s.get("attempts", 0) >= config.MAX_ATTEMPTS:
            return END
        return "refine"

    g.add_conditional_edges("verify", route, {"refine": "refine", END: END})
    g.add_edge("refine", "verify")   # re-verify after repair (loop)
    return g.compile()
