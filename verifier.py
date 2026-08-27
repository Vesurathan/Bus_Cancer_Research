"""
verifier.py -- consistency check between the draft report and the structured
findings, plus a repair. This turns the 'fluency vs clinical correctness'
divergence into a measurable gate.

Checks (unchanged from Phase A -- this is the objective gate):
  1. Coverage    -- every finding with prob >= HIGH_CONF must appear in the report.
  2. Contradiction -- a term asserted in the report whose finding prob <= LOW_CONF.
  3. Structure   -- the three template fields are present.

Repair (Phase B): if an LLM refiner is attached (config.REFINE_MODE == "llm"),
repair() asks it to rewrite the draft consistently while preserving fluent
phrasing; the deterministic Phase A repair remains as the fallback whenever the
LLM is unavailable or returns an unusable report. The graph's verify->refine
loop is unchanged.
"""
import re
import config
from vocab import TERMS, SYNONYMS
from draft_agent import compose_from_findings


def _match(term, text):
    text = text.lower()
    return any(re.search(rf"\b{re.escape(s)}\b", text) for s in SYNONYMS[term])


class Verifier:
    def __init__(self, refiner=None):
        self.refiner = refiner

    def check(self, report, findings):
        issues = []
        for t in TERMS:
            p = findings.get(t, 0.0)
            present = _match(t, report)
            if p >= config.HIGH_CONF and not present:
                issues.append(("missing", t, p))
            if present and p <= config.LOW_CONF:
                issues.append(("contradiction", t, p))
        for field_label in ["Tissue composition", "Shape and margin", "Findings"]:
            if field_label.lower() not in report.lower():
                issues.append(("structure", field_label, 0.0))
        return issues

    def repair(self, report, findings, issues):
        """LLM refine if a refiner is attached, else deterministic repair. The
        rule-based path is also the fallback when the LLM returns None."""
        if self.refiner is not None:
            refined = self.refiner.refine(report, findings, issues)
            if refined:
                return refined
        return self._rule_repair(report, findings, issues)

    def _rule_repair(self, report, findings, issues):
        """Deterministic repair: rebuild from findings if structure is broken,
        else append missing findings and remove contradicted surface terms."""
        if any(i[0] == "structure" for i in issues):
            return compose_from_findings(findings, config.HIGH_CONF)
        fixed = report
        missing = [t for k, t, _ in issues if k == "missing"]
        if missing:
            add = ", ".join(SYNONYMS[t][0] for t in missing)
            fixed = fixed.rstrip(". ") + f". Additional findings: {add}."
        for k, t, _ in issues:
            if k == "contradiction":
                for s in SYNONYMS[t]:
                    fixed = re.sub(rf"\b{re.escape(s)}\b", "", fixed, flags=re.I)
        return re.sub(r"\s{2,}", " ", fixed).strip()

    def verify(self, report, findings):
        issues = self.check(report, findings)
        return (len(issues) == 0), issues
