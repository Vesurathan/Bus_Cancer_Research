"""
draft_agent.py -- Phase A step 4. Produces the draft report.

Two modes (config.DRAFT_MODE):
  "external" -> your trained ViT + Transformer-decoder generator (GEN_CKPT).
  "template" -> compose a report from the findings using the three-field
                template, so the graph runs before the generator is wired.

The "external" path is the one that carries your generation contribution; the
"template" path is a scaffold and a rule-based baseline to ablate against.
"""
import config
from vocab import TERMS, FIELD_OF, FIELD_TEMPLATE, SYNONYMS


def load_generator():
    """Load the trained ViT+Transformer-decoder generator (train_generator.py)."""
    from generator_model import Generator
    return Generator(config.GEN_CKPT, config.DEVICE)


def _surface(term):
    return SYNONYMS[term][0]


def compose_from_findings(findings, thresh):
    """Rule-based report from asserted terms (prob >= thresh), grouped by field."""
    asserted = [t for t in TERMS if findings.get(t, 0.0) >= thresh]
    buckets = {"tissue": [], "shape": [], "findings": []}
    for t in asserted:
        buckets[FIELD_OF[t]].append(_surface(t))
    def phrase(items, default):
        return ", ".join(items) if items else default
    return FIELD_TEMPLATE.format(
        tissue=phrase(buckets["tissue"], "not specified"),
        shape=phrase(buckets["shape"], "not specified"),
        findings=phrase(buckets["findings"], "no significant findings"),
    )


class DraftAgent:
    def __init__(self):
        self.gen = load_generator() if config.DRAFT_MODE == "external" else None

    def draft(self, image_path, findings):
        if self.gen is not None:
            return self.gen.generate(image_path)
        return compose_from_findings(findings, config.HIGH_CONF)
