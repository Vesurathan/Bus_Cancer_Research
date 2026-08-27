"""
finding_agent.py -- Phase A step 3. Produces structured findings {term: prob}
over the 28-term vocabulary. This is where YOUR imbalance contribution lives.

Two modes (config.USE_CLASSIFIER):
  True  -> load your LDAM-DRW + curriculum-margin multi-label classifier (CLF_CKPT).
  False -> fallback: parse the retrieved exemplar reports for vocabulary terms,
           weighting each hit by the neighbour's similarity. Lets the whole graph
           run before the classifier is trained; swap to True when ready.
"""
import re
import torch
import config
from vocab import TERMS, SYNONYMS


# --------------------------------------------------------------------------- #
# ADAPTER -- replace the body of load_classifier() with YOUR model definition.
# It must return an object exposing .predict_proba(image_path) -> {term: prob}.
# --------------------------------------------------------------------------- #
def load_classifier():
    """Load the trained LDAM-DRW interpretation classifier (train_finding_agent.py)."""
    from finding_model import Predictor
    return Predictor(config.CLF_CKPT, config.DEVICE)


def _match(term, text):
    text = text.lower()
    return any(re.search(rf"\b{re.escape(s)}\b", text) for s in SYNONYMS[term])


class FindingAgent:
    def __init__(self):
        self.clf = load_classifier() if config.USE_CLASSIFIER else None

    def _from_classifier(self, image_path):
        probs = self.clf.predict_proba(image_path)
        return {t: float(probs.get(t, 0.0)) for t in TERMS}

    def _from_exemplars(self, exemplars):
        """Similarity-weighted vote: a term's score is the max similarity among
        neighbours whose report mentions it, giving a pseudo-probability in [0,1]."""
        scores = {t: 0.0 for t in TERMS}
        wsum = sum(e["sim"] for e in exemplars if e.get("report_text")) or 1.0
        for e in exemplars:
            rep = e.get("report_text", "")
            if not rep:
                continue
            for t in TERMS:
                if _match(t, rep):
                    scores[t] = max(scores[t], e["sim"])
        # normalise so the strongest neighbour maps toward 1.0
        top = max(scores.values()) or 1.0
        return {t: min(1.0, v / top) for t, v in scores.items()}

    def predict(self, image_path, exemplars):
        if self.clf is not None:
            return self._from_classifier(image_path)
        return self._from_exemplars(exemplars)
