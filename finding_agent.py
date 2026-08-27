"""
finding_agent.py -- produces structured findings {term: score} over the 28-term
vocabulary.

Primary mode is GROUNDED RETRIEVAL: a term's score is the similarity-weighted
fraction of the top-k nearest neighbours whose report mentions it. An ablation
showed this doubles the trained LDAM-DRW classifier's macro-F1 (0.197 vs 0.093)
and is the only method to score above zero on rare (medium/tail) terms, because it
recalls a similar past case rather than trying to learn an unlearnable class from
one or two examples. It also needs no training and is interpretable.

Set config.USE_CLASSIFIER = True only to fall back to the (weaker) LDAM classifier.
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

    def _from_exemplars(self, exemplars, k=5):
        """Grounded retrieval vote: a term's score is the similarity-weighted
        fraction of the top-k neighbours whose report mentions it (in [0,1]).
        k=5 with a low assertion threshold was best in the ablation, because it
        lets a single close neighbour surface a rare finding."""
        ex = sorted((e for e in exemplars if e.get("report_text")),
                    key=lambda e: -e.get("sim", 0.0))[:k]
        wsum = sum(e.get("sim", 0.0) for e in ex) or 1.0
        scores = {t: 0.0 for t in TERMS}
        for e in ex:
            rep = e["report_text"]
            for t in TERMS:
                if _match(t, rep):
                    scores[t] += e.get("sim", 0.0)
        return {t: v / wsum for t, v in scores.items()}

    def predict(self, image_path, exemplars):
        if self.clf is not None:
            return self._from_classifier(image_path)
        return self._from_exemplars(exemplars)
