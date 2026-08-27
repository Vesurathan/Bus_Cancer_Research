"""
diagnosis_agent.py -- the agentic malignant-vs-benign decision node.

It gathers three independent evidence streams for one case, fuses them into a
calibrated malignancy probability, and returns a decision plus a plain-language
rationale that names which evidence drove it (and flags disagreement between the
image and the descriptors -- the case a clinician should look at twice).

  vit        -- P(malignant | image), the learned visual stream (malignancy_model)
  knn        -- similarity-weighted vote of retrieved neighbours' known labels
  descriptor -- P(malignant | BI-RADS descriptors), the radiologist's read

Fusion is a small logistic meta-model (the same form cross-fitted in
evaluate_malignancy.fuse_stack); pass its fitted weights, or fall back to a mean.
The agent is evidence-source agnostic: give it whatever streams are available and
it fuses what it has.
"""
import numpy as np


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


class DiagnosisAgent:
    def __init__(self, meta=None, threshold=0.5, stream_order=("descriptor", "knn", "vit")):
        # meta: fitted sklearn LogisticRegression over [streams in stream_order];
        # None -> mean fusion. threshold: operating point for the decision.
        self.meta = meta
        self.threshold = threshold
        self.stream_order = stream_order

    def _fuse(self, streams):
        present = [s for s in self.stream_order if streams.get(s) is not None]
        if self.meta is not None and len(present) == len(self.stream_order):
            x = np.array([[streams[s] for s in self.stream_order]])
            return float(self.meta.predict_proba(x)[0, 1]), present
        # fallback: mean in logit space over available streams (robust to scale)
        vals = [streams[s] for s in present]
        return float(1 / (1 + np.exp(-np.mean([_logit(v) for v in vals])))), present

    def diagnose(self, streams, p=None):
        """streams: {"vit": p, "knn": p, "descriptor": p} (any may be None).
        p: use this fused probability (e.g. from the production model) so the
        rationale stays consistent with the reported decision; if None, fuse here.
        Returns the fused decision, probability, and a rationale."""
        used = [s for s in self.stream_order if streams.get(s) is not None]
        if p is None:
            p, used = self._fuse(streams)
        decision = "malignant" if p >= self.threshold else "benign"
        t = self.threshold
        conf = (p - t) / max(1e-6, 1 - t) if p >= t else (t - p) / max(1e-6, t)
        conf = max(0.0, min(1.0, conf))  # decisiveness relative to the threshold

        # rationale: rank the streams by how strongly each points to the decision.
        # Only treat a stream as "disagreeing" if it is meaningfully on the other
        # side (near-0.5 streams are neutral, not conflicting).
        contribs = []
        for s in used:
            v = streams[s]
            agrees = (v >= 0.5) == (p >= 0.5)
            contribs.append((s, v, agrees))
        contribs.sort(key=lambda c: -abs(c[1] - 0.5))
        lead = contribs[0] if contribs else None

        disagree = [s for s, v, a in contribs if not a and abs(v - 0.5) >= 0.15]
        parts = [f"Fused P(malignant) = {p:.2f} → {decision} "
                 f"({'high' if conf > 0.6 else 'moderate' if conf > 0.25 else 'low'} confidence)."]
        if lead:
            parts.append(f"Strongest evidence: {lead[0]} ({lead[1]:.2f}).")
        if disagree:
            parts.append("⚠ streams disagree — "
                         + ", ".join(f"{s} ({streams[s]:.2f})" for s in disagree)
                         + " point the other way; review recommended.")
        return {
            "p_malignant": p,
            "decision": decision,
            "confidence": conf,
            "streams": {s: (None if streams.get(s) is None else float(streams[s]))
                        for s in self.stream_order},
            "rationale": " ".join(parts),
        }
