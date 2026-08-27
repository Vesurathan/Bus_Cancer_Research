"""
refiner.py -- Phase B. An LLM-backed repair step for the verify->refine loop.

The Phase A verifier flagged coverage / contradiction / structure issues and
fixed them with deterministic string surgery ("Additional findings: ..."), which
kept the report clinically consistent but read mechanically and *lowered* the NLG
metrics (see README's full-row result). The LLM refiner instead rewrites the
draft so it (a) asserts every high-confidence finding, (b) drops every
contradicted term, and (c) keeps the three-field structure -- while preserving
fluent, reference-like phrasing.

Backend: a local Ollama server (default llama3.1:8b), so the whole pipeline stays
on-device and reproducible. Any failure (server down, timeout, empty/garbled
output) falls back to the caller's rule-based repair, so the graph never breaks.
"""
import json
import urllib.request
import urllib.error

import config
from vocab import TERMS, SYNONYMS


def _surface(term):
    return SYNONYMS[term][0]


class LLMRefiner:
    """Rewrites a draft report to be consistent with the structured findings,
    via a local Ollama model. Exposes .refine(report, findings, issues) -> str
    or None (None signals the caller to use its rule-based fallback)."""

    def __init__(self, url=None, model=None, timeout=None, num_predict=None):
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.num_predict = num_predict or config.OLLAMA_NUM_PREDICT

    # -- prompt ------------------------------------------------------------- #
    def _prompt(self, report, findings, issues):
        must_include = [_surface(t) for k, t, _ in issues if k == "missing"]
        must_exclude = [_surface(t) for k, t, _ in issues if k == "contradiction"]
        structure_broken = any(k == "structure" for k, _, _ in issues)

        # The authoritative finding set (independent of which issues fired), so
        # the model has the full clinical picture, not just the deltas.
        asserted = [_surface(t) for t in TERMS
                    if findings.get(t, 0.0) >= config.HIGH_CONF]

        lines = [
            "You are a radiology report editor for breast ultrasound (BUS) reports.",
            "Rewrite the DRAFT so it is clinically consistent with the FINDINGS, "
            "while keeping it fluent, concise, and in the exact three-field format:",
            "Tissue composition: <...>. Shape and margin: <...>. Findings: <...>.",
            "",
            f"DRAFT: {report}",
            "",
            f"Confirmed findings that MUST appear in the Findings field: "
            f"{', '.join(asserted) if asserted else '(none)'}.",
        ]
        if must_include:
            lines.append(f"Specifically add these missing findings: {', '.join(must_include)}.")
        if must_exclude:
            lines.append(f"Remove these unsupported terms entirely: {', '.join(must_exclude)}.")
        if structure_broken:
            lines.append("Restore any missing field so all three fields are present.")
        lines += [
            "",
            "Rules: keep the wording natural (do not append a raw 'Additional "
            "findings:' list); preserve any BI-RADS category present in the draft; "
            "do not invent findings beyond those listed. "
            "Output ONLY the corrected report on a single line, with no preamble, "
            "quotes, or explanation.",
        ]
        return "\n".join(lines)

    # -- backend ------------------------------------------------------------ #
    def _call_ollama(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.num_predict,
                "seed": config.SEED,
            },
        }
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")

    # -- post-processing ---------------------------------------------------- #
    @staticmethod
    def _clean(text):
        """Reduce the model output to a single report line. Returns "" if it
        doesn't look like a report (so the caller falls back)."""
        text = (text or "").strip()
        if not text:
            return ""
        # Prefer the line that carries the report structure.
        for line in text.splitlines():
            if "tissue composition" in line.lower():
                text = line.strip()
                break
        else:
            text = text.splitlines()[0].strip()
        text = text.strip().strip('"').strip()
        if "findings" not in text.lower():
            return ""            # not a usable three-field report
        return text

    def refine(self, report, findings, issues):
        try:
            raw = self._call_ollama(self._prompt(report, findings, issues))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None          # server/parse failure -> caller uses rule repair
        cleaned = self._clean(raw)
        return cleaned or None
