# BUS Agent — demo frontend

A Next.js app that **replays the real agentic pipeline** on breast-ultrasound
cases and animates each step: retrieve → find → draft → verify → refine.

Nothing here is faked. The data in `public/demo_data.json` is captured from an
actual run of the Python pipeline (`../export_demo.py`): findings from the trained
LDAM-DRW classifier, exemplars from BiomedCLIP + FAISS, the draft from the ViT +
decoder generator, issues from the verifier, and the refined report from the
local Ollama LLM. The UI just replays it — no models/GPU/Ollama needed at view
time. It is a **research demo, not a clinical device**.

## Run
```
# 1) (from the project root) capture fresh demo data — needs the models + Ollama
python export_demo.py                 # writes web/public/demo_data.json + images

# 2) start the UI
cd web
npm install
npm run dev                           # http://localhost:3100
```
If `public/demo_data.json` already exists you can skip step 1 and just run the UI.

## What it shows
- **Pipeline flow** — the five agent nodes light up as the agent runs.
- **Retrieve** — top-3 nearest train cases with cosine similarity.
- **Find** — per-term confidence bars grouped by head/medium/tail tier, with the
  0.60 assertion gate and ground-truth markers (the long-tail story, visualised).
- **Draft** — the generated three-field report (typed out).
- **Verify** — coverage/contradiction issues as badges.
- **Refine** — the LLM's rewrite (only when the draft was inconsistent).
- **Final** — agent output vs radiologist ground truth, side by side.

## Stack
Next.js (App Router) + React 19 + Framer Motion. Plain CSS (no Tailwind). Port 3100.

## Regenerating data
`export_demo.py` runs `N_CASES` (default 16) leak-free test-split cases through the
real graph. It needs the single-split checkpoints (`../interp_ldam_drw.pt`,
`../breast_report_baseline.pt`), the FAISS index (`../retrieval`), and Ollama up
(falls back to rule-based repair if not).
