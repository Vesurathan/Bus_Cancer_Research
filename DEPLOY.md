# Deploying the live demo

Two pieces:

| Piece | What | Where |
|---|---|---|
| **Frontend** (`web/`) | Next.js upload UI (`/live`) + replay demo (`/`) | **Vercel** |
| **Backend** (`serve.py`) | FastAPI wrapping the ML pipeline | **Hugging Face Spaces** (Docker) — or Render / Railway / Fly.io |

The frontend calls the backend's `POST /predict`. Vercel cannot run the ML pipeline
(PyTorch + BiomedCLIP + 343 MB of weights, no GPU, function-size/timeout limits), so the
model lives on a container host and Vercel only serves the UI.

> ⚠️ Research prototype — **not a medical device, not for clinical use.** Keep the disclaimer visible.

---

## 1. Backend → Hugging Face Spaces (recommended: free CPU, 16 GB RAM)

The backend needs `prod/` (≈ **345 MB**: `vit_malignancy.pt`, `knn_bank.npz`,
`descriptor.pkl`, `calibrator_image.pkl`, `birads.pkl`, `meta.json`). These are
git-ignored in the code repo, so upload them to the Space with **git-lfs**.

1. Create a new Space → **Docker** SDK → blank.
2. Add a `README.md` at the Space root with this frontmatter (tells Spaces the port):

   ```yaml
   ---
   title: BUS Agent API
   emoji: 🩻
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push the code + weights to the Space:

   ```bash
   git clone https://huggingface.co/spaces/<you>/bus-agent-api && cd bus-agent-api
   git lfs install
   # copy the backend files into the Space clone:
   #   serve.py config.py diagnose.py malignancy_predictor.py malignancy_model.py
   #   malignancy_data.py retrieval_tool.py finding_agent.py finding_model.py
   #   diagnosis_agent.py draft_agent.py generator_model.py verifier.py refiner.py
   #   explain.py vocab.py morphology_features.py roi_crop.py
   #   requirements-serve.txt Dockerfile .dockerignore  prod/
   git lfs track "prod/*.pt" "prod/*.npz"
   git add -A && git commit -m "BUS agent backend" && git push
   ```

   The Space builds the Dockerfile and starts `uvicorn serve:app` on port 7860.
   First boot downloads BiomedCLIP (~400 MB) — give it a few minutes.

4. Health check: open `https://<you>-bus-agent-api.hf.space/health` → `{"status":"ok",...}`.

**Render / Railway / Fly.io** work the same way (they build the `Dockerfile`); pick an
instance with **≥ 4 GB RAM** (the free 512 MB tiers are too small). They inject `$PORT`,
which the Dockerfile's `CMD` already honours.

## 2. Frontend → Vercel

1. Import the GitHub repo `Vesurathan/Bus_Cancer_Research` in Vercel.
2. **Root Directory** → `web`.
3. **Environment variable**:

   ```
   NEXT_PUBLIC_API_URL = https://<you>-bus-agent-api.hf.space
   ```

4. Deploy. Visit `/live`, upload a BUS image, get a live result.

Then lock CORS down on the backend by setting `ALLOWED_ORIGINS` to your Vercel URL
(e.g. `https://bus-agent.vercel.app`) instead of the default `*`.

---

## Local development

Backend (uses the local `.venv` with torch):

```bash
.venv/bin/python -m pip install fastapi uvicorn python-multipart
.venv/bin/uvicorn serve:app --host 0.0.0.0 --port 7860
```

Frontend (defaults `NEXT_PUBLIC_API_URL` to `http://localhost:7860`):

```bash
cd web && npm install && npm run dev
```

Open http://localhost:3000/live.

---

## API

`POST /predict` — multipart form:
- `image`: the BUS image file (required)
- `descriptors`: optional JSON string, e.g. `{"age":54,"Shape":"irregular"}`

Returns JSON: `decision`, `pMalignant`, `confidence`, `biRads`, `streams`,
`oodDistance`, `neighbours`, `rationale`, `review{flagged,reason}`, `findings[]`,
`possibleFindings[]`, `report`, `heatmap` (Grad-CAM data URI), `image` (echo data URI).

`GET /health` — liveness + which artifacts loaded.

Environment knobs: `PROD_DIR` (default `./prod`), `ALLOWED_ORIGINS` (CORS),
`REFINE_MODE` (`rule` on hosts without Ollama — the Dockerfile sets this).
