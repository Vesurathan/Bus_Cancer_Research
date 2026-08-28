# Live inference backend (serve.py) for the agentic BUS pipeline.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io — anything
# that builds a Dockerfile. CPU-only; needs ~3–4 GB RAM at runtime.
FROM python:3.11-slim

# libgomp1 is needed by torch/faiss (OpenMP); the rest keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (layer cache) — BiomedCLIP itself is downloaded at runtime.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code + trained artifacts. `prod/` MUST be present (see DEPLOY.md).
COPY . .

# Cache HF/torch downloads inside the container's writable home.
ENV HF_HOME=/app/.cache/hf \
    TORCH_HOME=/app/.cache/torch \
    PORT=7860 \
    REFINE_MODE=rule
# ^ default to the rule-based refiner: no Ollama on the host, so skip the LLM step.

EXPOSE 7860
# shell form so $PORT (set by Render/Railway/Fly) overrides the default
CMD uvicorn serve:app --host 0.0.0.0 --port ${PORT:-7860}
