# Single image, two entrypoints (API and UI) selected by docker-compose.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ui ./ui
COPY pytest.ini ./
COPY tests ./tests

# Generate the dataset and warm the retrieval index at build time so the container starts
# ready to answer. If the embedding model is not in the image cache, the build falls back
# to the deterministic TF-IDF encoder - the build never depends on the network.
RUN python -m app.dataset.generator && python scripts/build_index.py || true

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
