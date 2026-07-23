# ─────────────────────────────────────────────────────────
# CloudBurst — Dockerfile
# ─────────────────────────────────────────────────────────
# Build:    docker build -t cloudburst .
# Run:      docker run -p 8501:8501 --env-file .env cloudburst
# ─────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────
COPY . .

# ── Expose Streamlit port ──────────────────────────────
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run ─────────────────────────────────────────────────
CMD ["streamlit", "run", "src/prediction/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false"]

