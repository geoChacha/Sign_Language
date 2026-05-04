# ============================================================
# EmotiSign Backend — Dockerfile
# Build context: emotisign-backend/
# Python 3.10.9 — CPU only, no CUDA/GPU dependencies
# ============================================================

FROM python:3.10.9-slim

# ── System dependencies ───────────────────────────────────────
# libgl1        — required by OpenCV
# libglib2.0-0  — required by MediaPipe
# libgomp1      — required by PyTorch CPU (OpenMP)
# curl          — used by HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first — app code changes won't bust this layer
COPY requirements.txt .

# PyTorch CPU wheels are on a separate index.
# BuildKit cache mount keeps downloaded wheels on the host between builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip --quiet \
    && pip install --quiet \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Application code ──────────────────────────────────────────
# keypoints_best/ was copied into emotisign-backend/ so it's in build context
COPY . .

# ── Runtime directories ───────────────────────────────────────
RUN mkdir -p data uploads/videos uploads/audio uploads/tts_output static/signs

# ── Environment ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ML_DEVICE=cpu \
    KEYPOINTS_DIR=/app/keypoints_best

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
