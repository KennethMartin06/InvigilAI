# syntax=docker/dockerfile:1.6
#
# Backend image — FastAPI + PyTorch + MediaPipe + ensemble models.
# Built from the repo root: docker build -f proctoring_app/backend/Dockerfile .

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for OpenCV / MediaPipe / psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (cached layer)
COPY proctoring_app/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy backend source and trained models
COPY proctoring_app/backend/ /app/proctoring_app/backend/
COPY cheating_detection/models/ /app/cheating_detection/models/
COPY cheating_detection/config.py /app/cheating_detection/config.py
COPY cheating_detection/__init__.py /app/cheating_detection/__init__.py
COPY proctoring_app/__init__.py /app/proctoring_app/__init__.py

# Upload dir (persisted via volume in production)
RUN mkdir -p /app/proctoring_app/backend/uploads/screenshots

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run uvicorn with workers tuned for CPU (override via env var)
CMD ["uvicorn", "proctoring_app.backend.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
