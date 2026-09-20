# ═══════════════════════════════════════════════════════════════
# TumorNet v10 — Dockerfile API FastAPI
# Conforme Cahier des Charges : conteneurisation, portabilité,
# dépendances CUDA/TensorFlow, déploiement site ou cloud
# ═══════════════════════════════════════════════════════════════

FROM python:3.10-slim AS base

# Métadonnées
LABEL maintainer="TumorNet Team"
LABEL version="10.0"
LABEL description="TumorNet v10 - Brain Tumor Detection API - EfficientNetB0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELS_DIR=/app/models \
    CONF_THRESHOLD=0.70 \
    ANOMALY_MSE_THRESHOLD=0.02 \
    PORT=8000

# Répertoire de travail
WORKDIR /app

# Dépendances système (OpenCV + TensorFlow + PIL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Copier et installer requirements Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY api.py .


# Créer le répertoire modèles (les .h5 seront montés via volume)
RUN mkdir -p /app/models

# Port exposé FastAPI (Uvicorn)
EXPOSE 8000

# Healthcheck — Cible CDC : latence < 200ms
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée : Uvicorn ASYNCHRONE (exigence CDC)
# --workers 2 : gestion requêtes simultanées multi-services hospitaliers
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--access-log"]
