FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E '^(torch|torchvision|torchaudio)==' /app/backend/requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
COPY docker /app/docker

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/backend/uploads /app/models

WORKDIR /app/backend

EXPOSE 5000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
