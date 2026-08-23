# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/models \
    SSD_MODEL_NAME=ssdlite320-mobilenet-v3-large \
    SSD_MODEL_VERSION=torchvision-coco-v1 \
    SSD_SCORE_THRESHOLD=0.5

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip install . \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /models \
    && chown -R app:app /app /models

USER app

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=4).read()"]

CMD ["python", "-m", "uvicorn", "ssd_model.app:app", "--host", "0.0.0.0", "--port", "8002"]
