# ============================================================
# TN-NTN TFT xApp -- Multi-stage Docker Build
# Temporal Fusion Transformer for proactive broadband handover
# ============================================================

# ----- Stage 1: Builder (compile PyTorch + dependencies) -----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
        --timeout 600 --retries 3 \
        -r requirements.txt

# ----- Stage 2: Runtime -----
FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime for PyTorch parallel ops
# curl: Docker health-check probes
# wget/ca-certificates: RMR C library download
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Install RMR shared library AFTER builder copy
RUN wget -q https://packagecloud.io/o-ran-sc/release/packages/debian/stretch/rmr_4.9.4_amd64.deb/download.deb \
        -O /tmp/rmr.deb \
    && dpkg -i /tmp/rmr.deb \
    && rm -f /tmp/rmr.deb \
    && ldconfig

WORKDIR /app

# Application source
COPY src/        ./src/
COPY schemas/    ./schemas/

# TFT model checkpoint (mount or bake in)
# COPY models/     ./models/

# xApp descriptors (O-RAN onboarding)
COPY config-file.json xapp-descriptor.yaml ./
COPY requirements.txt ./

# Model directory mount point
RUN mkdir -p /app/models /app/data

# Non-root user
RUN groupadd -r tftxapp && useradd -r -g tftxapp -d /app -s /sbin/nologin tftxapp \
    && chown -R tftxapp:tftxapp /app
USER tftxapp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TFT_XAPP_HOST=0.0.0.0 \
    TFT_XAPP_PORT=8447 \
    TFT_CHECKPOINT_PATH=/app/models/tft_best.ckpt \
    TFT_DEVICE=cpu \
    DBAAS_SERVICE_HOST=dbaas

EXPOSE 8447 4560 4561

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8447/health || exit 1

CMD ["python", "-m", "src.main"]
