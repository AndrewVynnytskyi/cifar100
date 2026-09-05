# ============================================================
#  CIFAR-100 VGG11 Project — Multi-stage Dockerfile
# ============================================================
# Stage 1: builder  — install dependencies into a venv
# Stage 2: runtime  — copy only the venv + source code
#
# Multi-stage keeps the final image lean (~500 MB vs ~1.5 GB).
# ============================================================

# ---------- Stage 1: builder --------------------------------
FROM python:3.11-slim AS builder

# System build deps (needed by some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# `pip install ".[dev]"` below builds this project itself, which needs the actual
# package source (src/) and the file pyproject.toml's readme= field points at
# (README.md) present — not just pyproject.toml.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip then install project dependencies (CPU-only torch for CI/inference)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0+cpu \
        torchvision==0.18.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dev]"


# ---------- Stage 2: runtime --------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security best practice
RUN useradd --create-home appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy source code (respects .dockerignore — data/ and outputs/ are excluded)
COPY --chown=appuser:appuser . .

# Pre-create the data directory so CIFAR-100 can download here
RUN mkdir -p /app/data /app/outputs && chown -R appuser:appuser /app

USER appuser

# Expose W&B environment variable (pass your key with -e WANDB_API_KEY=...)
ENV WANDB_API_KEY=""
ENV PYTHONUNBUFFERED=1

# Note: no PYTHONPATH override needed — the project is installed into the venv
# (see builder stage), and all internal imports use the `src.` prefix
# (e.g. `from src.model import VGG11CIFAR100`), matching that installed layout.

# Default: run training with the bundled config.
# Override with: docker run <image> python -m src.experiments.hierarchical ...
CMD ["python", "-m", "src.train"]
