# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

# --- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build deps for any wheels lacking an arm64 build (asyncpg ships one, but
# this keeps the build robust across base-image bumps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer is cached independently of source changes.
COPY pyproject.toml README.md ./
COPY cribrix ./cribrix

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && pip install .

# --- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system --gid 1001 cribrix \
    && useradd --system --uid 1001 --gid cribrix --create-home cribrix

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=cribrix:cribrix cribrix ./cribrix

USER cribrix

EXPOSE 8000

# Single worker by default: the workload is I/O-bound and async, so scale with
# replicas (which also gives you independent health checks) rather than with
# in-process workers.
CMD ["uvicorn", "cribrix.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
