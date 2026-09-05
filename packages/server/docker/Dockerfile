# syntax=docker/dockerfile:1.7
#
# vtdb-mcp SERVER image — the FastMCP server (noarch, slim CPython).
#
# Two-stage build modeled on exen-mcp's docker/Dockerfile.log-service: stage 1
# resolves the venv with uv from the committed lock (the biocircle dep comes
# from the private bioturing index — the token is a BuildKit SECRET, never a
# layer/env); stage 2 is a slim runtime with just the venv, a non-root user,
# and tini as PID 1.
#
# Build context is packages/server/ (the Dockerfile lives in its docker/ dir):
#
#   docker build \
#     --secret id=bioturing_token,env=UV_INDEX_BIOTURING_PASSWORD \
#     -f packages/server/docker/Dockerfile -t vtdb-mcp-server:dev \
#     packages/server
#
# or simply `docker compose up -d server` from the repo root (compose passes
# the secret from the UV_INDEX_BIOTURING_PASSWORD env var itself).

# ---------------------------------------------------------------------------- #
# Stage 1: builder — resolve + install the venv with uv.
# ---------------------------------------------------------------------------- #
FROM python:3.14-slim AS builder

# Release metadata (injected by CI/manual builds); labels only, no behavior.
ARG VERSION="" GIT_SHA="" BUILD_DATE=""

# Username for the private index. The password/token comes in as a secret
# (same pattern as exen-mcp's docker/Dockerfile.base).
ARG UV_INDEX_BIOTURING_USERNAME=bioturing

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=180

# ca-certificates: HTTPS to PyPI; git: some uv sources resolve over git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Same pinned uv as exen-mcp's images.
COPY --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /usr/local/bin/

WORKDIR /app

# Project metadata + lock first (cache-friendly), then the source tree the
# uv_build backend packages (pyproject declares readme = "README.md").
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src

# Install from the committed lock, runtime deps only (no dev group).
# fd limit: BuildKit RUN steps default to a low soft NOFILE (1024 on Docker
# 27); UV_COMPILE_BYTECODE fans out a thread-per-file compiler that exhausts
# it -> "No file descriptors available (os error 24)". Bump soft to hard
# (same fix as exen-mcp's Dockerfiles). The token secret is optional so a
# warm uv cache can rebuild without it; without either, uv fails on the
# private 401.
# --no-editable: `uv sync` defaults to an EDITABLE project install (a
# server.pth pointing at /app/src) — the runtime stage only copies the venv,
# so install the project as a real wheel instead.
RUN --mount=type=secret,id=bioturing_token \
    --mount=type=cache,target=/root/.cache/uv \
    ulimit -n "$(ulimit -Hn)" && \
    export UV_INDEX_BIOTURING_USERNAME="${UV_INDEX_BIOTURING_USERNAME}" && \
    if [ -f /run/secrets/bioturing_token ]; then \
        export UV_INDEX_BIOTURING_PASSWORD="$(cat /run/secrets/bioturing_token)"; \
    fi && \
    for attempt in 1 2 3; do \
        echo ">>> uv sync attempt ${attempt}/3" && \
        uv sync --frozen --no-dev --no-editable && break || { \
            echo ">>> attempt ${attempt} failed; retrying" >&2; \
            sleep ${attempt}; \
        }; \
    done

# ---------------------------------------------------------------------------- #
# Stage 2: runtime — slim image with just the venv.
# ---------------------------------------------------------------------------- #
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    # Bind all interfaces inside the container (compose publishes the port).
    VTDB_HOST=0.0.0.0 \
    VTDB_PORT=8000

# Release metadata so the image self-identifies. Empty on a manual run.
ARG VERSION="" GIT_SHA="" BUILD_DATE=""
LABEL org.opencontainers.image.title="vtdb-mcp-server" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# tini for clean signal handling as PID 1 (uvicorn gets SIGTERM on
# `docker stop`).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Record the release version in a file operators can cat. Empty-string guard
# keeps manual (tag-less) builds working.
RUN if [ -n "$VERSION" ]; then printf '%s\n' "$VERSION" > /app/VERSION; fi

# Non-root user created UP FRONT so the COPY below sets ownership inline via
# --chown (a trailing `chown -R` would duplicate every copied byte into a new
# layer — same lesson as exen-mcp's Dockerfile.server).
RUN useradd --create-home --uid 10001 vtdb

# Bring over only the resolved venv — `uv sync --no-editable` installs the
# `server` project itself as a wheel, so no source tree is needed at runtime.
COPY --chown=vtdb:vtdb --from=builder /app/.venv /app/.venv

USER vtdb

# MCP (streamable-http at /mcp) + /health share this port.
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "server"]
# Extra flags, e.g. ["--port", "8000", "--debug"]; env VTDB_* also work.
CMD []
