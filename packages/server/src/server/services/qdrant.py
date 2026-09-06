"""Shared Qdrant client, lazily built from env (QDRANT_HOST/PORT/GRPC_PORT).

One client per process: qdrant_client threads its own connection pool, so a
singleton is both safe and preferred here.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_lock = threading.Lock()


def reset_client() -> None:
    """Drop the cached client (tests repoint env before the next get_client)."""
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                logger.debug("Qdrant client close failed during reset: %s", exc)
            _client = None


def get_client() -> QdrantClient:
    """Process-wide Qdrant client (gRPC-preferred), with connect retry."""
    global _client
    with _lock:
        if _client is not None:
            return _client
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))
        grpc_port = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
        client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=180,
            check_compatibility=False,
        )
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                client.get_collections()
                logger.info(
                    "Qdrant client connected (host=%s http:%d grpc:%d)",
                    host,
                    port,
                    grpc_port,
                )
                break
            except Exception as exc:  # noqa: BLE001 — transient startup race
                last_exc = exc
                logger.info("Qdrant not ready (attempt %d/5): %s", attempt, exc)
                time.sleep(1)
        else:
            raise ConnectionError(
                f"cannot reach Qdrant at {host}:{port}/{grpc_port}"
            ) from last_exc
        _client = client
    return _client
