"""Load the subcellular-embeddings dataset into Qdrant — idempotent bootstrap.

One-shot by design (the `ingest` service in compose.yaml), also runnable
locally against the compose-published ports::

    uv run python -m server.ingest                       # full load
    uv run python -m server.ingest --limit 20000         # smoke-size load
    uv run python -m server.ingest --force               # rebuild collections

Checks-before-load contract (safe to re-run any time):

- dataset files must exist → fail fast with a clear path message;
- collection missing → created; schema drift (dims/distance changed) or
  ``--force`` → collection dropped + recreated (= the "migration" path);
- collection already holds exactly the expected row count → skip the view
  (loads are upserts keyed by row index, so a partial re-run converges).

Vectors are raw model outputs (cell norms ≈ 24.6, image norms ≈ 5.8 — nowhere
near unit length), so collections use COSINE distance; see
__scratch/data-exam/main.ipynb §6.

Bulk-load posture follows Qdrant's indexing-performance guidance: payload
indexes are created while the collection is empty (before HNSW materializes),
HNSW build is deferred via a huge ``indexing_threshold`` during the upload and
re-enabled afterwards, and uploads go through batched ``upload_points`` with
parallel workers.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Qdrant connection (compose injects QDRANT_HOST=qdrant; local CLI hits the
# loopback-published ports of the compose container).
_QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
_QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
_QDRANT_TIMEOUT = int(os.environ.get("QDRANT_TIMEOUT", "180"))

# Dataset location: container path in compose, host path for local runs.
_DATA_DIR = Path(
    os.environ.get("VTDB_DATA_DIR", "/home/toanvong/data/subcellular_embeddings")
)

# Upload tuning (Qdrant bulk-load guidance: batches of 64–256, 2–4 streams).
_BATCH_SIZE = int(os.environ.get("VTDB_INGEST_BATCH_SIZE", "256"))
_PARALLEL = int(os.environ.get("VTDB_INGEST_PARALLEL", "4"))

# Deferred-HNSW trick: indexing_threshold is in KiB (default 20000 = 20 MiB);
# setting it astronomically high keeps the optimizer from building HNSW during
# the bulk load (the `m=0` approach is legacy/deprecated).
_INDEXING_THRESHOLD_DEFERRED = 100_000_000
_INDEXING_THRESHOLD_DEFAULT = 20_000

_DISTANCE = models.Distance.COSINE


@dataclass(frozen=True)
class ViewSpec:
    """One dataset view ↔ one Qdrant collection."""

    collection: str
    dir_name: str  # under the dataset dir
    dims: int
    payload_indexes: dict[str, models.PayloadSchemaType]
    on_disk_vectors: bool = False


CELLS = ViewSpec(
    collection="cells",
    dir_name="cell_embedding",
    dims=1536,
    payload_indexes={
        "gene_names": models.PayloadSchemaType.KEYWORD,
        "compartment": models.PayloadSchemaType.KEYWORD,
        "cell_line": models.PayloadSchemaType.KEYWORD,
        "if_plate_id": models.PayloadSchemaType.INTEGER,
    },
)
IMAGES = ViewSpec(
    collection="images",
    dir_name="image_embedding",
    dims=1024,
    payload_indexes={
        "protein": models.PayloadSchemaType.KEYWORD,
        "cell_line": models.PayloadSchemaType.KEYWORD,
        "antibody": models.PayloadSchemaType.KEYWORD,
        "compartment": models.PayloadSchemaType.KEYWORD,
    },
)
VIEWS: dict[str, ViewSpec] = {v.collection: v for v in (CELLS, IMAGES)}

_KEY_SANITIZER = re.compile(r"[^0-9A-Za-z]+")


def _sanitize_key(name: str) -> str:
    """Metadata column -> payload key ("cell line" -> "cell_line")."""
    return _KEY_SANITIZER.sub("_", name).strip("_").lower()


def _connect_with_retry(client: QdrantClient, attempts: int = 20) -> None:
    """Wait for the qdrant service (compose gives it no healthcheck — the
    distroless image ships no shell/curl to probe with)."""
    for attempt in range(1, attempts + 1):
        try:
            client.get_collections()
            logger.info(
                "Connected to Qdrant at %s (http:%d grpc:%d)",
                _QDRANT_HOST,
                _QDRANT_PORT,
                _QDRANT_GRPC_PORT,
            )
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.info("Qdrant not ready (attempt %d/%d): %s", attempt, attempts, exc)
            time.sleep(2)


def ensure_collection(client: QdrantClient, view: ViewSpec, force: bool) -> bool:
    """Make the collection match the spec; payload indexes included.

    Returns True if the collection was (re)created. Drop+recreate on force or
    schema drift — nothing else in this dataset is migrate-able in place.
    """
    exists = client.collection_exists(view.collection)
    if exists:
        info = client.get_collection(view.collection)
        params = info.config.params.vectors
        drift = (
            params is None
            or isinstance(params, dict)  # named/multi-vector: unexpected here
            or params.size != view.dims
            or params.distance != _DISTANCE
        )
        if not force and not drift:
            return False
        logger.warning(
            "Recreating collection %r (%s)",
            view.collection,
            "--force" if force else f"schema drift: {params}",
        )
        client.delete_collection(view.collection)

    logger.info(
        "Creating collection %r (%d-d, cosine; HNSW deferred during bulk load)",
        view.collection,
        view.dims,
    )
    client.create_collection(
        collection_name=view.collection,
        vectors_config=models.VectorParams(
            size=view.dims,
            distance=_DISTANCE,
            on_disk=view.on_disk_vectors,
        ),
        # Defer HNSW while points stream in; restored after the load.
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=_INDEXING_THRESHOLD_DEFERRED
        ),
    )
    # Payload indexes now — before HNSW builds — so the filterable vector
    # index picks them up (creating them after HNSW is an anti-pattern).
    for field, schema in view.payload_indexes.items():
        client.create_payload_index(view.collection, field, schema)
        logger.info("  payload index: %s (%s)", field, schema.value)
    return True


def _iter_points(
    view: ViewSpec, view_dir: Path, limit: int, resume_from: int
) -> Iterator[tuple[list[models.PointStruct], int, int, int]]:
    """Yield (batch, start, end, total); point id = row index (idempotent).

    ``resume_from`` skips rows [0, resume_from): the writer below uploads in
    ascending row-id order, so a partial run's point count is exactly the
    resume offset.
    """
    emb = np.load(view_dir / "embeddings.npy", mmap_mode="r")
    meta = pl.read_csv(view_dir / "metadata.csv")
    n_rows = meta.height
    if emb.shape[0] != n_rows:
        raise ValueError(f"{view.dir_name}: emb {emb.shape[0]} rows vs meta {n_rows}")
    if emb.shape[1] != view.dims:
        raise ValueError(f"{view.dir_name}: {emb.shape[1]}-d != spec {view.dims}")

    n = min(n_rows, limit) if limit else n_rows
    keys = [_sanitize_key(c) for c in meta.columns]
    for start in range(resume_from, n, _BATCH_SIZE * _PARALLEL):
        end = min(start + _BATCH_SIZE * _PARALLEL, n)
        vectors = emb[start:end].astype(np.float32)
        rows = meta.slice(start, end - start).iter_rows()
        batch = []
        for i, row in enumerate(rows):
            payload: dict[str, Any] = {
                k: v for k, v in zip(keys, row, strict=True) if v is not None
            }
            batch.append(
                models.PointStruct(
                    id=start + i,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )
        yield batch, start, end, n


def load_view(
    client: QdrantClient, view: ViewSpec, data_dir: Path, limit: int, force: bool
) -> None:
    view_dir = data_dir / view.dir_name
    if not (view_dir / "embeddings.npy").is_file():
        raise FileNotFoundError(f"dataset view missing: {view_dir}")

    created = ensure_collection(client, view, force)
    total = pl.scan_csv(view_dir / "metadata.csv").select(pl.len()).collect().item()
    assert isinstance(total, int)
    expected = min(total, limit) if limit else total

    current = client.count(view.collection, exact=True).count
    if not created and current == expected:
        logger.info(
            "Collection %r already holds %d points — skipping load",
            view.collection,
            current,
        )
        return
    if not created and current > expected:
        logger.warning(
            "Collection %r has %d points but the dataset expects %d — recreating",
            view.collection,
            current,
            expected,
        )
        created = ensure_collection(client, view, force=True)

    # Resume-friendly: ascending row ids means a partial run's point count IS
    # the resume offset. Full re-upsert only when nothing landed yet.
    resume_from = 0 if created else max(current, 0)
    if resume_from:
        logger.info(
            "Resuming %r from point id %d (%d remain)",
            view.collection,
            resume_from,
            expected - resume_from,
        )
    logger.info(
        "Loading %d %s vectors (%s) at batch=%d parallel=%d ...",
        expected - resume_from,
        view.collection,
        view_dir,
        _BATCH_SIZE,
        _PARALLEL,
    )
    t0 = time.monotonic()
    for batch, start, end, total in _iter_points(view, view_dir, limit, resume_from):
        client.upload_points(
            view.collection,
            batch,
            batch_size=_BATCH_SIZE,
            parallel=_PARALLEL,
            max_retries=5,
        )
        done = end - resume_from
        remaining = total - end
        rate = done / (time.monotonic() - t0)
        eta_s = remaining / rate if rate else 0.0
        logger.info(
            "  %s: %d/%d (%.1f%%) %.0f pts/s ETA %s",
            view.collection,
            end,
            total,
            100.0 * end / total,
            rate,
            time.strftime("%H:%M:%S", time.gmtime(max(eta_s, 0))),
        )

    count = client.count(view.collection, exact=True).count
    if count != expected:
        raise RuntimeError(
            f"{view.collection}: uploaded {count} != expected {expected}"
        )

    # Re-enable HNSW; the optimizer now builds the graph asynchronously.
    client.update_collection(
        view.collection,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=_INDEXING_THRESHOLD_DEFAULT
        ),
    )
    logger.info(
        "Loaded %r (%d points in %.0fs); HNSW re-enabled (builds async)",
        view.collection,
        count,
        time.monotonic() - t0,
    )


def wait_for_green(client: QdrantClient, timeout_s: float = 3600.0) -> None:
    """Block until every collection reports green (HNSW/optimizers settled)."""
    deadline = time.monotonic() + timeout_s
    statuses: dict[str, Any] = {}
    while time.monotonic() < deadline:
        statuses = {
            c.name: client.get_collection(c.name).status
            for c in client.get_collections().collections
        }
        if all(s == models.CollectionStatus.GREEN for s in statuses.values()):
            logger.info("All collections green: %s", statuses)
            return
        time.sleep(5)
    raise TimeoutError(f"collections not green after {timeout_s}s: {statuses}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the subcellular-embeddings dataset into Qdrant."
    )
    parser.add_argument(
        "--force", action="store_true", help="drop + recreate collections"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("VTDB_INGEST_LIMIT", "0") or 0),
        help="ingest only the first N rows of each view (0 = all)",
    )
    parser.add_argument(
        "--only",
        choices=sorted(VIEWS),
        action="append",
        default=None,
        help="load only this view (repeatable; default: all)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="skip waiting for green status (HNSW build) before exiting",
    )
    args = parser.parse_args()

    if not _DATA_DIR.is_dir():
        sys.exit(f"dataset dir not found: {_DATA_DIR} (set VTDB_DATA_DIR)")

    client = QdrantClient(
        host=_QDRANT_HOST,
        port=_QDRANT_PORT,
        grpc_port=_QDRANT_GRPC_PORT,
        prefer_grpc=True,
        timeout=_QDRANT_TIMEOUT,
    )
    _connect_with_retry(client)

    wanted = {name: VIEWS[name] for name in (args.only or list(VIEWS))}
    for view in wanted.values():
        load_view(client, view, _DATA_DIR, args.limit, args.force)

    if args.no_wait:
        logger.info("Skipping green-status wait (--no-wait)")
        return
    wait_for_green(client)


if __name__ == "__main__":
    from server.log import configure_logging

    configure_logging()
    main()
