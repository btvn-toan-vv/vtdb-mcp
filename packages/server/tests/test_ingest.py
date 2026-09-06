"""Ingest logic: key sanitizing, point iteration, and the idempotency contract
(skip / resume / recreate) — all against FakeQdrant, no server needed."""

from pathlib import Path

import pytest
from qdrant_client import models

from server.ingest import (
    _INDEXING_THRESHOLD_DEFAULT,
    _INDEXING_THRESHOLD_DEFERRED,
    ViewSpec,
    _iter_points,
    _sanitize_key,
    ensure_collection,
    load_view,
    wait_for_green,
)

from .conftest import FakeQdrant


# ---- helpers -------------------------------------------------------------


def test_sanitize_key() -> None:
    assert _sanitize_key("cell line") == "cell_line"
    assert _sanitize_key("Time (ms)") == "time_ms"
    assert _sanitize_key("if_plate_id") == "if_plate_id"
    assert _sanitize_key("umap2d x") == "umap2d_x"


def _uploaded_ids(client: FakeQdrant, name: str) -> set[int]:
    return set(client.collections[name]["points"])


# ---- point iteration -------------------------------------------------------


def test_iter_points_ids_and_payload(view: ViewSpec, dataset: Path) -> None:
    view_dir = dataset / view.dir_name
    batches = list(_iter_points(view, view_dir, limit=0, resume_from=0))

    assert len(batches) == 1
    points, start, end, total = batches[0]
    assert (start, end, total) == (0, 5, 5)
    assert [p.id for p in points] == [0, 1, 2, 3, 4]

    # payload keys sanitized; None values dropped (row 1 has no compartment)
    p0 = points[0].payload
    assert p0["cell_line"] == "HeLa"
    assert p0["time_ms"] == pytest.approx(21.1)
    assert "time (ms)" not in p0
    assert "compartment" not in points[1].payload
    assert len(points[0].vector) == view.dims


def test_iter_points_limit(view: ViewSpec, dataset: Path) -> None:
    view_dir = dataset / view.dir_name
    batches = list(_iter_points(view, view_dir, limit=3, resume_from=0))
    assert batches[0][3] == 3
    assert [p.id for p in batches[0][0]] == [0, 1, 2]


def test_iter_points_resume(view: ViewSpec, dataset: Path) -> None:
    view_dir = dataset / view.dir_name
    batches = list(_iter_points(view, view_dir, limit=0, resume_from=3))
    assert [p.id for p in batches[0][0]] == [3, 4]
    assert batches[0][1] == 3  # start offset preserved


def test_iter_points_shape_mismatch(
    view: ViewSpec, dataset: Path, tmp_path: Path
) -> None:
    view_dir = dataset / view.dir_name
    import numpy as np

    np.save(view_dir / "embeddings.npy", np.random.random((4, view.dims)))
    with pytest.raises(ValueError, match="rows vs meta"):
        list(_iter_points(view, view_dir, limit=0, resume_from=0))


# ---- ensure_collection: create / skip / migrate ----------------------------


def test_ensure_collection_creates(client: FakeQdrant, view: ViewSpec) -> None:
    assert ensure_collection(client, view, force=False) is True

    coll = client.collections[view.collection]
    assert coll["vectors"].size == view.dims
    assert coll["vectors"].distance == models.Distance.COSINE
    # HNSW deferred while loading; payload indexes created BEFORE any points.
    assert coll["optimizers"].indexing_threshold == _INDEXING_THRESHOLD_DEFERRED
    assert [f for f, _ in coll["indexes"]] == list(view.payload_indexes)


def test_ensure_collection_skips_existing(client: FakeQdrant, view: ViewSpec) -> None:
    ensure_collection(client, view, force=False)
    calls_before = list(client.calls)

    assert ensure_collection(client, view, force=False) is False
    assert "delete_collection" not in calls_before
    assert client.calls == calls_before  # untouched on the skip path


def test_ensure_collection_force_recreates(client: FakeQdrant, view: ViewSpec) -> None:
    ensure_collection(client, view, force=False)
    client.calls.clear()

    assert ensure_collection(client, view, force=True) is True
    assert "delete_collection" in client.calls


def test_ensure_collection_recreates_on_drift(
    client: FakeQdrant, view: ViewSpec
) -> None:
    ensure_collection(client, view, force=False)
    # Simulate a stale schema from an older dataset version.
    client.collections[view.collection]["vectors"] = models.VectorParams(
        size=view.dims + 1, distance=models.Distance.COSINE
    )
    client.calls.clear()

    assert ensure_collection(client, view, force=False) is True
    assert "delete_collection" in client.calls
    assert client.collections[view.collection]["vectors"].size == view.dims


# ---- load_view: the idempotency contract ------------------------------------


def test_load_view_full_then_skip(
    client: FakeQdrant, view: ViewSpec, dataset: Path
) -> None:
    load_view(client, view, dataset, limit=0, force=False)
    assert _uploaded_ids(client, view.collection) == {0, 1, 2, 3, 4}
    # HNSW indexing threshold restored after the load.
    updated = client.collections[view.collection]["updated_optimizers"]
    assert updated[-1].indexing_threshold == _INDEXING_THRESHOLD_DEFAULT

    uploads_before = client.uploads
    load_view(client, view, dataset, limit=0, force=False)
    assert client.uploads == uploads_before  # second run: nothing re-uploaded


def test_load_view_resumes_partial(
    client: FakeQdrant, view: ViewSpec, dataset: Path
) -> None:
    ensure_collection(client, view, force=False)
    client.seed_points(view.collection, range(3))  # pretend a run died at row 3

    load_view(client, view, dataset, limit=0, force=False)

    assert _uploaded_ids(client, view.collection) == {0, 1, 2, 3, 4}
    # Only the missing rows were uploaded; seeded points were not overwritten.
    assert client.collections[view.collection]["points"][0] == {"seeded": True}


def test_load_view_recreate_on_overshoot(
    client: FakeQdrant, view: ViewSpec, dataset: Path
) -> None:
    # Dataset shrank since the last load (7 stale points for a 5-row view) —
    # resuming makes no sense, so the collection is rebuilt from scratch.
    ensure_collection(client, view, force=False)
    client.seed_points(view.collection, range(7))
    client.calls.clear()

    load_view(client, view, dataset, limit=0, force=False)

    assert "delete_collection" in client.calls
    assert _uploaded_ids(client, view.collection) == {0, 1, 2, 3, 4}


def test_load_view_limit(client: FakeQdrant, view: ViewSpec, dataset: Path) -> None:
    load_view(client, view, dataset, limit=3, force=False)
    assert _uploaded_ids(client, view.collection) == {0, 1, 2}


def test_load_view_missing_dataset(
    client: FakeQdrant, view: ViewSpec, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="dataset view missing"):
        load_view(client, view, tmp_path, limit=0, force=False)


# ---- green-status wait ----------------------------------------------------


def test_wait_for_green_immediate(client: FakeQdrant, view: ViewSpec) -> None:
    ensure_collection(client, view, force=False)
    wait_for_green(client, timeout_s=1.0)  # all GREEN in the fake — returns at once
