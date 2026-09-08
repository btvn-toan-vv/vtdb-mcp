"""Live smoke (tier 2): the DSL against the dev stack with the REAL dataset.

Exact counts pinned to the current ~/data/subcellular_embeddings snapshot —
when the dataset changes, these pins change with it. Opt-in ONLY:

    pytest -m live        # runs just this tier (requires compose up + ingest)
"""

import socket
from collections.abc import Callable

import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _qdrant_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6333), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.live

_PROD_ENV = {}  # defaults: localhost:6333/6334 (compose-published dev stack)


@pytest.fixture(autouse=True)
def _point_at_dev_stack():
    """Tier isolation: the tier-1 session fixture repaints QDRANT_* env (and
    caches the client singleton) at the synthetic test container; when both
    tiers run in ONE process (`-m ""`), that paint would leak here. Repaint to
    the dev stack's loopback ports + drop the cached client before each test.
    """
    import os

    for k in ("QDRANT_HOST", "QDRANT_PORT", "QDRANT_GRPC_PORT"):
        os.environ.pop(k, None)  # fall back to defaults (localhost:6333/6334)
    from server.services.qdrant import reset_client

    reset_client()

    # Fail loudly when explicitly live but the stack is down — better than a
    # quiet skip for an opt-in tier.
    if not _qdrant_up():
        pytest.fail(
            "live tests require the dev stack: docker compose up -d && "
            "docker compose up ingest"
        )
    yield
    reset_client()


def _resolve_count(call_tool: Callable[..., CallToolResult], code: str) -> int:
    result = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["num_ids"]


def test_live_u2os_images(call_tool) -> None:
    assert (
        _resolve_count(
            call_tool,
            'db = database("image")\n'
            'ids = db.resolve_ids(col("cell_line") == "U2OS")\n'
            'output({"num_ids": len(ids)})',
        )
        == 25_160
    )


def test_live_u2os_pos_x(call_tool) -> None:
    assert (
        _resolve_count(
            call_tool,
            'db = database("image")\n'
            'ids = db.resolve_ids((col("cell_line") == "U2OS") & (col("umap2d_x") > 0))\n'
            'output({"num_ids": len(ids)})',
        )
        == 18_565
    )


def test_live_cells_compartment_null(call_tool) -> None:
    assert (
        _resolve_count(
            call_tool,
            'db = database("cell")\n'
            'ids = db.resolve_ids(col("compartment").is_null())\n'
            'output({"num_ids": len(ids)})',
        )
        == 72_487
    )


def test_live_cells_null_or_a1cf(call_tool) -> None:
    assert (
        _resolve_count(
            call_tool,
            'db = database("cells")\n'
            'ids = db.resolve_ids(col("compartment").is_null() | (col("gene_names") == "A1CF"))\n'
            'output({"num_ids": len(ids)})',
        )
        == 72_560
    )


def test_live_cell_path_null_count(call_tool) -> None:
    result = call_tool(
        "query",
        {"code": 'output({"n": database("cells").count(col("cell_path").is_null())})'},
    )
    assert result.data["result"]["n"] == 102_042


def test_live_meta_default_cap_message(call_tool) -> None:
    # The loud default-cap error must now teach the two right moves, since the
    # legendary trap was "pass limit=5 → error anyway".
    result = call_tool(
        "query",
        {"code": 'output({"rows": database("cells").meta(None)})'},
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert "truncates" in error["message"]  # explains explicit-limit semantics
    assert "resolve_ids" in error["message"]  # points at the sampling idiom


def test_live_meta_explicit_limit_truncates(call_tool) -> None:
    rows = call_tool(
        "query",
        {"code": 'output({"rows": database("cells").meta(None, limit=3)})'},
    ).data["result"]["rows"]
    assert len(rows) == 3
    assert [r["id"] for r in rows] == [0, 1, 2]


def test_live_groupby_limit_message(call_tool) -> None:
    # Real data has >1000 genes per cell_line, so default group_by caps trip;
    # the error must name the correct kwarg placement (the famous session trap).
    result = call_tool(
        "query",
        {
            "code": 'output({"rows": database("images").group_by("genes").agg(row_count())})'
        },
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert ".agg(..., limit=" in error["message"]
