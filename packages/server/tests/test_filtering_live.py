"""Live smoke (tier 2): the DSL against the dev stack with the REAL dataset.

Exact counts pinned to the current ~/data/subcellular_embeddings snapshot —
when the dataset changes, these pins change with it. Opt-in ONLY:

    pytest -m live        # runs just this tier (requires compose up + ingest)
"""

import socket
from typing import Callable

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


@pytest.fixture(autouse=True)
def _require_live_stack() -> None:
    # Fail loudly when explicitly asked to run live but the stack is down —
    # better than a quiet skip for an opt-in tier.
    if not _qdrant_up():
        pytest.fail(
            "live tests require the dev stack: docker compose up -d && "
            "docker compose up ingest"
        )


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
