"""Filtering DSL — end-to-end against the dev stack's Qdrant.

Polars-shaped expression syntax over payload fields (``col("x") == "a"``,
comparisons parenthesized before ``&``/``|``, ``.str`` namespace). Field names
are the *sanitized* payload keys (`cell line` → ``cell_line``); unknown fields
fail resolve with the available list.

Counts are exact pins from the dataset (see __scratch/data-exam/main.ipynb).
Skips when the compose Qdrant isn't reachable.
"""

import socket
from collections.abc import Callable

import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _qdrant_up() -> bool:
    import os

    try:
        with socket.create_connection(
            ("127.0.0.1", int(os.environ.get("QDRANT_PORT", "6333"))), timeout=0.5
        ):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_up(), reason="compose stack down (docker compose up -d qdrant)"
)

EXPECTED_U2OS_IMAGES = 25_160
EXPECTED_U2OS_POS_X = 18_565
EXPECTED_IS_IN_POS_X = 20_401
EXPECTED_CELLS_COMPARTMENT_NULL = 72_487
EXPECTED_NULL_OR_A1CF = 72_560


def _resolve_count(call_tool: Callable[..., CallToolResult], code: str) -> int:
    result = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["num_ids"]


def test_equality_filter(call_tool) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(col("cell_line") == "U2OS")
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == EXPECTED_U2OS_IMAGES


def test_combined_numeric_and_string(call_tool) -> None:
    # Every comparison parenthesized before & — precedence note in the module
    # docstring and in dsl/expressions.py.
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(
            (col("cell_line") == "U2OS") & (col("umap2d_x") > 0)
        )
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == EXPECTED_U2OS_POS_X


def test_or_with_is_in(call_tool) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(
            col("cell_line").is_in(["U2OS", "MCF-7"]) & (col("umap2d_x") > 0)
        )
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == EXPECTED_IS_IN_POS_X


def test_null_semantics_on_cells(call_tool) -> None:
    # Null = payload key absent (ingest drops Nones).
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("cell")
        ids = db.resolve_ids(col("compartment").is_null())
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == EXPECTED_CELLS_COMPARTMENT_NULL


def test_or_null_and_equality(call_tool) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("cells")
        ids = db.resolve_ids(
            col("compartment").is_null() | (col("gene_names") == "A1CF")
        )
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == EXPECTED_NULL_OR_A1CF


def test_count_shortcut(call_tool) -> None:
    result = call_tool(
        "query",
        {"code": 'output({"n": database("cells").count(col("cell_path").is_null())})'},
    )
    assert result.data["result"]["n"] == 102_042


def test_unknown_field_message(call_tool) -> None:
    result = call_tool(
        "query",
        {"code": 'output(database("images").count(col("cell line") == "U2OS"))'},
    )
    error = result.data["error"]
    assert "UnknownFieldError" in error["type"]
    assert "cell_line" in error["message"]


def test_unknown_view_message(call_tool) -> None:
    result = call_tool("query", {"code": 'output(database("protein"))'})
    error = result.data["error"]
    assert "UnknownViewError" in error["type"]
    assert "cells" in error["message"] and "images" in error["message"]
