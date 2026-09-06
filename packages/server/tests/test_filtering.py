"""Filtering DSL — hermetic integration (tier 1).

Polars-shaped expressions over payload fields, executed through the real MCP
tool against a throwaway Qdrant container loaded with a deterministic
synthetic dataset (fixtures in conftest.hermetic_env). No dependence on
the ~/data/subcellular_embeddings snapshot — expected counts are COMPUTED from
the fixture, not pinned. Needs Docker; skips cleanly without it.

Field names are the *sanitized* payload keys (`cell line` → ``cell_line``);
unknown fields fail resolve with the available list. Comparison parentheses
matter (see the module-level note in dsl/expressions.py).
"""

from collections.abc import Callable

from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _resolve_count(call_tool: Callable[..., CallToolResult], code: str) -> int:
    result = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["num_ids"]


def test_equality_filter(call_tool, hermetic_env) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(col("cell_line") == "U2OS")
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == hermetic_env["images"]["u2os"]


def test_combined_numeric_and_string(call_tool, hermetic_env) -> None:
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
    assert num_ids == hermetic_env["images"]["u2os_pos_x"]


def test_or_with_is_in(call_tool, hermetic_env) -> None:
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
    assert num_ids == hermetic_env["images"]["is_in_pos_x"]


def test_null_semantics_on_cells(call_tool, hermetic_env) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("cell")
        ids = db.resolve_ids(col("compartment").is_null())
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == hermetic_env["cells"]["compartment_null"]


def test_or_null_and_equality(call_tool, hermetic_env) -> None:
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("cells")
        ids = db.resolve_ids(
            col("compartment").is_null() | (col("gene_names") == "G07")
        )
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids == hermetic_env["cells"]["null_or_g07"]


def test_count_shortcut(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {"code": 'output({"n": database("cells").count(col("cell_path").is_null())})'},
    )
    assert result.data["result"]["n"] == hermetic_env["cells"]["path_null"]


def test_unknown_field_message(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {"code": 'output(database("images").count(col("cell line") == "U2OS"))'},
    )
    # spaces are NOT the payload keys — the sanitizer renamed them at ingest
    error = result.data["error"]
    assert "UnknownFieldError" in error["type"]
    assert "cell_line" in error["message"]


def test_unknown_view_message(call_tool, hermetic_env) -> None:
    result = call_tool("query", {"code": 'output(database("protein"))'})
    error = result.data["error"]
    assert "UnknownViewError" in error["type"]
    assert "cells" in error["message"] and "images" in error["message"]
