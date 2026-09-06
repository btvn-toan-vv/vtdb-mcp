"""Filtering DSL sketches — polars-shaped syntax for the future
``database() -> db`` vocabulary (``db.resolve_ids(filter_expr)``).

These pin the *surface* we want before the symbols are registered in
VtdbSymbolContext: filter expressions imitate polars
(``col("x") == "a"`` / comparison before ``&`` / ``|``, string ops under the
``.str`` namespace). Strict-xfail until the vocabulary lands; XPASS then fails
loudly so the marker gets removed.

Precedence reminder (why the parens): Python binds ``&`` and ``|`` TIGHTER
than ``==``/``>`` — ``a == b & c > d`` parses as ``a == (b & (c > d))``.
Polars style parenthesizes every comparison for exactly this reason.
"""

from typing import Callable

import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code

# Not registered yet (Phase 0 vocabulary is builtins + output only).
pytestmark = pytest.mark.xfail(
    strict=True,
    reason="filtering vocabulary (database, col, resolve_ids) not in the symbol context yet",
)


def _resolve_count(call_tool: Callable[..., CallToolResult], code: str) -> int:
    result = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["num_ids"]


def test_equality_filter(call_tool) -> None:
    # Polars: pl.col("cell line") == "U2OS"  (NOT .equals — that compares
    # expression metadata, not values; .eq() is the named variant).
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(col("cell line") == "U2OS")
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids > 0


def test_combined_numeric_and_string(call_tool) -> None:
    # Every comparison parenthesized before & — precedence foot-gun documented
    # in the module docstring.
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids(
            (col("cell line") == "U2OS") & (col("umap2d x") > 0)
        )
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids > 0


def test_or_with_string_namespace(call_tool) -> None:
    # Polars string ops hang off the .str namespace: .str.contains(), not
    # a top-level .str_contains().
    num_ids = _resolve_count(
        call_tool,
        """
        db = database("image")
        u2os_scored = (col("cell line") == "U2OS") & (col("umap2d x") > 0)
        starts_u_scored = col("cell line").str.contains("U") & (col("umap2d x") > 0)
        ids = db.resolve_ids(u2os_scored | starts_u_scored)
        output({"num_ids": len(ids)})
        """,
    )
    assert num_ids > 0
