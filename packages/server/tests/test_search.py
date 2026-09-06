"""REVIEW DRAFT — filtered top-k vector search: ``db.search(...)``.

API being pinned (mirrors resolve_ids/meta; Qdrant query_points underneath):

    db = database("cells")
    db.search(expr, k=10, like=ID, columns=None, with_vector=False)
        -> list[dict]

- ``expr``: a FilterExpr restricting the candidates (None = whole view).
- ``like``: anchor point id **of the same view** — "more like this one".
- ``k``: top-k by cosine similarity; defaults 10. k > matches returns the
  (smaller) match set; k <= 0 raises.
- Rows: ``{"id", "score", ...payload}`` sorted by score DESC; score is cosine
  similarity in [-1, 1]; anchor scores 1.0 when it matches the filter.
- ``columns=[...]`` projects payload keys (like meta); `"id"`/`"score"` always
  present. ``with_vector=True`` adds the raw ``"vector"`` list (float32 dims).
- Anchor id missing from the view -> DSLUsageError (anchors are load-bearing;
  silent empties hide bugs, unlike rows-dropped leniency in meta).

All rows use conftest.PayloadOracle; fixture: 240 cells / 120 images
(cells have no umap*; gene_names cycle "G00".."G11", cell_line U2OS/HeLa/MCF-7).
"""

import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code

pytestmark = pytest.mark.xfail(
    strict=True, reason="ViewHandle.search is not implemented yet — review draft"
)


def _search(call_tool, code: str) -> list[dict]:
    result: CallToolResult = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["rows"]


def _search_code(view: str = "cells", expr: str | None = None, **kw) -> str:
    expr_src = expr or "None"
    args = ", ".join(
        f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in kw.items()
    )
    return (
        f"db = database(\"{view}\")\noutput({{'rows': db.search({expr_src}, {args})}})"
    )


# ---- shape / ordering -----------------------------------------------------


def test_search_top_k_shape_and_order(call_tool, hermetic_env) -> None:
    rows = _search(
        call_tool, _search_code(expr='col("gene_names") == "G03"', k=5, like=3)
    )
    assert len(rows) == 5
    assert all(r["gene_names"] == "G03" for r in rows)
    assert {*rows[0].keys()} >= {"id", "score", "gene_names"}
    # score descending; anchor included with self-similarity 1.0
    assert rows[0]["id"] == 3 and rows[0]["score"] == pytest.approx(1.0)
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert len({r["id"] for r in rows}) == 5  # distinct


def test_search_payload_matches_artifact(call_tool, hermetic_env) -> None:
    rows = _search(
        call_tool, _search_code(expr='col("gene_names") == "G03"', k=3, like=3)
    )
    oracle = hermetic_env["payload_of"]
    for r in rows:
        expect = {**oracle("cells", r["id"]), "id": r["id"], "score": r["score"]}
        assert r == expect


def test_search_anchor_excluded_when_filtered_out(call_tool, hermetic_env) -> None:
    # Anchor id 3 is gene G03; filtering to G04 must NOT surface the anchor.
    rows = _search(
        call_tool, _search_code(expr='col("gene_names") == "G04"', k=5, like=3)
    )
    assert 3 not in {r["id"] for r in rows}
    assert all(r["gene_names"] == "G04" for r in rows)


def test_search_k_larger_than_matches_returns_all(call_tool, hermetic_env) -> None:
    # compartment "Vesicles" rows: i % 4 == 3 -> 60 of 240.
    rows = _search(
        call_tool,
        _search_code(expr='col("compartment") == "Vesicles"', k=500, like=3),
    )
    assert len(rows) == 60


def test_search_no_match_is_empty(call_tool, hermetic_env) -> None:
    rows = _search(
        call_tool,
        _search_code(
            expr='(col("gene_names") == "G03") & (col("cell_line") == "HeLaX")',
            k=5,
            like=3,
        ),
    )
    # careful: HeLaX unknown value, not unknown field — valid filter, zero hits
    assert rows == []


# ---- payload / vector projection ------------------------------------------


def test_search_columns_project(call_tool, hermetic_env) -> None:
    rows = _search(
        call_tool,
        _search_code(
            expr='col("gene_names") == "G07"', k=3, like=7, columns=["gene_names"]
        ),
    )
    assert all(set(r.keys()) == {"id", "score", "gene_names"} for r in rows)


def test_search_with_vector(call_tool, hermetic_env) -> None:
    rows = _search(
        call_tool,
        _search_code(expr='col("gene_names") == "G07"', k=1, like=7, with_vector=True),
    )
    assert len(rows) == 1
    assert isinstance(rows[0]["vector"], list)
    assert len(rows[0]["vector"]) == 1536  # cells dims (ingest ViewSpec)


# ---- misuses are guided ----------------------------------------------------


def test_search_anchor_missing_is_error(call_tool, hermetic_env) -> None:
    result: CallToolResult = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").search(None, k=5, like=999_999)})'
            )
        },
    )
    assert result.data["error"]["type"] == "DSLUsageError"
    assert "999999" in result.data["error"]["message"]


def test_search_requires_anchor_like(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").search(col("gene_names") == "G03")})'
            )
        },
    )
    assert result.data["error"]["type"] in {"DSLUsageError", "TypeError"}


def test_search_rejects_bad_k(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").search(None, k=0, like=3)})'
            )
        },
    )
    assert result.data["error"]["type"] == "DSLUsageError"


def test_search_rejects_string_expr(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").search("cell_line == U2OS", k=5, like=3)})'
            )
        },
    )
    assert result.data["error"]["type"] == "DSLUsageError"
