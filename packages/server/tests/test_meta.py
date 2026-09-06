"""REVIEW DRAFT — ``db.meta(...)`` metadata fetcher, pinned before implementation.

API being pinned (polars-flavored, mirrors resolve_ids):

    db = database("cells")
    db.meta(query, columns=None, limit=10000) -> list[dict]

- ``query``: a FilterExpr (col(...) == ...) OR an int id list OR a single int id.
- Rows are payload dicts with the point id attached under ``"id"``.
- Row order follows the requested ids (ids form); filter form follows scroll
  order (don't rely on it beyond "consistent within one call").
- ``columns=[...]`` projects payload keys (Qdrant with_payload filter).
- Missing ids are silently dropped (Qdrant retrieve semantics).
- ``limit`` defaults to 10_000 and FAILS LOUD when exceeded (no silent caps).

All assertions use conftest.PayloadOracle (the fixture's own rows), so the
suite stays valid if the fixture generator changes.
"""

from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _meta(call_tool, code: str):
    result: CallToolResult = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["rows"]


def test_meta_by_explicit_ids_keeps_order(call_tool, hermetic_env) -> None:
    rows = _meta(
        call_tool,
        'output({"rows": database("cells").meta([3, 0, 7])})',
    )
    assert [r["id"] for r in rows] == [3, 0, 7]
    oracle = hermetic_env["payload_of"]
    for r in rows:
        expected = oracle("cells", r["id"])
        expected["id"] = r["id"]
        assert r == expected  # full payload, exact


def test_meta_by_single_int(call_tool, hermetic_env) -> None:
    rows = _meta(call_tool, 'output({"rows": database("images").meta(5)})')
    assert len(rows) == 1
    oracle = hermetic_env["payload_of"]
    assert rows[0] == {**oracle("images", 5), "id": 5}


def test_meta_by_filter(call_tool, hermetic_env) -> None:
    # Every row 0..239 has cell_path=None on i%5==0 → first hits in scroll order.
    rows = _meta(
        call_tool,
        'output({"rows": database("cells").meta(col("cell_path").is_null())})',
    )
    n_null = hermetic_env["cells"]["path_null"]
    assert len(rows) == hermetic_env["cells"]["path_null"] <= 10_000
    oracle = hermetic_env["payload_of"]
    for r in rows:
        assert r == {**oracle("cells", r["id"]), "id": r["id"]}
    assert n_null > 0  # guard against a vacuous fixture


def test_meta_filter_roundtrip_with_resolve_ids(call_tool, hermetic_env) -> None:
    rows = _meta(
        call_tool,
        """
        db = database("image")
        ids = db.resolve_ids((col("cell_line") == "U2OS") & (col("umap2d_x") > 0))
        output({"rows": db.meta(ids)})
        """,
    )
    oracle = hermetic_env["payload_of"]
    assert len(rows) == hermetic_env["images"]["u2os_pos_x"]
    assert all(r["umap2d_x"] > 0 for r in rows)
    for r in rows:
        assert r == {**oracle("images", r["id"]), "id": r["id"]}


def test_meta_column_projection(call_tool, hermetic_env) -> None:
    rows = _meta(
        call_tool,
        'output({"rows": database("cells").meta([0, 1], columns=["cell_line", "gene_names"])})',
    )
    assert [set(r) for r in rows] == [{"id", "cell_line", "gene_names"}] * 2
    assert [r["cell_line"] for r in rows] == ["U2OS", "HeLa"]  # fixture rotation


def test_meta_projection_on_filter(call_tool, hermetic_env) -> None:
    # The combo that wasn't in the id-form: filter path must project identically.
    rows = _meta(
        call_tool,
        """
        output({"rows": database("cells").meta(
            (col("cell_line") == "U2OS") & (col("cell_id") < 2),
            columns=["cell_line", "gene_names"],
        )})
        """,
    )
    assert [r["id"] for r in rows] == [0]  # row 0: U2OS (0%3) and cell_id 0<2
    assert set(rows[0]) == {"id", "cell_line", "gene_names"}
    assert rows[0]["gene_names"] == "G00"


def test_meta_unknown_column(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {"code": 'output({"rows": database("cells").meta([0], columns=["foo"])})'},
    )
    error = result.data["error"]
    assert error["type"] == "UnknownFieldError"
    assert "cell_line" in error["message"]  # available list offered


def test_meta_limit_guard(call_tool, hermetic_env) -> None:
    # Fixture has 240 cells; cap 5 should trip loudly, not truncate silently.
    result = call_tool(
        "query",
        {"code": 'output({"rows": database("cells").meta(None, limit=5)})'},
    )
    assert "limit" in result.data["error"]["message"]


def test_meta_missing_ids_are_dropped(call_tool, hermetic_env) -> None:
    # 999_999 does not exist; qdrant retrieve omits missing points.
    rows = _meta(
        call_tool,
        'output({"rows": database("cells").meta([0, 999_999])})',
    )
    assert [r["id"] for r in rows] == [0]


def test_meta_bad_type(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query", {"code": 'output({"rows": database("cells").meta("U2OS")})'}
    )
    assert result.data["error"]["type"] == "DSLUsageError"


def test_meta_no_args_is_guided(call_tool, hermetic_env) -> None:
    result = call_tool("query", {"code": 'output({"rows": database("cells").meta()})'})
    # no-arg is too risky to full-scan by default — guide the user instead.
    assert result.data["error"]["type"] in {"DSLUsageError", "TypeError"}
