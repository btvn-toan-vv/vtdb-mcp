"""Chainable row sorting: ``<rows>.sortby(...)`` on meta/groupby-agg/search rows.

API (polars-flavored, Python-stable Timsort underneath):

    rows = db.group_by("cell_line").agg(row_count())
    rows.sortby("count")                    # ascending by field
    rows.sortby("count", descending=True)   # descending
    rows.sortby(["cell_line", "count"])     # multi-key, in order
    output({"rows": rows.sortby("count")})

Rows come back as ``Rows`` (a ``list[dict]`` subclass) from
``meta()``/``group_by(...).agg(...)``/``search()``; ``resolve_ids`` stays a
plain ``list[int]`` (sorting ids is meaningless — pin-tested below as misuse).

Semantics pinned:

- by: field name str, or list of names (later keys break ties)
- descending: single bool (False default); ties keep input order (stable)
- rows missing a ``by`` field sort LAST in both directions
- empty input returns an empty list
- ``by`` referencing a key present in NO row → DSLUsageError with available
  key names
"""

from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _sorted(call_tool, code: str) -> list[dict]:
    result: CallToolResult = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["rows"]


def _run(call_tool, code: str) -> CallToolResult:
    return call_tool("query", {"code": dedent_code(code)})


# ---- basics ----------------------------------------------------------------


def test_sortby_ascending(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("gene_names").agg(row_count())
        output({"rows": rows.sortby("gene_names")})
        """,
    )
    assert [r["gene_names"] for r in rows] == [f"G{i:02d}" for i in range(12)]
    assert all(r["count"] == 20 for r in rows)


def test_sortby_descending(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("gene_names").agg(row_count())
        output({"rows": rows.sortby("gene_names", descending=True)})
        """,
    )
    assert [r["gene_names"] for r in rows] == [f"G{i:02d}" for i in reversed(range(12))]


def test_sortby_numeric_field(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("if_plate_id").agg(row_count())
        output({"rows": rows.sortby("if_plate_id")})
        """,
    )
    assert [r["if_plate_id"] for r in rows] == [807, 808, 809]


def test_sortby_multi_key(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by(["gene_names", "cell_line"]).agg(row_count())
        output({"rows": rows.sortby(["cell_line", "gene_names"])})
        """,
    )
    keys = [(r["cell_line"], r["gene_names"]) for r in rows]
    assert keys == sorted(keys)  # cell_line asc primary, gene asc secondary


def test_sortby_varargs_equals_list_form(call_tool, hermetic_env) -> None:
    varargs = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by(["gene_names", "cell_line"]).agg(row_count())
        output({"rows": rows.sortby("cell_line", "gene_names")})
        """,
    )
    as_list = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by(["gene_names", "cell_line"]).agg(row_count())
        output({"rows": rows.sortby(["cell_line", "gene_names"])})
        """,
    )
    assert varargs == as_list
    keys = [(r["cell_line"], r["gene_names"]) for r in varargs]
    assert keys == sorted(keys)


def test_sortby_needs_a_key(call_tool, hermetic_env) -> None:
    result = _run(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("cell_line").agg(row_count())
        output({"rows": rows.sortby()})
        """,
    )
    assert result.data["error"]["type"] == "DSLUsageError"


def test_sortby_ties_keep_input_order(call_tool, hermetic_env) -> None:
    # 4 groups, each count 60; null group is first in the (ascending) input.
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("compartment").agg(row_count())
        output({"rows": rows.sortby("count", descending=True)})
        """,
    )
    assert [r["count"] for r in rows] == [60, 60, 60, 60]
    assert rows[0]["compartment"] is None  # ties keep input order


def test_sortby_null_sort_last_both_directions(call_tool, hermetic_env) -> None:
    code = """
        db = database("cells")
        rows = db.group_by("compartment").agg(row_count())
        output({"rows": rows.sortby("compartment", descending=False)})
    """
    rows = _sorted(call_tool, code)
    assert rows[-1]["compartment"] is None
    rows_desc = _sorted(call_tool, code.replace("descending=False", "descending=True"))
    assert rows_desc[-1]["compartment"] is None  # nulls last even descending
    assert rows_desc[0]["compartment"] is not None


def test_sortby_search_rows_by_score(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.search(like=3, k=8)
        output({"rows": rows.sortby("score", descending=True)})
        """,
    )
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_sortby_empty_is_empty(call_tool, hermetic_env) -> None:
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        rows = db.meta(db.resolve_ids(col("cell_line") == "NOPE"))
        output({"rows": rows.sortby("gene_names")})
        """,
    )
    assert rows == []


def test_sortby_result_is_still_rows(call_tool, hermetic_env) -> None:
    # sorted output round-trips as a plain list of dicts on the wire
    rows = _sorted(
        call_tool,
        """
        db = database("cells")
        output({"rows": db.meta(db.resolve_ids(col("gene_names") == "G05"),
                                columns=["gene_names"]).sortby("gene_names")})
        """,
    )
    assert rows and all(set(r) == {"gene_names", "id"} for r in rows)


# ---- misuse -----------------------------------------------------------------


def test_sortby_unknown_field(call_tool, hermetic_env) -> None:
    result = _run(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("cell_line").agg(row_count())
        output({"rows": rows.sortby("tissue")})
        """,
    )
    error = result.data["error"]
    assert error["type"] in {"DSLUsageError", "AttributeError"}
    assert "cell_line" in error["message"] or "tissue" in error["message"]


def test_sortby_on_plain_ids(call_tool, hermetic_env) -> None:
    # resolve_ids returns list[int] — ids have no payload fields to sort by.
    result = _run(
        call_tool,
        """
        db = database("cells")
        ids = db.resolve_ids(col("cell_line") == "U2OS")
        output({"rows": ids.sortby("count")})
        """,
    )
    assert result.data["error"]["type"] in {"AttributeError", "NameError"}
    assert "sortby" in result.data["error"]["message"]


def test_sortby_bad_by_type(call_tool, hermetic_env) -> None:
    result = _run(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("cell_line").agg(row_count())
        output({"rows": rows.sortby(42)})
        """,
    )
    assert result.data["error"]["type"] in {"DSLUsageError", "TypeError"}


def test_sortby_bad_descending_type(call_tool, hermetic_env) -> None:
    result = _run(
        call_tool,
        """
        db = database("cells")
        rows = db.group_by("cell_line").agg(row_count())
        output({"rows": rows.sortby("count", descending="yes")})
        """,
    )
    assert result.data["error"]["type"] in {"DSLUsageError", "TypeError"}
