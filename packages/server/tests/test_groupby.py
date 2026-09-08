"""REVIEW DRAFT — groupby + aggregation DSL, pinned before implementation.

API being pinned (polars-shaped; gold truth computed with polars on the
fixture's own payloads via conftest oracle.frame):

    db = database("cells")
    db.group_by("cell_line")                  # str key or list[str]
      .agg(
          col("time_ms").mean(),            # default name: "time_ms_mean"
          col("time_ms").max().alias("t_max"),
          row_count(),                       # group size, default name "count"
      )                                     # -> list[dict]

- Row keys: the group-by field(s) + one output column per agg. Rows are SORTED
  by the group key(s) ascending. Null group keys form their own group
  (polars-compatible).
- Aggregations on FieldExpr: count (non-null), sum, mean, median, min, max,
  std, var, quantile(q), first, last, n_unique; plus free row_count(). Mean/
  std/var/quantile skip nulls like polars; string fields reject numeric aggs
  with DSLUsageError naming the field.
- Optional ``among=`` pre-filter (same FilterExpr shape as search) restricts
  the rows grouped; ``limit=`` (default 1000) caps the group count and FAILS
  LOUD when exceeded.
- ``.alias("name")`` renames any output column.

Fixture facts (cells, n=240): cell_line cycles U2OS/HeLa/MCF-7 (80/80/80);
compartment cycles incl. None every 4th (60); gene_names G00..G11 (20 each);
if_plate_id 807/808/809 (80 each); position P<i%96>; time_ms float with
~14% null; cell_path None for i%5==0. images (n=120): cell_line 40/40/40,
umap2d_x floats.
"""

import polars as pl
import pytest
from fastmcp.client.client import CallToolResult

from .conftest import dedent_code


def _agg_rows(call_tool, code: str) -> list[dict]:
    result: CallToolResult = call_tool("query", {"code": dedent_code(code)})
    return result.data["result"]["rows"]


def _truth(hermetic_env, view: str) -> pl.DataFrame:
    """The fixture's own payloads as a polars frame = independent gold truth."""
    return hermetic_env["payload_of"].frame(view)


# ---- basics --------------------------------------------------------------


def test_groupby_row_count(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(row_count())})',
    )
    assert [(r["cell_line"], r["count"]) for r in rows] == [
        ("HeLa", 80),
        ("MCF-7", 80),
        ("U2OS", 80),
    ]
    assert list(rows[0].keys()) == ["cell_line", "count"]  # key first, sorted asc


def test_groupby_field_count_nonnull(call_tool, hermetic_env) -> None:
    # col.count() counts non-null in group: gene_names has no nulls → equals
    # group size; time_ms drops ~14% nulls.
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(col("time_ms").count())})',
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by("cell_line")
        .agg(pl.col("time_ms").count())
    )
    for r in rows:
        # polars names the bare count column after the field
        assert (
            r["time_ms_count"]
            == gold.filter(pl.col("cell_line") == r["cell_line"])["time_ms"][0]
        )


def test_groupby_numeric_aggs(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        """
        output({"rows": database("cells").group_by("cell_line").agg(
            col("time_ms").mean(),
            col("time_ms").median(),
            col("time_ms").std(),
            col("time_ms").min(),
            col("time_ms").max(),
        )})
        """,
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by("cell_line")
        .agg(
            mean=pl.col("time_ms").mean(),
            median=pl.col("time_ms").median(),
            std=pl.col("time_ms").std(),
            min=pl.col("time_ms").min(),
            max=pl.col("time_ms").max(),
        )
    )
    truth = {g["cell_line"]: g for g in gold.iter_rows(named=True)}
    assert {r["cell_line"] for r in rows} == set(truth)
    for r in rows:
        t = truth[r["cell_line"]]
        assert r["time_ms_mean"] == pytest.approx(t["mean"], rel=1e-4)
        assert r["time_ms_median"] == pytest.approx(t["median"], rel=1e-4)
        assert r["time_ms_std"] == pytest.approx(t["std"], rel=1e-3)
        assert r["time_ms_min"] == pytest.approx(t["min"])
        assert r["time_ms_max"] == pytest.approx(t["max"])


def test_groupby_sum_and_var(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("images").group_by("cell_line").agg(col("umap2d_x").sum(), col("umap2d_x").var())})',
    )
    gold = (
        _truth(hermetic_env, "images")
        .group_by("cell_line")
        .agg(s=pl.col("umap2d_x").sum(), v=pl.col("umap2d_x").var())
    )
    truth = {g["cell_line"]: g for g in gold.iter_rows(named=True)}
    for r in rows:
        t = truth[r["cell_line"]]
        assert r["umap2d_x_sum"] == pytest.approx(t["s"], rel=1e-4)
        assert r["umap2d_x_var"] == pytest.approx(t["v"], rel=1e-3)


def test_groupby_quantile(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(col("time_ms").quantile(0.9))})',
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by("cell_line")
        .agg(pl.col("time_ms").quantile(0.9, interpolation="nearest"))
    )
    truth = {g["cell_line"]: g["time_ms"] for g in gold.iter_rows(named=True)}
    for r in rows:
        assert r["time_ms_q0.9"] == pytest.approx(truth[r["cell_line"]])


def test_groupby_first_last_scroll_order(call_tool, hermetic_env) -> None:
    # first/last follow storage order (ascending row index of the view).
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(col("cell_id").first(), col("cell_id").last())})',
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by("cell_line")
        .agg(f=pl.col("cell_id").first(), l=pl.col("cell_id").last())
    )
    truth = {g["cell_line"]: g for g in gold.iter_rows(named=True)}
    for r in rows:
        assert r["cell_id_first"] == truth[r["cell_line"]]["f"]
        assert r["cell_id_last"] == truth[r["cell_line"]]["l"]


def test_groupby_n_unique(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(col("gene_names").n_unique())})',
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by("cell_line")
        .agg(pl.col("gene_names").n_unique())
    )
    truth = {g["cell_line"]: g["gene_names"] for g in gold.iter_rows(named=True)}
    for r in rows:
        assert r["gene_names_n_unique"] == truth[r["cell_line"]]


def test_groupby_alias(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line").agg(col("time_ms").mean().alias("t_mean"), row_count().alias("n"))})',
    )
    assert list(rows[0].keys()) == ["cell_line", "t_mean", "n"]


# ---- multi-key / among / caps ---------------------------------------------


def test_groupby_multi_key(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by(["cell_line", "compartment"]).agg(row_count())})',
    )
    gold = (
        _truth(hermetic_env, "cells")
        .group_by(["cell_line", "compartment"])
        .agg(pl.len())
    )
    truth = {
        (g["cell_line"], g["compartment"]): g["len"] for g in gold.iter_rows(named=True)
    }
    assert rows
    assert {(r["cell_line"], r["compartment"]) for r in rows} == set(truth)
    for r in rows:
        assert r["count"] == truth[(r["cell_line"], r["compartment"])]


def test_groupby_among_prefilter(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line", among=col("cell_line") == "HeLa").agg(row_count())})',
    )
    assert [(r["cell_line"], r["count"]) for r in rows] == [("HeLa", 80)]


def test_groupby_limit_fails_loud(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("position", limit=3).agg(row_count())})'
            )
        },
    )
    assert result.data["error"]["type"] == "DSLUsageError"
    assert "limit" in result.data["error"]["message"]


def test_groupby_agg_limit_override(call_tool, hermetic_env) -> None:
    # limit= on agg() itself works too (same as group_by(limit=)) — the error
    # message in the wild named only "pass limit=", so both paths exist now.
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("position").agg(row_count(), limit=200)})',
    )
    assert len(rows) == 96

    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("position").agg(row_count(), limit=3)})'
            )
        },
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    # the message must show the exact remedy — no "pass limit=" vagueness
    assert "group_by(" in error["message"] and ".agg(..., limit=" in error["message"]


def test_groupby_empty_match_is_empty(call_tool, hermetic_env) -> None:
    rows = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("cell_line", among=col("cell_line") == "NOPE").agg(row_count())})',
    )
    assert rows == []


# ---- misuse ----------------------------------------------------------------


def test_groupby_unknown_by_field(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("tissue").agg(row_count())})'
            )
        },
    )
    error = result.data["error"]
    assert error["type"] == "UnknownFieldError"
    assert "cell_line" in error["message"]


def test_groupby_unknown_agg_field(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("cell_line").agg(col("tissue").mean())})'
            )
        },
    )
    error = result.data["error"]
    assert error["type"] == "UnknownFieldError"
    assert "time_ms" in error["message"]


def test_groupby_numeric_op_on_string_field(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("cell_line").agg(col("cell_line").mean())})'
            )
        },
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert "cell_line" in error["message"]


def test_agg_rejects_filter_expr(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("cell_line").agg(col("time_ms") > 0)})'
            )
        },
    )
    error = result.data["error"]
    assert error["type"] == "DSLUsageError"
    assert "agg" in error["message"]


def test_groupby_bad_by_type(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by(42).agg(row_count())})'
            )
        },
    )
    assert result.data["error"]["type"] in {"DSLUsageError", "TypeError"}


def test_quantile_bounds(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"rows": database("cells").group_by("cell_line").agg(col("time_ms").quantile(1.5))})'
            )
        },
    )
    assert result.data["error"]["type"] == "DSLUsageError"


# ---- where() chaining (filter-before-group) --------------------------------


def test_where_groupby_matches_among(call_tool, hermetic_env) -> None:
    # Chained form equals the among= form:
    #   db.where(a).group_by(b)  ==  db.group_by(b, among=a)
    chained = _agg_rows(
        call_tool,
        'output({"rows": database("cells").where(col("cell_line") == "HeLa").group_by("compartment").agg(row_count())})',
    )
    direct = _agg_rows(
        call_tool,
        'output({"rows": database("cells").group_by("compartment", among=col("cell_line") == "HeLa").agg(row_count())})',
    )
    assert chained == direct


def test_where_count_quick_path(call_tool, hermetic_env) -> None:
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"n": database("cells").where(col("cell_line") == "U2OS").count()})'
            )
        },
    )
    assert result.data["result"]["n"] == 80


def test_where_handle_unfinished_chain(call_tool, hermetic_env) -> None:
    # Handles aren't JSON-serializable: output(...) of a bare where-handle
    # returns its guidance repr (named accessors are the intended next step).
    result = call_tool(
        "query",
        {
            "code": dedent_code(
                'output({"v": database("cells").where(col("gene_names") == "G01")})'
            )
        },
    )
    assert "group_by" in result.data["result"]["v"]
    assert "FilteredView" in result.data["result"]["v"]
