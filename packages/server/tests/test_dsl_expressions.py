"""Pure AST tests for the filtering DSL (no server, no Qdrant)."""

import pytest
from server.dsl import DSLUsageError, col


def test_comparison_builds_cmp_node() -> None:
    node = (col("cell_line") == "U2OS").node
    assert node == {"kind": "cmp", "field": "cell_line", "op": "eq", "value": "U2OS"}


def test_all_operators() -> None:
    for py_op, op in [
        ("__ne__", "ne"),
        ("__gt__", "gt"),
        ("__ge__", "ge"),
        ("__lt__", "lt"),
        ("__le__", "le"),
    ]:
        expr = getattr(col("umap2d_x"), py_op)(3.5)
        assert expr.node["op"] == op
        assert expr.node["value"] == 3.5


def test_and_or_not_tree() -> None:
    a = col("cell_line") == "U2OS"
    b = col("umap2d_x") > 0
    combo = (a & b) | ~a
    assert combo.node["kind"] == "or"
    assert combo.node["parts"][0]["kind"] == "and"
    assert combo.node["parts"][1] == {"kind": "not", "child": a.node}


def test_parenthesized_precedence_shape() -> None:
    # Python's real parse of the polars-style one-liner:
    expr = (col("a") == 1) & (col("b") > 2)
    assert expr.node["kind"] == "and"
    fields = [p["field"] for p in expr.node["parts"]]
    assert fields == ["a", "b"]


def test_truthiness_raises_guided_error() -> None:
    with pytest.raises(DSLUsageError, match="Combine with &"):
        bool(col("x") == 1)
    with pytest.raises(DSLUsageError):
        bool(col("x"))


def test_methods() -> None:
    assert col("cell_line").is_in(["HeLa", "U2OS"]).node == {
        "kind": "is_in",
        "field": "cell_line",
        "values": ["HeLa", "U2OS"],
    }
    assert col("cell_path").is_null().node == {
        "kind": "is_null",
        "field": "cell_path",
    }
    assert col("cell_path").is_not_null().node == {
        "kind": "is_not_null",
        "field": "cell_path",
    }
    assert col("cell_line").str.contains("U").node == {
        "kind": "str_contains",
        "field": "cell_line",
        "needle": "U",
    }


def test_bad_value_types_rejected() -> None:
    with pytest.raises(DSLUsageError):
        _ = col("x") == object()  # pyright: ignore[reportOperatorIssue]
    with pytest.raises(DSLUsageError):
        col("x").is_in([object()])  # type: ignore[list-item]
    with pytest.raises(DSLUsageError):
        col("x").str.contains(42)  # type: ignore[arg-type]


def test_col_name_validation() -> None:
    with pytest.raises(DSLUsageError):
        col("")
    with pytest.raises(DSLUsageError):
        col(42)  # type: ignore[arg-type]


def test_iter_fields_walks_tree() -> None:
    from server.dsl.expressions import iter_fields

    # Mind the parens: a > 0 | b would parse as a > (0 | b) — the exact trap
    # the module docstring warns about.
    expr = (col("cell_line") == "U2OS") & (
        (col("umap2d_x") > 0) | col("compartment").is_null()
    )
    assert sorted(iter_fields(expr)) == ["cell_line", "compartment", "umap2d_x"]
