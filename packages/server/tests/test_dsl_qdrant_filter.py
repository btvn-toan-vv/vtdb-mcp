"""AST → qdrant_client Filter translation shape tests (no server)."""

from typing import Any

import pytest
from qdrant_client import models
from server.dsl import DSLUsageError, col
from server.dsl.qdrant_filter import to_qdrant_filter

_FIELD_TYPES = (
    models.FieldCondition,
    models.IsEmptyCondition,
    models.Filter,
)


def _translate(expr) -> models.Filter:
    f = to_qdrant_filter(expr)
    assert f is not None
    return f


def _as_list(v: Any) -> list:
    """qdrant models type must/should/must_not as Condition | List[Condition]."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _cond_keys(cond: object) -> list[tuple[str, str]]:
    """(kind, key) pairs deep inside Filter/condition compositions."""
    if isinstance(cond, models.FieldCondition):
        if cond.range is not None:
            kind = "Range"
        elif cond.match is not None:
            kind = type(cond.match).__name__
        else:
            kind = "FieldCondition"
        return [(kind, cond.key)]
    if isinstance(cond, models.IsEmptyCondition):
        return [("IsEmpty", cond.is_empty.key)]
    if isinstance(cond, models.Filter):
        out: list[tuple[str, str]] = []
        for group in (
            _as_list(cond.must) + _as_list(cond.should) + _as_list(cond.must_not)
        ):
            out.extend(_cond_keys(group))
        return out
    raise AssertionError(f"unexpected node: {type(cond).__name__}")


def test_none_passthrough() -> None:
    assert to_qdrant_filter(None) is None


def test_equality() -> None:
    f = _translate(col("cell_line") == "U2OS")
    assert _cond_keys(f) == [("MatchValue", "cell_line")]


def test_range_ops() -> None:
    for op_kw, expected in [
        ("__gt__", "gt"),
        ("__ge__", "gte"),
        ("__lt__", "lt"),
        ("__le__", "lte"),
    ]:
        f = _translate(getattr(col("umap2d_x"), op_kw)(1.5))
        cond = _as_list(f.must)[0]
        assert getattr(cond.range, expected) == 1.5


def test_ne_is_must_not_eq() -> None:
    f = _translate(col("cell_line") != "HeLa")
    assert _cond_keys(_as_list(f.must_not)[0]) == [
        ("MatchValue", "cell_line")
    ] or _cond_keys(f) == [("MatchValue", "cell_line")]
    assert f.must_not  # something sits in must_not


def test_is_in() -> None:
    f = _translate(col("cell_line").is_in(["HeLa", "U2OS"]))
    assert _as_list(f.must)[0].match.any == ["HeLa", "U2OS"]


def test_null_semantics_empty_not_null() -> None:
    # Ingest drops Nones from payloads -> "null" means the key is ABSENT,
    # so is_null maps to Qdrant's IsEmptyCondition, not IsNullCondition.
    f = _translate(col("cell_path").is_null())
    assert _cond_keys(f) == [("IsEmpty", "cell_path")]
    f2 = _translate(col("cell_path").is_not_null())
    assert f2.must_not


def test_eq_none_becomes_is_null() -> None:
    # `== None` IS the DSL op here — overloaded __eq__ builds an is-null
    # FilterExpr; this is not a boolean comparison, so the noqa is deliberate.
    f = _translate(col("compartment") == None)  # noqa: E711
    assert _cond_keys(f) == [("IsEmpty", "compartment")]


def test_str_contains_is_matchtext() -> None:
    f = _translate(col("compartment").str.contains("Nuc"))
    assert isinstance(_as_list(f.must)[0].match, models.MatchText)


def test_and_flattens_must_groups() -> None:
    f = _translate((col("cell_line") == "U2OS") & (col("umap2d_x") > 0))
    keys = _cond_keys(f)
    assert keys == [("MatchValue", "cell_line"), ("Range", "umap2d_x")]
    assert f.should is None  # stayed a pure must conjunction


def test_or_becomes_should() -> None:
    f = _translate((col("cell_line") == "U2OS") | (col("umap2d_x") > 0))
    should = _as_list(f.should)
    assert len(should) == 2


def test_not_becomes_must_not() -> None:
    f = _translate(~(col("cell_line") == "U2OS"))
    assert f.must_not and _cond_keys(_as_list(f.must_not)[0]) == [
        ("MatchValue", "cell_line")
    ]


def test_range_rejects_non_numbers() -> None:
    with pytest.raises(DSLUsageError, match="needs a number"):
        to_qdrant_filter(col("umap2d_x") > "high")


def test_non_expression_rejected() -> None:
    with pytest.raises(DSLUsageError, match="expected a filter expression"):
        to_qdrant_filter("col('x') == 1")  # type: ignore[arg-type]
