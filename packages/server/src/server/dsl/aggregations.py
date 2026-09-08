"""Aggregation expression vocabulary for `db.group_by(...).agg(...)`.

Polars parity: aggregations hang off FieldExpr (``col("time_ms").mean()``),
``.alias("name")`` renames, and ``row_count()`` is the free group-size symbol.
Aggregation runs server-side via polars (gold-truth parity by construction).

Pinned surface (tests/test_groupby.py): count, sum, mean, median, min, max,
std, var, quantile(q), first, last, n_unique — plus row_count(). Numeric-only
ops (sum/mean/median/std/var/quantile) reject non-numeric fields at agg() time
(dsl/views.py validates against dsl/schema.py types).
"""

from __future__ import annotations

from typing import Any

from server.dsl.expressions import DSLUsageError

# ops this module knows how to pin (used by views.py for validation/execution)
AGG_OPS_WITH_FIELD = frozenset(
    {
        "count",  # non-null rows
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "quantile",
        "first",
        "last",
        "n_unique",
    }
)
NUMERIC_ONLY_OPS = frozenset({"sum", "mean", "median", "std", "var", "quantile"})


def default_alias(field: str, op: str, q: float | None = None) -> str:
    """Output column name when no alias given: time_ms_mean, time_ms_q0.9."""
    if op == "quantile":
        return f"{field}_q{q}"
    return f"{field}_{op}"


class AggExpr:
    """One aggregated output column for GroupBy.agg: col(field).<op>(...)."""

    __hash__ = object.__hash__

    def __init__(self, node: dict[str, Any]) -> None:
        self.node = node

    def alias(self, name: str) -> AggExpr:
        """Rename the output column (polars' Expr.alias)."""
        if not isinstance(name, str) or not name:
            raise DSLUsageError("alias() expects a non-empty name (str)")
        node = dict(self.node)
        node["alias"] = name
        return AggExpr(node)

    def __bool__(self) -> bool:
        raise DSLUsageError(
            "an aggregation expression can't be used as a boolean — it belongs "
            "inside .agg(...) after group_by(...)"
        )

    def __repr__(self) -> str:
        return f"AggExpr({self.node!r})"


def row_count() -> AggExpr:
    """Group size including nulls (polars' pl.len())."""
    return AggExpr({"kind": "agg", "op": "row_count", "field": None})


def _agg(field_expr: Any, op: str, **params: Any) -> AggExpr:
    """Guard + builder for FieldExpr.<op>() aggregation methods."""
    from server.dsl.expressions import FieldExpr  # local: avoid module cycle

    if not isinstance(field_expr, FieldExpr):
        raise DSLUsageError(
            f"aggregation {op}() is a column method: col('field').{op}(...)"
        )
    return AggExpr(
        {
            "kind": "agg",
            "op": op,
            "field": field_expr.name,
            "params": params,
            "alias": None,
        }
    )
