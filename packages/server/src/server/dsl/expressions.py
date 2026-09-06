"""Pure-Python filter-expression AST for the query DSL (polars-shaped).

No qdrant imports here — translation lives in qdrant_filter.py. Rules of the
surface (polars parity):

- ``col("cell line") == "U2OS"``` builds a FilterExpr — comparisons NEVER return
  bool; truthiness on either expression class raises a guided error.
- `&`, `|`, `~` combine expressions. Comparison must come first:
  ``(col("a") == 1) & (col("b") > 0)`` — Python binds ``&``/``|`` tighter than
  comparisons; the parens are not optional (same rule as polars/pandas).
- String ops hang off the ``.str`` namespace: ``col("c").str.contains("U")``.

Scalar values must be JSON primitives (str / int / float / bool / None); the
AST is meant to be small, inspectable, and cheap to validate.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

Scalar = str | int | float | bool | None
Node = dict[
    str, Any
]  # {"kind": ..., ...} — plain dicts: inspectable + trivially serializable


class DSLUsageError(ValueError):
    """Misuse of the filtering DSL (truthiness, bad value types, …)."""


def _ast_usage_error(what: str) -> DSLUsageError:
    return DSLUsageError(
        f"{what} is a filter expression, not a truthy value. "
        "Combine with & / | / ~ (not `and` / `or` / `not`), and remember that "
        "comparisons need parentheses around them: (col('a') == 1) & (col('b') > 2)."
    )


class FilterExpr:
    """A complete boolean predicate over payload fields (the AST root)."""

    # Keep identity hashing: __and__ etc. never touch __eq__ here.
    __hash__ = object.__hash__

    def __init__(self, node: Node) -> None:
        self.node = node

    def __and__(self, other: FilterExpr) -> FilterExpr:
        if not isinstance(other, FilterExpr):
            raise DSLUsageError(
                f"cannot combine FilterExpr with {type(other).__name__}"
            )
        return FilterExpr({"kind": "and", "parts": [self.node, other.node]})

    def __rand__(self, other: FilterExpr) -> FilterExpr:
        return self.__and__(other)

    def __or__(self, other: FilterExpr) -> FilterExpr:
        if not isinstance(other, FilterExpr):
            raise DSLUsageError(
                f"cannot combine FilterExpr with {type(other).__name__}"
            )
        return FilterExpr({"kind": "or", "parts": [self.node, other.node]})

    def __ror__(self, other: FilterExpr) -> FilterExpr:
        return self.__or__(other)

    def __invert__(self) -> FilterExpr:
        return FilterExpr({"kind": "not", "child": self.node})

    def __bool__(self) -> bool:
        raise _ast_usage_error("FilterExpr")

    def __repr__(self) -> str:
        return f"FilterExpr({self.node!r})"


class StrExpr:
    """``col("…").str`` — the string-ops namespace (contains; more later)."""

    def __init__(self, field: str) -> None:
        self._field = field

    def contains(self, needle: str) -> FilterExpr:
        if not isinstance(needle, str):
            raise DSLUsageError(
                f"str.contains expects str, got {type(needle).__name__}"
            )
        return FilterExpr(
            {"kind": "str_contains", "field": self._field, "needle": needle}
        )


class FieldExpr:
    """``col("name")`` — a field reference; comparisons build FilterExpr."""

    __hash__ = object.__hash__

    def __init__(self, name: str) -> None:
        self.name = name

    # -- comparisons (polars: == on Expr) ---------------------------------
    def _cmp(self, op: str, value: Any) -> FilterExpr:
        if not (value is None or isinstance(value, (str, int, float, bool))):
            raise DSLUsageError(
                f"col({self.name!r}) {op} only takes str/int/float/bool/None, "
                f"got {type(value).__name__}"
            )
        return FilterExpr({"kind": "cmp", "field": self.name, "op": op, "value": value})

    def __eq__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return self._cmp("eq", other)

    def __ne__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return self._cmp("ne", other)

    def __gt__(self, other: Any) -> FilterExpr:
        return self._cmp("gt", other)

    def __ge__(self, other: Any) -> FilterExpr:
        return self._cmp("ge", other)

    def __lt__(self, other: Any) -> FilterExpr:
        return self._cmp("lt", other)

    def __le__(self, other: Any) -> FilterExpr:
        return self._cmp("le", other)

    # -- membership / nullness -------------------------------------------
    def is_in(self, values: list[Scalar] | tuple[Scalar, ...]) -> FilterExpr:
        if not isinstance(values, (list, tuple)):
            raise DSLUsageError(
                f"is_in expects a list/tuple, got {type(values).__name__}"
            )
        for v in values:
            if not (v is None or isinstance(v, (str, int, float, bool))):
                raise DSLUsageError(
                    f"is_in values must be primitives, got {type(v).__name__}"
                )
        return FilterExpr({"kind": "is_in", "field": self.name, "values": list(values)})

    def is_null(self) -> FilterExpr:
        return FilterExpr({"kind": "is_null", "field": self.name})

    def is_not_null(self) -> FilterExpr:
        return FilterExpr({"kind": "is_not_null", "field": self.name})

    # -- string namespace -------------------------------------------------
    @property
    def str(self) -> StrExpr:
        return StrExpr(self.name)

    def __bool__(self) -> bool:
        raise _ast_usage_error(f"col({self.name!r})")

    def __repr__(self) -> str:
        return f"col({self.name!r})"


def col(name: str) -> FieldExpr:
    """Reference a payload field: ``col("cell line")`` (polars' pl.col)."""
    if not isinstance(name, str) or not name:
        raise DSLUsageError("col() expects a non-empty field name (str)")
    return FieldExpr(name)


def iter_fields(expr: FilterExpr) -> Iterator[str]:
    """All field names referenced by an expression (for schema validation)."""
    kind = expr.node["kind"]
    if kind in ("cmp", "is_in", "is_null", "is_not_null", "str_contains"):
        yield expr.node["field"]
    elif kind in ("and", "or"):
        for part in expr.node["parts"]:
            yield from iter_fields(FilterExpr(part))
    elif kind == "not":
        yield from iter_fields(FilterExpr(expr.node["child"]))
    else:  # pragma: no cover - registration bugs land here
        raise DSLUsageError(f"unknown expression node kind: {kind!r}")
