"""Translate the DSL AST (dsl/expressions.py) into qdrant_client Filter models.

Semantics notes:

- ``is_null`` maps to **IsEmptyCondition**, not IsNullCondition — ingest drops
  None values from payloads, so "null" in this stack means the key is ABSENT.
  ``is_not_null`` is its negation.
- ``cmp`` with ``value=None`` normalizes to is_null / is_not_null.
- ``str_contains`` translates to a full-text MatchText — the field needs a
  ``text`` payload index in Qdrant (our views ship keyword indexes; MatchText
  then only supports word matching on the indexed text).
- ``ne`` is ``must_not [eq]`` (same shape as polars' ``!=``).
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from server.dsl.expressions import DSLUsageError, FilterExpr


def _range_condition(field: str, op: str, value: float) -> models.FieldCondition:
    kwargs = {}
    if op == "gt":
        kwargs["gt"] = value
    elif op == "ge":
        kwargs["gte"] = value
    elif op == "lt":
        kwargs["lt"] = value
    elif op == "le":
        kwargs["lte"] = value
    return models.FieldCondition(key=field, range=models.Range(**kwargs))


def _eq_condition(field: str, value: Any) -> models.FieldCondition:
    # `value` is a Scalar at the AST boundary (expressions._cmp validates);
    # Any here sidesteps qdrant's own union-of-stricts ValueVariants typing.
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def _is_null_condition(field: str) -> models.IsEmptyCondition:
    # Absent-key semantics (see module docstring).
    return models.IsEmptyCondition(is_empty=models.PayloadField(key=field))


def _node_to_filter(node: dict) -> models.Filter:
    kind = node["kind"]
    if kind == "cmp":
        field, op = node["field"], node["op"]
        value = node["value"]
        if value is None:
            inner = models.Filter(must=[_is_null_condition(field)])
            return inner if op == "eq" else models.Filter(must_not=[inner])
        if op == "eq":
            return models.Filter(must=[_eq_condition(field, value)])
        if op == "ne":
            return models.Filter(must_not=[_eq_condition(field, value)])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise DSLUsageError(
                f"col({field!r}) {op} needs a number, got {type(value).__name__}"
            )
        return models.Filter(must=[_range_condition(field, op, value)])
    if kind == "is_in":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=node["field"], match=models.MatchAny(any=node["values"])
                )
            ]
        )
    if kind == "is_null":
        return models.Filter(must=[_is_null_condition(node["field"])])
    if kind == "is_not_null":
        return models.Filter(must_not=[_is_null_condition(node["field"])])
    if kind == "str_contains":
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=node["field"],
                    match=models.MatchText(text=node["needle"]),
                )
            ]
        )
    if kind == "and":
        must: list = []
        for part in node["parts"]:
            f = _node_to_filter(part)
            must.extend(f.must or [])
            if f.should or f.must_not:
                must.append(
                    f
                )  # keep nested structure where flattening breaks semantics
        return models.Filter(must=must)
    if kind == "or":
        should: list = []
        for part in node["parts"]:
            f = _node_to_filter(part)
            if f.should and not (f.must or f.must_not):
                should.extend(f.should)
            else:
                should.append(f)
        return models.Filter(should=should)
    if kind == "not":
        # NOT child  ==  Filter(must_not=[child-as-filter])
        inner = _node_to_filter(node["child"])
        if inner.must and not (inner.should or inner.must_not):
            return models.Filter(must_not=inner.must)
        return models.Filter(must_not=[inner])
    raise DSLUsageError(f"unknown expression node kind: {kind!r}")  # pragma: no cover


def to_qdrant_filter(expr: FilterExpr | None) -> models.Filter | None:
    """AST → models.Filter; None (no filter) passes through as None."""
    if expr is None:
        return None
    if not isinstance(expr, FilterExpr):
        raise DSLUsageError(
            f"expected a filter expression like col('x') == 1, got {type(expr).__name__}"
        )
    return _node_to_filter(expr.node)
