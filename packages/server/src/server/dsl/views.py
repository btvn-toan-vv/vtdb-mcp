"""View access: database() handle + resolve/count against Qdrant collections.

``database("image")`` → ViewHandle bound to the `images` collection; filters
are polars-shaped FieldExpr trees (dsl/expressions.py) translated server-side
(dsl/qdrant_filter.py). Ids are Qdrant point ids, i.e. row indices of the
views' metadata.csv (and embeddings.npy) — see services/../ingest.py.
"""

from __future__ import annotations

import logging
from typing import Any

from server.dsl.expressions import DSLUsageError, FilterExpr, iter_fields
from server.dsl.qdrant_filter import to_qdrant_filter
from server.dsl.schema import FIELDS, VIEW_ALIASES
from server.services.qdrant import get_client

logger = logging.getLogger(__name__)

_DEFAULT_ID_LIMIT = 100_000
_SCROLL_PAGE = 4_096


class UnknownViewError(ValueError):
    """Raised for `database("nope")` — message lists the available views."""


class UnknownFieldError(ValueError):
    """A filter references a field not in the view's payload schema."""


class ViewHandle:
    """Handle to one view (cells | images) for resolve/count operations."""

    def __init__(self, collection: str) -> None:
        self._collection = collection

    @property
    def collection(self) -> str:
        """Qdrant collection name ("cells" / "images")."""
        return self._collection

    @property
    def fields(self) -> dict[str, str]:
        """Payload schema: field name → python type name."""
        return dict(FIELDS[self._collection])

    def _validate_fields(self, expr: FilterExpr | None) -> None:
        if expr is None:
            return
        unknown = sorted(set(iter_fields(expr)) - set(FIELDS[self._collection]))
        if unknown:
            available = ", ".join(sorted(FIELDS[self._collection]))
            raise UnknownFieldError(
                f"unknown field(s) for view {self._collection!r}: "
                f"{', '.join(unknown)}. Available: {available}"
            )

    def count(self, expr: FilterExpr | None = None) -> int:
        """Number of points matching the filter (exact)."""
        self._validate_fields(expr)
        result = get_client().count(
            collection_name=self._collection,
            count_filter=to_qdrant_filter(expr),
            exact=True,
        )
        return int(result.count)

    def resolve_ids(
        self, expr: FilterExpr | None = None, limit: int = _DEFAULT_ID_LIMIT
    ) -> list[int]:
        """Point ids (metadata row indices) matching the filter, scroll order.

        Vectors are NOT fetched. `limit` guards against accidental full-table
        materialization — hitting it raises instead of truncating silently.
        """
        self._validate_fields(expr)
        filt = to_qdrant_filter(expr)
        ids: list[int] = []
        offset: Any = None
        while True:
            page, offset = get_client().scroll(
                collection_name=self._collection,
                scroll_filter=filt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            # Point ids are ints by ingest construction (row indices).
            ids.extend(int(p.id) for p in page if isinstance(p.id, int))
            if offset is None:
                break
            if len(ids) > limit:
                raise DSLUsageError(
                    f"filter matched more than the id limit ({limit}). "
                    "Tighten the filter or pass limit= explicitly."
                )
        logger.info(
            "resolve_ids %s: %d ids (limit=%d)", self._collection, len(ids), limit
        )
        return ids


def database(name: str) -> ViewHandle:
    """Select a view: ``database("cells")`` / ``database("images")``."""
    if not isinstance(name, str):
        raise UnknownViewError(f"view name must be str, got {type(name).__name__}")
    collection = VIEW_ALIASES.get(name.lower())
    if collection is None:
        choices = ", ".join(sorted(VIEW_ALIASES))
        raise UnknownViewError(f"unknown view {name!r}. Available views: {choices}")
    return ViewHandle(collection)
