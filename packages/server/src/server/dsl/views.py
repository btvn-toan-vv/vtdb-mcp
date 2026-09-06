"""View access: database() handle + resolve/count/meta against Qdrant.

``database("image")`` → ViewHandle bound to the `images` collection; filters
are polars-shaped FieldExpr trees (dsl/expressions.py) translated server-side
(dsl/qdrant_filter.py). Ids are Qdrant point ids, i.e. row indices of the
views' metadata.csv (and embeddings.npy) — see services/../ingest.py.
`meta()` is payload-only (see docs/meta-feature-plan.md for the contract).
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
_DEFAULT_META_LIMIT = 10_000
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

    # -- metadata fetch (docs/meta-feature-plan.md) -------------------------

    def _validate_columns(self, columns: list[str] | None) -> list[str] | None:
        if columns is None:
            return None
        if not isinstance(columns, list) or not all(
            isinstance(c, str) for c in columns
        ):
            raise DSLUsageError("columns= expects a list of field names (str)")
        unknown = sorted(set(columns) - set(FIELDS[self._collection]))
        if unknown:
            available = ", ".join(sorted(FIELDS[self._collection]))
            raise UnknownFieldError(
                f"unknown column(s) for view {self._collection!r}: "
                f"{', '.join(unknown)}. Available: {available}"
            )
        return columns

    @staticmethod
    def _row(record: Any, columns: list[str] | None) -> dict[str, Any]:
        # `id` attaches AFTER the projection: it is the point id, not a payload
        # field, so it's present even under columns= selection.
        payload = dict(record.payload or {})
        if columns is not None:
            payload = {k: v for k, v in payload.items() if k in columns}
        payload["id"] = int(record.id)
        return payload

    def meta(
        self,
        query: FilterExpr | list[int] | int | None,
        columns: list[str] | None = None,
        limit: int = _DEFAULT_META_LIMIT,
    ) -> list[dict[str, Any]]:
        """Full metadata rows for a filter / id list / single id / all rows.

        Rows are payload dicts with the point id attached under ``"id"``.
        Order: id-inputs preserve request order (missing ids are dropped);
        filter/None inputs follow scroll order. `limit` fails loudly BEFORE
        fetching (no silent caps).
        """
        cols = self._validate_columns(columns)

        if isinstance(query, FilterExpr) or query is None:
            self._validate_fields(query)
            filt = to_qdrant_filter(query)
            n = self.count(query)
            if n > limit:
                raise DSLUsageError(
                    f"meta would return {n} rows, over the limit"
                    f" ({limit}). Tighten the filter or pass limit= explicitly."
                )
            rows: list[dict[str, Any]] = []
            offset: Any = None
            while True:
                page, offset = get_client().scroll(
                    collection_name=self._collection,
                    scroll_filter=filt,
                    limit=_SCROLL_PAGE,
                    offset=offset,
                    with_payload=cols if cols is not None else True,
                    with_vectors=False,
                )
                rows.extend(self._row(p, cols) for p in page)
                if offset is None:
                    break
            logger.info(
                "meta %s: %d rows (limit=%d)", self._collection, len(rows), limit
            )
            return rows

        if isinstance(query, int) and not isinstance(query, bool):
            ids = [query]
        elif isinstance(query, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in query
        ):
            ids = query
        else:
            raise DSLUsageError(
                "meta expects an id, a list of ids, a filter expression "
                "(col(...) == ...), or None for all rows"
            )

        if len(ids) > limit:
            raise DSLUsageError(
                f"meta got {len(ids)} ids, over the limit ({limit})."
                " Pass fewer ids or raise limit=."
            )
        records = get_client().retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=cols if cols is not None else True,
            with_vectors=False,
        )
        by_id = {int(r.id): self._row(r, cols) for r in records}
        rows = [by_id[i] for i in ids if i in by_id]  # request order; drops missing
        logger.info("meta %s: %d/%d ids", self._collection, len(rows), len(ids))
        return rows


def database(name: str) -> ViewHandle:
    """Select a view: ``database("cells")`` / ``database("images")``."""
    if not isinstance(name, str):
        raise UnknownViewError(f"view name must be str, got {type(name).__name__}")
    collection = VIEW_ALIASES.get(name.lower())
    if collection is None:
        choices = ", ".join(sorted(VIEW_ALIASES))
        raise UnknownViewError(f"unknown view {name!r}. Available views: {choices}")
    return ViewHandle(collection)
