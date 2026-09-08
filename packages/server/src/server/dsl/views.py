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

import numpy as np
import polars as pl
from qdrant_client import models

from server.dsl.aggregations import (
    NUMERIC_ONLY_OPS,
    AggExpr,
    default_alias,
)
from server.dsl.expressions import DSLUsageError, FilterExpr, iter_fields
from server.dsl.qdrant_filter import to_qdrant_filter
from server.dsl.schema import FIELDS, VIEW_ALIASES
from server.dsl.sorting import Rows, as_rows
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
    ) -> Rows:
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
            return as_rows(rows)

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
        return as_rows(rows)

    # -- filtered top-k search (tests/test_search.py pins the contract) ------

    def _scored_row(
        self, point: Any, columns: list[str] | None, with_vector: bool
    ) -> dict[str, Any]:
        row = self._row(point, columns)
        row["score"] = float(point.score)
        if with_vector:
            row["vector"] = list(point.vector)  # dense unnamed vector
        return row

    def _anchor_vector(self, like: int | list[int] | FilterExpr | None) -> list[float]:
        """Resolve `like` to a query vector: anchor point's own vector, or the
        centroid (mean) for a list of ids / a FilterExpr's resolved ids.

        Cosine distance is invariant to scaling of the query vector, so the
        mean needs no normalization.
        """
        if isinstance(like, FilterExpr):
            # Filtered anchor set → centroid. NOT resolve_ids(): its 100k cap
            # exists for user-facing id fetches, while anchor sets on real
            # filters are routinely larger (U2OS cells ≈ 300k). Scroll ids
            # directly, no cap, size logged ahead of the mean.
            self._validate_fields(like)
            filt = to_qdrant_filter(like)
            total = self.count(like)
            logger.info("anchor filter resolved %d ids in %s", total, self._collection)
            ids_from_filter: list[int] = []
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
                ids_from_filter.extend(int(p.id) for p in page)
                if offset is None:
                    break
            like = ids_from_filter
        if isinstance(like, bool):
            ids: list[int] = []
        elif isinstance(like, int):
            ids = [like]
        elif isinstance(like, list) and all(
            isinstance(i, int) and not isinstance(i, bool) for i in like
        ):
            ids = list(like)
        else:
            raise DSLUsageError(
                "like= expects a point id, a list of point ids, or a filter "
                "expression (e.g. col('x') == 1 resolved to ids)"
            )
        if not ids:
            raise DSLUsageError("like= resolved to zero anchor ids")

        records = get_client().retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=False,
            with_vectors=True,
        )
        found = {int(r.id) for r in records}
        missing = [i for i in ids if i not in found]
        if missing:
            raise DSLUsageError(
                f"anchor id(s) {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"not found in view {self._collection!r}"
            )
        vectors = [r.vector for r in records]
        assert all(isinstance(v, list) for v in vectors)  # dense unnamed vecs
        centroid = np.asarray(vectors, dtype=np.float32).mean(axis=0)
        return centroid.tolist()

    def search(
        self,
        like: int | list[int] | FilterExpr | None = None,
        among: FilterExpr | None = None,
        k: int = 10,
        columns: list[str] | None = None,
        with_vector: bool = False,
        exact: bool = False,
    ) -> Rows:
        """Top-k payload rows by similarity to ``like``, **among** a filtered pool.

        ``like`` (first arg) is a point id, a list of point ids (centroid), or
        a FilterExpr (resolved ids → centroid) — the query direction.
        ``among`` is the candidate pool (None = the whole view).
        ``like`` is a point id, a list of point ids (centroid anchor), or a
        FilterExpr (its resolved ids → centroid). Rows are
        ``{"id", "score", ...payload}`` ordered by score DESC. ``exact=True``
        turns off the HNSW approximation (brute-force within the filtered set).
        """
        cols = self._validate_columns(columns)

        among_given = among
        if not (among is None or isinstance(among, FilterExpr)):
            raise DSLUsageError(
                "search's `among` argument must be a filter expression like "
                f"col('x') == 1 (or None = the whole view), got {type(among_given).__name__}"
            )
        self._validate_fields(among)

        if like is None:
            raise DSLUsageError(
                "search needs like=<point id | list of ids | filter expr> — "
                "the anchor vector(s) whose (centroid) direction to rank by"
            )
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise DSLUsageError(f"k must be a positive int, got {k!r}")

        anchor_vec = self._anchor_vector(like)
        client = get_client()

        hits = client.query_points(
            collection_name=self._collection,
            query=anchor_vec,  # the anchor's own vector (see docstring)
            query_filter=to_qdrant_filter(among),
            limit=k,
            with_payload=cols if cols is not None else True,
            with_vectors=with_vector,
            # NOTE: the client kwarg is search_params (REST field: params).
            search_params=models.SearchParams(exact=True) if exact else None,
        ).points
        rows = [self._scored_row(p, cols, with_vector) for p in hits]
        logger.info(
            "search %s: %d hits (k=%d, like=%d)", self._collection, len(rows), k, like
        )
        return as_rows(rows)

    # -- grouping entry point ----------------------------------------------

    def group_by(
        self, by: str | list[str], among: FilterExpr | None = None, limit: int = 1000
    ) -> GroupBy:
        """Group rows by one or more payload fields; .agg(...) then aggregates."""
        if isinstance(by, str):
            by = [by]
        if (
            not isinstance(by, list)
            or not by
            or not all(isinstance(b, str) for b in by)
        ):
            raise DSLUsageError(
                "group_by() expects a field name or a non-empty list of them"
            )
        return GroupBy(self, by, among=among, limit=limit)

    def where(self, expr: FilterExpr) -> FilteredView:
        """Filter this view for the next read — x.where(f).group_by(...)."""
        if not isinstance(expr, FilterExpr):
            raise DSLUsageError(
                "where() expects a filter expression like col('x') == 1"
            )
        return FilteredView(self, expr)


def _compose_and(a: FilterExpr | None, b: FilterExpr | None) -> FilterExpr | None:
    if a is None:
        return b
    if b is None:
        return a
    return FilterExpr({"kind": "and", "parts": [a.node, b.node]})


class FilteredView:
    """A view handle with a permanent pre-filter (from ``db.where(expr)``).

    Every read method forwards to the underlying handle with the stored filter
    ANDed onto the call-local one. ``group_by`` is the star: where-then-group.
    """

    def __init__(self, handle: ViewHandle, expr: FilterExpr) -> None:
        self._handle = handle
        self._filter = expr

    @property
    def collection(self) -> str:
        return self._handle.collection

    def where(self, expr: FilterExpr) -> FilteredView:
        if not isinstance(expr, FilterExpr):
            raise DSLUsageError(
                "where() expects a filter expression like col('x') == 1"
            )
        combined = _compose_and(self._filter, expr)
        assert combined is not None
        return FilteredView(self._handle, combined)

    def count(self, expr: FilterExpr | None = None) -> int:
        return self._handle.count(_compose_and(self._filter, expr))

    def resolve_ids(
        self, expr: FilterExpr | None = None, limit: int = _DEFAULT_ID_LIMIT
    ) -> list[int]:
        return self._handle.resolve_ids(_compose_and(self._filter, expr), limit=limit)

    def meta(
        self,
        query: FilterExpr | list[int] | int | None = None,
        columns: list[str] | None = None,
        limit: int = _DEFAULT_META_LIMIT,
    ) -> Rows:
        """meta(None)/meta(filter) apply the where filter (ANDed); meta(ids) is
        rejected — id fetches don't compose meaningfully with a prefilter."""
        if isinstance(query, FilterExpr):
            return self._handle.meta(
                _compose_and(self._filter, query), columns=columns, limit=limit
            )
        if query is None:
            return self._handle.meta(self._filter, columns=columns, limit=limit)
        raise DSLUsageError(
            "meta() by id(s) can't combine with where(); pass a filter instead "
            "of ids (or call meta on the unfiltered database() handle)"
        )

    def group_by(
        self, by: str | list[str], among: FilterExpr | None = None, limit: int = 1000
    ) -> GroupBy:
        """Group the pre-filtered view (and compose any call-local prefilter)."""
        return self._handle.group_by(
            by, among=_compose_and(self._filter, among), limit=limit
        )

    def __repr__(self) -> str:
        return (
            f"FilteredView({self.collection}, filter={self._filter.node!r})"
            " — finish with .group_by(...).agg(...), .count(), .resolve_ids(),"
            " .meta(...),  or another .where(...)"
        )


class GroupBy:
    """Concrete pending-aggregation state: handle + by-keys + composed filter."""

    def __init__(
        self,
        handle: ViewHandle,
        by: list[str],
        among: FilterExpr | None,
        limit: int,
    ) -> None:
        self._handle = handle
        self._by = by
        self._among = among
        self._limit = limit

    def agg(self, *exprs: AggExpr) -> Rows:
        """Aggregate each group: ``.agg(col("t").mean(), row_count())``."""
        if not exprs:
            raise DSLUsageError(
                "agg() needs at least one aggregation: col('x').mean(), "
                "row_count(), ..."
            )
        for e in exprs:
            if not isinstance(e, AggExpr):
                raise DSLUsageError(
                    "agg() takes aggregation expressions (col('x').mean(), "
                    f"row_count()), got {type(e).__name__}"
                )

        handle = self._handle
        fields = FIELDS[handle.collection]

        def _known(name: str | None) -> None:
            if name is not None and name not in fields:
                available = ", ".join(sorted(fields))
                raise UnknownFieldError(
                    f"unknown field for view {handle.collection!r}: {name}. "
                    f"Available: {available}"
                )

        for key in self._by:
            _known(key)

        jobs: list[tuple[str, dict[str, Any]]] = []
        seen_names: set[str] = set()
        for e in exprs:
            node = e.node
            job_name = node.get("alias") or (
                "count"
                if node["op"] == "row_count"
                else default_alias(
                    node["field"], node["op"], node.get("params", {}).get("q")
                )
            )
            if job_name in seen_names:
                raise DSLUsageError(
                    f"duplicate output column {job_name!r} in agg() — "
                    "rename one with .alias('...')"
                )
            seen_names.add(job_name)
            if node["op"] != "row_count":
                _known(node["field"])
                kind = fields[node["field"]]
                if node["op"] in NUMERIC_ONLY_OPS and kind not in ("int", "float"):
                    raise DSLUsageError(
                        f"aggregation {node['op']} needs a numeric field; "
                        f"{node['field']!r} is {kind}"
                    )
            jobs.append((job_name, node))

        # Fetch only what aggregation needs: by-keys + aggregated fields.
        payload_fields = set(self._by) | {j[1]["field"] for j in jobs if j[1]["field"]}
        filt = to_qdrant_filter(self._among)
        client = get_client()
        records: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            page, offset = client.scroll(
                collection_name=handle.collection,
                scroll_filter=filt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=sorted(payload_fields) if payload_fields else True,
                with_vectors=False,
            )
            for p in page:
                records.append(dict(p.payload or {}))
            if offset is None:
                break

        df = pl.DataFrame(records) if records else None
        if df is None or df.is_empty():
            out: Rows = Rows()
            logger.info("groupby %s on %s: 0 rows", self._by, handle.collection)
            return out

        pl_exprs: list[Any] = []
        for job_name, node in jobs:
            if node["op"] == "row_count":
                pl_exprs.append(pl.len().alias(job_name))
                continue
            f = node["field"]
            op = node["op"]
            q = node.get("params", {}).get("q")
            c = pl.col(f)
            pl_expr = (
                c.count()
                if op == "count"
                else c.sum()
                if op == "sum"
                else c.mean()
                if op == "mean"
                else c.median()
                if op == "median"
                else c.min()
                if op == "min"
                else c.max()
                if op == "max"
                else c.std()
                if op == "std"
                else c.var()
                if op == "var"
                else c.quantile(q, interpolation="nearest")
                if op == "quantile"
                else c.first()
                if op == "first"
                else c.last()
                if op == "last"
                else c.n_unique()
                if op == "n_unique"
                else None
            )
            assert pl_expr is not None, node
            pl_exprs.append(pl_expr.alias(job_name))

        grouped = (
            df.group_by(self._by)
            .agg(*pl_exprs)
            # ascending, nulls first (polars sort kwarg is nulls_last)
            .sort(self._by, nulls_last=False)
        )
        n_groups = grouped.height
        if n_groups > self._limit:
            raise DSLUsageError(
                f"group_by produced {n_groups} groups, over the limit"
                f" ({self._limit}). Tighten with among=/where() or pass limit=."
            )
        logger.info(
            "groupby %s on %s: %d groups from %d rows",
            self._by,
            handle.collection,
            n_groups,
            df.height,
        )
        return as_rows(grouped.to_dicts())


def database(name: str) -> ViewHandle:
    """Select a view: ``database("cells")`` / ``database("images")``."""
    if not isinstance(name, str):
        raise UnknownViewError(f"view name must be str, got {type(name).__name__}")
    collection = VIEW_ALIASES.get(name.lower())
    if collection is None:
        choices = ", ".join(sorted(VIEW_ALIASES))
        raise UnknownViewError(f"unknown view {name!r}. Available views: {choices}")
    return ViewHandle(collection)
