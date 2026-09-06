# `ViewHandle.meta` — implementation plan

Status: planned (tests pinned: `packages/server/tests/test_meta.py`, 10 strict-xfail
review tests, committed as caa9158).

## Goal

`db.meta(query, columns=None, limit=10_000) -> list[dict]` — fetch full metadata
rows (payloads) by **int id**, **id list**, **FilterExpr**, or **None** (all
rows, capped by `limit`). Rows = sanitized payload dicts + attached `"id"`
(point id, i.e. the source `metadata.csv` row index).

## Files touched (minimal, existing layout)

| File | Change |
|---|---|
| `src/server/dsl/views.py` | the whole feature: `ViewHandle.meta(...)` + two private helpers |
| `tests/test_meta.py` | drop the module xfail — it becomes the passing contract |
| `api/templates/query.md.j2` | one line: advertise `meta` + tiny example |
| `docs/filtering-dsl-plan.md` | status addendum (meta landed) |
| `packages/server/README.md` | vocabulary line update |

No new modules; **no registration changes** — `ViewHandle` is already
class-registered with `method_filter=None`, so `meta` auto-discovers into the
DSL namespace.

## Behavior matrix (as pinned by tests)

| Input | Path | Semantics |
|---|---|---|
| `meta(5)` / `meta([3,0,7])` | `client.retrieve(ids=..., with_payload=..., with_vectors=False)` | request order preserved (re-index by id; retrieve order isn't guaranteed); missing ids dropped silently |
| `meta(col(...) == ...)` / `meta(None)` | `scroll(..., with_vectors=False, with_payload=..., page 4k)` | scroll order (unpinned); up-front `count()` for the cap |
| `columns=[...]` | → `with_payload=[...]` on both paths | unknown column → `UnknownFieldError` listing available fields; `"id"` stays attached post-projection |
| over limit | count-first for expr/None, `len(ids)` for id path | `DSLUsageError` mentioning `limit`, raised before fetching |
| wrong type (`"U2OS"`) / bare `meta()` | validated up front | `DSLUsageError` listing accepted shapes / Python `TypeError` |

## Sharp corners being fixed deliberately

1. **Order**: retrieve → `{id: record}` map → re-emit in request order (the
   pinned `[3, 0, 7]` case).
2. **`"id"` attachment**: `payload | {"id": int(record_id)}`; ingest payload
   keys never collide with `"id"` (documented in the docstring; if one ever
   did, `id` wins).
3. **Limit check before I/O**: expr/None path does `count()` first — a fat
   filter errors in ~ms instead of burning a scroll.
4. **Projection validation**: one set-diff against the schema map in
   `dsl/schema.py` (`FIELDS[collection]`).

## Build order + verification

1. `views.py` implementation (~60 lines); remove the xfail marker.
2. `pytest tests/test_meta.py` → expect 10/10 after flip; oracle-based exact
   equality shakes out order/type/None-drop drift.
3. Gates: `uv run pytest` (full hermetic suite), `ruff check/format`,
   `pyright src+tests`.
4. Live sanity through the dev server: one `meta([0, 12345])` against real
   `cells`.
5. Template/README/plan-note touch-ups; commit only when asked.

## Out of scope (deferred, docstring-noted)

- live-tier pins for meta (follow-up);
- vector payloads (`with_vectors`) — this is metadata-only by design;
- pagination/continuation tokens.

Tradeoffs chosen (push back if wrong): silent drop on missing ids (not error);
bare `meta()` relies on Python's `TypeError` (tests accept either);
loud-over-limit posture.

Return contract stays `list[dict]` JSON-native. Revisit to a frame-like object
only if vectors-in-row or `.select()` ergonomics get requested.
