# Filtering DSL — implementation plan

Status: **implemented** (2026-09-06). Notes vs. plan: class auto-discovery
needs explicit `method_filter=None` (default `()` registers nothing); DSL
"null" maps to Qdrant `IsEmptyCondition` because ingest drops None payload
keys; the executor's `ExecutionError` gets peeled one level so
`UnknownFieldError`/`UnknownViewError` surface as first-class error types.
Everything else is as written below.

### Testing strategy (tiered, 2026-09-06)

- **Tier 1 (default `pytest`)**: hermetic. `tests/test_filtering.py` runs the
  real MCP tool path against a throwaway Qdrant container (port 56333) loaded
  with a deterministic synthetic dataset via the real `ingest` code path;
  expected counts are *computed from the fixture*, never pinned. Skips cleanly
  when no Docker daemon.
- **Tier 2 (`pytest -m live`)**: `tests/test_filtering_live.py` — exact-count
  pins against the full real dataset in the dev stack; fails loudly if run
  while the stack is down.
- The old silent-skip socket probe is gone from both tiers.

### Follow-ups landed

- 2026-09-06: `ViewHandle.meta` — payload fetch by ids / filter / projection
  (plan: `docs/meta-feature-plan.md`; contract: `tests/test_meta.py`).
- 2026-09-06: `ViewHandle.search` — filtered top-k by anchor id; anchor vector
  is fetched and passed explicitly (Qdrant's id-as-query self-excludes, which
  the tests pin otherwise); ANN by default, `exact=True` for deterministic
  brute-force within the filtered set (contract: `tests/test_search.py`).
- 2026-09-06: search signature finalized as `search(among=None|FilterExpr,
  k=10, like=int|list[int]|FilterExpr, columns=None, with_vector=False,
  exact=False)` — `among` names the candidate pool; `like` aggregates multiple
  anchors by centroid (cosine is scale-invariant; no normalization needed).
- 2026-09-06: groupby+aggregation DSL: `db.group_by(by[, among]).agg(col(f).<op>,
  row_count())` (+ `.where(f)` chains) — polars-shaped, evaluated server-side
  with polars so gold-truth comparisons are exact (contract:
  `tests/test_groupby.py`). Handle-reprs guide unfinished chains
  (FilteredView reprs print the valid next steps).
Target: make the three `xfail(strict=True)` tests in
`packages/server/tests/test_filtering.py` pass against the dev stack.

## Driving examples (the pinned contract)

```python
db = database("image")                                      # view handle
ids = db.resolve_ids(col("cell line") == "U2OS")
output({"num_ids": len(ids)})
```

```python
ids = db.resolve_ids((col("cell line") == "U2OS") & (col("umap2d x") > 0))
```

```python
u2os_scored    = (col("cell line") == "U2OS") & (col("umap2d x") > 0)
starts_u_scored = col("cell line").str.contains("U") & (col("umap2d x") > 0)
ids = db.resolve_ids(u2os_scored | starts_u_scored)
```

Filter expressions imitate polars: comparison before `&`/`|` (Python binds
`&`/`|` tighter than `==`/`>`, so comparisons must be parenthesized), string
ops under the `.str` namespace, `==` for equality (polars `.equals` is
expr-metadata equality — not for values).

## Key design decision

Expression objects must **materialize server-side**, not inside the tracer
sandbox. biocircle traces calls against mock stubs, so operator overloads are
registered as **class methods on the SymbolContext** (exen's
`class_context.register_method` pattern); `col("x") == "U2OS"` then traces as
recorded execution steps and the real AST objects get built during
`run_prepared` on the server. No custom object ever crosses the
trace-subprocess wire — avoids the argument-serialization problem entirely.

## Symbols to build

| Symbol | Kind | Signature / behavior |
|---|---|---|
| `database` | DSL function | `database(name) -> ViewHandle`; `"cell"`/`"cells"`, `"image"`/`"images"`; unknown name → `ValueError` listing available views |
| `col` | DSL function | `col(name) -> FieldExpr`; unknown field validated at resolve time with a "did you mean …" key list |
| `FieldExpr` | registered class | `__eq__ __ne__ __gt__ __ge__ __lt__ __le__` → `FilterExpr` (never bool; `__bool__` raises with a "use `&`, `|`, `~` — not `and`/`or`/`not`" hint) |
| `FieldExpr.is_in` | method | `is_in(values: list) -> FilterExpr` |
| `FieldExpr.is_null` / `is_not_null` | methods | → `FilterExpr` (maps to Qdrant is-null / is-empty conditions) |
| `FieldExpr.str` | property → `StrExpr` | `.contains(needle)` this phase |
| `FilterExpr` | registered class | `__and__`, `__or__`, `__invert__` combinable tree; `__bool__` guard |
| `ViewHandle.resolve_ids` | method | `resolve_ids(expr: FilterExpr \| None = None, limit: int = 100_000) -> list[int]` — AST → Qdrant `Filter`, `scroll` collecting ids only (no vectors), limit cap fails loudly |
| `ViewHandle.count` | method | `count(expr: FilterExpr \| None = None) -> int` — one `count()` RPC; cheapest sanity signal |

**Deferred (documented, not built):** `.eq()` named variant (redundant with
`==`), `db.similar(id, k, filter=…)` vector search, cell↔image join helpers
(`cell_path` → `file_prefix` derivation), `.str.starts_with/ends_with/regex`,
plan/explain introspection on `ViewHandle`.

## File management

New `dsl/` package in the server package — one responsibility per module,
biocircle coupling isolated to `registration.py`:

```
packages/server/src/server/
├── core/context.py               # VtdbSymbolContext (gains register_dsl_symbols(ctx) call)
├── dsl/                          # NEW
│   ├── __init__.py               # public re-exports: database, col, ViewHandle, FieldExpr, FilterExpr
│   ├── expressions.py            # FieldExpr/StrExpr/FilterExpr — PURE python AST, zero qdrant imports
│   ├── schema.py                 # payload-field map per view (single source of truth;
│   │                             #   ingest.py switches to it, replacing duplicated ViewSpec keys)
│   ├── views.py                  # database(), ViewHandle.resolve_ids/count
│   ├── qdrant_filter.py          # AST → models.Filter translation (only qdrant_client.models consumer)
│   └── registration.py           # register into SymbolContext (mirror exen executor/registration/core.py)
├── services/
│   ├── pipeline.py               # unchanged (trace→seal→run spine)
│   └── qdrant.py                 # NEW: lazy shared QdrantClient (QDRANT_HOST/PORT/GRPC_PORT env)
tests/
├── test_dsl_expressions.py       # pure AST shape/typing/precedence unit tests
├── test_dsl_qdrant_filter.py     # AST → Filter translation shape tests
├── test_ingest.py                # unchanged except schema import switch
└── test_filtering.py             # integration — drop xfail once green
```

Rationale: `expressions.py` stays pure → fast tests; one translation boundary
(`qdrant_filter.py`); `registration.py` is the only biocircle-coupled file;
`schema.py` kills ingest/query drift on payload keys.

## Build order

1. **Spike first (the risk)**: ~20-line probe proving
   `function_context.registrar` + `class_context.register_method` on a custom
   class survive trace→execute for a tiny `col`-equals sample before
   mass-coding the rest. If scaffold quirks bite, fall back to whatever exen's
   `executor/registration/class_registration.py` does for bound methods.
2. `expressions.py` + unit tests (pure; `__bool__` guard; precedence notes).
3. `qdrant_filter.py` + translation tests.
4. `services/qdrant.py`, `views.py`, `registration.py`; wire into
   `core/context.py`.
5. Integration: dev stack already holds full data. Drop the xfail markers from
   `test_filtering.py`, run live; spot-check against notebook numbers
   (`cell line == "U2OS"` → 25,160 images).
6. Gates when done: `pytest`, `ruff check/format`, `pyright` (src + tests),
   then a live MCP `query` call for each of the three expression shapes.

## Known contract notes (already pinned in tests)

- `==`/`>`/… never return bool for expressions — misuse gets the guided error.
- Errors surface in the standard `{"error": {...}}` payload; unknown fields /
  views list what IS available (biocircle NameError style).
- NaN/Inf in numeric comparisons: out of scope (strict-JSON wire already
  normalizes them to null).
