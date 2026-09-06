# vtdb-mcp server

A minimal [FastMCP](https://gofastmcp.com) server (streamable-http at `/mcp`,
unauthenticated `/health` next to it), modeled on
`~/exen-mcp/packages/server`. Tools: `query` (runs sandboxed Python via the
biocircle trace→seal→run spine against a `VtdbSymbolContext` with the
filtering DSL registered — `database(view)`, polars-style `col()` filter
expressions, `db.resolve_ids(...)` / `db.count(...)` against Qdrant; see
`docs/filtering-dsl-plan.md`) and `server_info`.

## Run locally

```bash
uv sync                      # needs UV_INDEX_BIOTURING_* from repo .envrc
uv run server                # or: uv run python -m server
# -> http://127.0.0.1:8000/mcp   /health for probes
```

Config via flags or env: `--host/--port/--mcp-path/--debug/--reload`
(`VTDB_HOST`, `VTDB_PORT`, `VTDB_MCP_PATH`, `VTDB_DEBUG`, `VTDB_RELOAD`).

## Tests

```bash
uv run pytest                 # packages/server/tests (no network / no Qdrant)
```

Coverage of the idempotency contract (skip / resume / force-recreate) lives in
`tests/test_ingest.py` against an in-memory fake Qdrant; the tool surface and
`/health` in `tests/test_app.py` via FastMCP's in-memory client + Starlette's
TestClient.

## Full stack (develop + deploy)

`../../compose.yaml` runs the server + Qdrant + Grafana Alloy/Loki/Grafana on
plain Docker — the only orchestration path (no k8s anymore). For development
the source is bind-mounted and `VTDB_RELOAD=1` auto-reloads the worker on edit
(~1s, no image rebuild; rebuild only when `uv.lock` changes):

```bash
docker compose up -d --build
docker compose down                        # add -v to wipe ALL data incl. Qdrant
```

UIs: Grafana http://localhost:3000 ("vtdb-mcp · Logs", admin/admin) · MCP
http://localhost:8000/mcp · Qdrant http://localhost:6333/dashboard.

### Qdrant bootstrap (`server.ingest`)

The one-shot `ingest` service loads the subcellular-embeddings dataset
(`VTDB_DATA_DIR`, default `~/data/subcellular_embeddings`) into two cosine
collections — `cells` (1536-d) and `images` (1024-d) — with payload indexes on
the filterable metadata columns. It is idempotent: re-runs skip views whose
point count already matches the dataset, resume partial loads from the exact
row offset, and only recreate collections on `--force` or schema drift
(dims/distance changed). The `server` service waits for it (`--no-deps` skips
that while a full load runs). Set `VTDB_QDRANT_DATA` to bind-mount Qdrant's
storage somewhere roomy (the full cells view is ~9 GiB).

```bash
docker compose up ingest                   # full / check-only run
VTDB_INGEST_LIMIT=20000 docker compose up ingest   # smoke load
docker compose run --rm ingest --force             # rebuild collections
uv run python -m server.ingest --limit 20000       # local CLI (host ports)
```

## Docker

Context is this directory; the private-index token is a BuildKit secret:

```bash
export UV_INDEX_BIOTURING_PASSWORD=...   # in repo .envrc
docker build \
  --secret id=bioturing_token,env=UV_INDEX_BIOTURING_PASSWORD \
  -f docker/Dockerfile -t vtdb-mcp-server:dev .
```
